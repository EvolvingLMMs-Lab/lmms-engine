from __future__ import annotations

import asyncio
from importlib import import_module
from typing import Any

from lmms_engine.rl.config import RolloutManagerConfig
from lmms_engine.rl.core.interfaces import RolloutManager
from lmms_engine.rl.lmms_eval import LMMSEvalTrajectoryAdapter
from lmms_engine.rl.protocol import RewardedTrajectory, RolloutTask


def _require_ray():
    try:
        import ray
    except ImportError as exc:
        raise ImportError("Ray is required for RL rollout orchestration. Install `lmms_engine[rl]`.") from exc
    return ray


class RayRolloutActor:
    """Ray actor body that owns one synchronous lmms-eval rollout worker."""

    def __init__(
        self,
        worker_config: dict[str, Any] | None = None,
        trajectory_adapter: Any = None,
    ) -> None:
        from lmms_eval.agentic.rollout import SyncEpisodeRolloutWorker

        self.worker = SyncEpisodeRolloutWorker(**(worker_config or {}))
        self.trajectory_adapter = _build_trajectory_adapter(trajectory_adapter)

    def run(self, task: RolloutTask) -> RewardedTrajectory:
        return self.run_batch([task])[0]

    def run_batch(self, tasks: list[RolloutTask]) -> list[RewardedTrajectory]:
        if not tasks:
            return []
        if len(tasks) == 1:
            episode = self.worker.run_episode(tasks[0].payload)
            return [self.trajectory_adapter.from_episode(tasks[0], episode)]
        try:
            episodes = self._run_batched_episodes(tasks)
        except (AttributeError, TypeError):
            episodes = [self.worker.run_episode(task.payload) for task in tasks]
        return [
            self.trajectory_adapter.from_episode(task, episode) for task, episode in zip(tasks, episodes, strict=True)
        ]

    def _run_batched_episodes(self, tasks: list[RolloutTask]) -> list[Any]:
        from lmms_eval.agentic.loop.manager import LoopManager, RolloutJob

        jobs = []
        shared_model_server = None
        for index, task in enumerate(tasks):
            components = self.worker.component_builder.build(task.payload)
            if shared_model_server is None:
                shared_model_server = components.model_server
            loop_worker = self.worker.component_builder.factory.build_loop_worker(
                task.payload.loop_worker,
                model_server=components.model_server,
                env_manager=components.env_manager,
                observation_parser=components.observation_parser,
                model_output_parser=components.model_output_parser,
                action_parser=components.action_parser,
                max_steps=task.payload.max_steps,
                generation_kwargs=task.payload.generation_kwargs,
                request_metadata=task.payload.request_metadata,
            )
            jobs.append(
                RolloutJob(
                    index=index,
                    make_session=lambda _model_server, task=task, loop_worker=loop_worker: loop_worker.new_session(
                        task.payload.doc,
                        seed=task.payload.seed,
                        agent_id=task.payload.agent_id,
                    ),
                    run_serial=lambda _model_server, task=task: self.worker.run_episode(task.payload),
                )
            )
        return LoopManager(max_workers=len(jobs)).run_jobs(jobs, shared_model_server)


def make_ray_rollout_actor(actor_options: dict[str, Any] | None = None):
    ray = _require_ray()
    actor_options = dict(actor_options or {})
    actor_options.setdefault("scheduling_strategy", "DEFAULT")
    return ray.remote(**actor_options)(RayRolloutActor)


class RayRolloutManager(RolloutManager):
    """Thin Ray actor pool for async rollout production."""

    def __init__(self, config: RolloutManagerConfig | None = None) -> None:
        self.config = config or RolloutManagerConfig()
        self._actors: list[Any] = []
        self._inflight: list[tuple[Any, int]] = []
        self._pending: list[RolloutTask] = []
        self._next_actor = 0
        self._paused = False

    def start(self) -> None:
        if self._actors:
            return
        actor_cls = make_ray_rollout_actor(self.config.actor_options)
        self._actors = [
            actor_cls.remote(self.config.worker_config, self.config.trajectory_adapter)
            for _ in range(self.config.num_workers)
        ]

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def submit(self, task: RolloutTask) -> bool:
        if self._paused:
            return False
        if not self._actors:
            self.start()
        capacity = max(
            1,
            self.config.num_workers * self.config.max_inflight_per_worker * max(1, int(self.config.batch_size)),
        )
        if self.inflight >= capacity:
            return False
        self._pending.append(task)
        if len(self._pending) >= max(1, int(self.config.batch_size)):
            self._flush_pending()
        return True

    async def submit_async(self, task: RolloutTask) -> bool:
        return await asyncio.to_thread(self.submit, task)

    def poll_completed(
        self,
        timeout_s: float | None = None,
        max_trajectories: int | None = None,
    ) -> list[RewardedTrajectory]:
        if max_trajectories is not None and max_trajectories <= 0:
            return []
        self._flush_pending()
        if not self._inflight:
            return []
        ray = _require_ray()
        timeout = self.config.poll_timeout_s if timeout_s is None else timeout_s
        refs = [ref for ref, _count in self._inflight]
        num_returns = self._max_ready_refs(max_trajectories)
        ready, remaining = ray.wait(refs, num_returns=num_returns, timeout=timeout)
        remaining_ids = set(remaining)
        self._inflight = [(ref, count) for ref, count in self._inflight if ref in remaining_ids]
        if not ready:
            return []
        completed = ray.get(ready)
        return [trajectory for batch in completed for trajectory in batch]

    async def poll_completed_async(
        self,
        timeout_s: float | None = None,
        max_trajectories: int | None = None,
    ) -> list[RewardedTrajectory]:
        return await asyncio.to_thread(self.poll_completed, timeout_s, max_trajectories)

    @property
    def inflight(self) -> int:
        return len(self._pending) + sum(count for _ref, count in self._inflight)

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        if not self._actors:
            self.start()
        batch_size = max(1, int(self.config.batch_size))
        while self._pending:
            batch = self._pending[:batch_size]
            del self._pending[:batch_size]
            actor = self._actors[self._next_actor]
            self._next_actor = (self._next_actor + 1) % len(self._actors)
            self._inflight.append((actor.run_batch.remote(batch), len(batch)))

    def _max_ready_refs(self, max_trajectories: int | None) -> int:
        if max_trajectories is None:
            return len(self._inflight)

        selected_refs = 0
        selected_trajectories = 0
        for _ref, count in self._inflight:
            if selected_refs > 0 and selected_trajectories + count > max_trajectories:
                break
            selected_refs += 1
            selected_trajectories += count
            if selected_trajectories >= max_trajectories:
                break
        return max(1, selected_refs)


def _build_trajectory_adapter(spec: Any):
    if spec is None:
        return LMMSEvalTrajectoryAdapter()
    if hasattr(spec, "from_episode"):
        return spec
    if isinstance(spec, str):
        cls = _import_from_path(spec)
        return cls()
    if callable(spec):
        return spec()
    raise TypeError(f"Unsupported trajectory adapter spec: {type(spec).__name__}")


def _import_from_path(path: str) -> Any:
    module_name, sep, attr = path.partition(":")
    if not sep:
        module_name, sep, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Import path must be 'module:attribute' or 'module.attribute', got {path!r}")
    module = import_module(module_name)
    return getattr(module, attr)
