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
        episode = self.worker.run_episode(task.payload)
        return self.trajectory_adapter.from_episode(task, episode)


def make_ray_rollout_actor(actor_options: dict[str, Any] | None = None):
    ray = _require_ray()
    actor_options = dict(actor_options or {})
    return ray.remote(**actor_options)(RayRolloutActor) if actor_options else ray.remote(RayRolloutActor)


class RayRolloutManager(RolloutManager):
    """Thin Ray actor pool for async rollout production."""

    def __init__(self, config: RolloutManagerConfig | None = None) -> None:
        self.config = config or RolloutManagerConfig()
        self._actors: list[Any] = []
        self._inflight: list[Any] = []
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
        capacity = max(1, self.config.num_workers * self.config.max_inflight_per_worker)
        if len(self._inflight) >= capacity:
            return False
        actor = self._actors[self._next_actor]
        self._next_actor = (self._next_actor + 1) % len(self._actors)
        self._inflight.append(actor.run.remote(task))
        return True

    async def submit_async(self, task: RolloutTask) -> bool:
        return await asyncio.to_thread(self.submit, task)

    def poll_completed(self, timeout_s: float | None = None) -> list[RewardedTrajectory]:
        if not self._inflight:
            return []
        ray = _require_ray()
        timeout = self.config.poll_timeout_s if timeout_s is None else timeout_s
        ready, remaining = ray.wait(self._inflight, num_returns=len(self._inflight), timeout=timeout)
        self._inflight = list(remaining)
        return list(ray.get(ready)) if ready else []

    async def poll_completed_async(self, timeout_s: float | None = None) -> list[RewardedTrajectory]:
        return await asyncio.to_thread(self.poll_completed, timeout_s)

    @property
    def inflight(self) -> int:
        return len(self._inflight)


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
