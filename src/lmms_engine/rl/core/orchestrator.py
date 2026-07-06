from __future__ import annotations

from collections import deque
from typing import Deque

from lmms_engine.rl.config import RLRunConfig
from lmms_engine.rl.core.factory import DEFAULT_RL_COMPONENT_FACTORY, RLComponentFactory
from lmms_engine.rl.core.interfaces import (
    BatchBuilder,
    DataBuffer,
    RolloutManager,
    WeightSyncClient,
)
from lmms_engine.rl.protocol import (
    ModelVersion,
    RewardedTrajectory,
    RolloutTask,
    TrainBatch,
)


class RLOrchestrator:
    """High-level coordinator for the RL MVP components.

    The orchestrator is intentionally policy-agnostic. Algorithms own
    TrainBatchAdapter/trainer logic; this class only moves data across the
    rollout-buffer-training-weight-sync boundaries.
    """

    def __init__(
        self,
        config: RLRunConfig | None = None,
        component_factory: RLComponentFactory | None = None,
        rollout_manager: RolloutManager | None = None,
        data_buffer: DataBuffer | None = None,
        batch_builder: BatchBuilder | None = None,
        weight_sync: WeightSyncClient | None = None,
    ) -> None:
        self.config = config or RLRunConfig()
        self.component_factory = component_factory or DEFAULT_RL_COMPONENT_FACTORY
        self.rollout_manager = rollout_manager or self.component_factory.build_rollout_manager(self.config.rollout)
        self.data_buffer = data_buffer or self.component_factory.build_data_buffer(self.config.data_buffer)
        self.batch_builder = batch_builder or self.component_factory.build_batch_builder(self.config.training)
        self.weight_sync = weight_sync or self.component_factory.build_weight_sync(self.config.vllm)
        self._completed_backlog: Deque[RewardedTrajectory] = deque()

    def start(self) -> None:
        self.rollout_manager.start()

    def submit_rollout(self, task: RolloutTask) -> bool:
        self._apply_backpressure()
        if self._should_pause_rollout_submission():
            return False
        return self.rollout_manager.submit(task)

    def drain_rollouts(self, timeout_s: float | None = None) -> int:
        accepted = self._drain_completed_backlog()
        free_capacity = self._data_buffer_free_capacity()
        if free_capacity > 0 and not self._completed_backlog:
            completed = self.rollout_manager.poll_completed(
                timeout_s=timeout_s,
                max_trajectories=free_capacity,
            )
            self._completed_backlog.extend(completed)
            accepted += self._drain_completed_backlog()
        self._apply_backpressure()
        return accepted

    def next_train_batch(self) -> TrainBatch | None:
        return self.data_buffer.pop_train_batch()

    def build_train_batch(self, trajectories, model_version: ModelVersion | None) -> TrainBatch:
        return self.batch_builder.build(list(trajectories), model_version)

    def reload_policy_weights(self, model_version: ModelVersion) -> dict:
        return self.weight_sync.reload_weights(model_version)

    def _apply_backpressure(self) -> None:
        if not self.config.rollout.pause_on_buffer_high_watermark:
            return
        if self._should_pause_rollout_submission():
            self.rollout_manager.pause()
        else:
            self.rollout_manager.resume()

    def _should_pause_rollout_submission(self) -> bool:
        return bool(self._completed_backlog) or self.data_buffer.should_pause_rollout()

    def _drain_completed_backlog(self) -> int:
        accepted = 0
        while self._completed_backlog and self._data_buffer_free_capacity() > 0:
            trajectory = self._completed_backlog.popleft()
            try:
                self.data_buffer.push(trajectory)
            except BufferError:
                self._completed_backlog.appendleft(trajectory)
                break
            accepted += 1
        return accepted

    def _data_buffer_free_capacity(self) -> int:
        stats = self.data_buffer.stats()
        return max(0, self.config.data_buffer.max_trajectories - stats.pending_trajectories)
