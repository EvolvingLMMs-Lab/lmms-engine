from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lmms_engine.rl.protocol import (
    BufferStats,
    ModelVersion,
    RewardedTrajectory,
    RolloutTask,
    TrainBatch,
)


class TrajectoryAdapter(ABC):
    """Converts rollout-engine output into engine-side rewarded trajectories."""

    @abstractmethod
    def from_episode(self, task: RolloutTask, episode: Any) -> RewardedTrajectory:
        raise NotImplementedError


class DataBuffer(ABC):
    """Producer-consumer boundary between rollout and training."""

    @abstractmethod
    def push(self, trajectory: RewardedTrajectory) -> BufferStats:
        raise NotImplementedError

    @abstractmethod
    def pop_train_batch(self) -> TrainBatch | None:
        raise NotImplementedError

    @abstractmethod
    def should_pause_rollout(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> BufferStats:
        raise NotImplementedError


class RolloutManager(ABC):
    """Schedules rollout work and returns completed trajectories."""

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def pause(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def resume(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def submit(self, task: RolloutTask) -> bool:
        raise NotImplementedError

    @abstractmethod
    def poll_completed(
        self,
        timeout_s: float | None = None,
        max_trajectories: int | None = None,
    ) -> list[RewardedTrajectory]:
        raise NotImplementedError

    @property
    @abstractmethod
    def inflight(self) -> int:
        raise NotImplementedError


class BatchBuilder(ABC):
    """Builds TrainBatch objects from buffered trajectories."""

    @abstractmethod
    def build(self, trajectories: list[RewardedTrajectory], model_version: ModelVersion | None) -> TrainBatch:
        raise NotImplementedError


class TrajectoryAnnotator(ABC):
    """Adds trainer-side signals to completed rollout trajectories."""

    @abstractmethod
    def annotate(self, trajectories: list[RewardedTrajectory]) -> list[RewardedTrajectory]:
        raise NotImplementedError


class TrainBatchAdapter(ABC):
    """Converts TrainBatch into tensors consumed by a trainer implementation."""

    @abstractmethod
    def to_trainer_batch(self, batch: TrainBatch) -> Any:
        raise NotImplementedError


class TrainerBridge(ABC):
    """Training-engine boundary used by the RL orchestration loop."""

    @abstractmethod
    def train_batch(self, batch: TrainBatch) -> dict[str, Any]:
        raise NotImplementedError


class WeightSyncClient(ABC):
    """Publishes or reloads policy weights after trainer updates."""

    @abstractmethod
    def reload_weights(self, model_version: ModelVersion) -> dict[str, Any]:
        raise NotImplementedError
