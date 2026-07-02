from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Version tag carried by rollout samples and training batches."""

    version_id: int
    checkpoint_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutTask:
    """A unit of rollout work scheduled by the RolloutManager."""

    task_id: str
    payload: Any
    model_version: ModelVersion
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryStep:
    """One environment/model transition in a rewarded trajectory."""

    observation: Any = None
    request: Any = None
    response: Any = None
    action: Any = None
    reward: float | dict[str, float] | None = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RewardedTrajectory:
    """Complete rollout payload consumed by the independent DataBuffer."""

    trajectory_id: str
    task_id: str
    model_version: ModelVersion
    steps: list[TrajectoryStep] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    final_state: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        total = 0.0
        for step in self.steps:
            if isinstance(step.reward, (float, int)):
                total += float(step.reward)
        return total


@dataclass(slots=True)
class TrainBatch:
    """Fixed global batch boundary between DataBuffer and TrainingEngine."""

    batch_id: str
    model_version: ModelVersion | None
    trajectories: list[RewardedTrajectory]
    global_batch_size: int
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BufferStats:
    pending_trajectories: int
    pending_steps: int
    high_watermark: int
    low_watermark: int
    paused: bool
