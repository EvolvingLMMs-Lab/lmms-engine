from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lmms_engine.rl.core.interfaces import TrainBatchAdapter
from lmms_engine.rl.protocol import RewardedTrajectory, TrainBatch


@dataclass(slots=True)
class GRPOConfig:
    """GRPO trainer-side adapter knobs.

    This is intentionally only the boundary config. Tensorization, advantage
    computation, old-logprob plumbing, and model-specific loss details should be
    implemented behind `GRPOBatchAdapter`.
    """

    group_size: int = 1
    beta: float = 0.0
    clip_range_low: float = 0.2
    clip_range_high: float = 0.2
    normalize_advantages: bool = True
    advantage_clip: float | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GRPOPayload:
    """Algorithm-specific payload passed from TrainBatchAdapter to trainer."""

    trajectories: list[RewardedTrajectory]
    config: GRPOConfig
    tensors: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GRPOBatchAdapter(TrainBatchAdapter):
    """Convert engine TrainBatch into a GRPO trainer payload."""

    def __init__(self, config: GRPOConfig | None = None) -> None:
        self.config = config or GRPOConfig()

    def to_trainer_batch(self, batch: TrainBatch) -> GRPOPayload:
        return GRPOPayload(
            trajectories=batch.trajectories,
            config=self.config,
            metadata={
                **batch.metadata,
                "algorithm": "grpo",
                "batch_id": batch.batch_id,
                "model_version": batch.model_version.version_id if batch.model_version else None,
            },
        )
