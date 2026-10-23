from __future__ import annotations

from lmms_engine.rl.config import TrainingEngineConfig
from lmms_engine.rl.core.interfaces import BatchBuilder as BatchBuilderBase
from lmms_engine.rl.protocol import ModelVersion, RewardedTrajectory, TrainBatch


class FixedGlobalBatchBuilder(BatchBuilderBase):
    """Build a fixed global TrainBatch from rewarded trajectories."""

    def __init__(self, config: TrainingEngineConfig | None = None) -> None:
        self.config = config or TrainingEngineConfig()

    def build(self, trajectories: list[RewardedTrajectory], model_version: ModelVersion | None) -> TrainBatch:
        if self.config.global_batch_size is None:
            raise ValueError("TrainingEngineConfig.global_batch_size must be resolved before building TrainBatch.")
        return TrainBatch(
            batch_id=f"train-batch-{model_version.version_id if model_version else 'unknown'}",
            model_version=model_version,
            trajectories=trajectories,
            global_batch_size=self.config.global_batch_size,
            metadata={"trainer_type": self.config.trainer_type},
        )


BatchBuilder = FixedGlobalBatchBuilder
