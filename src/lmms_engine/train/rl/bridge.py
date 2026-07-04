from __future__ import annotations

from typing import Any

from lmms_engine.rl.core.interfaces import TrainBatchAdapter, TrainerBridge
from lmms_engine.rl.protocol import TrainBatch


class FSDP2RLTrainerBridge(TrainerBridge):
    """Boundary between RL orchestration and an FSDP2-only trainer."""

    def __init__(self, trainer: Any, batch_adapter: TrainBatchAdapter) -> None:
        self.trainer = trainer
        self.batch_adapter = batch_adapter

    def train_batch(self, batch: TrainBatch) -> dict[str, Any]:
        trainer_batch = self.batch_adapter.to_trainer_batch(batch)
        prepare_batch = getattr(self.trainer, "prepare_rl_batch", None)
        if prepare_batch is not None:
            trainer_batch = prepare_batch(trainer_batch)
        train_batch = getattr(self.trainer, "train_batch", None)
        if train_batch is None:
            raise NotImplementedError("The selected trainer must expose train_batch(...) for RL.")
        return dict(train_batch(trainer_batch) or {})
