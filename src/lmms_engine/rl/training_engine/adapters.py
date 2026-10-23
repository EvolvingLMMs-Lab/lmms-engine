from lmms_engine.rl.core.interfaces import TrainBatchAdapter


class PassthroughTrainBatchAdapter(TrainBatchAdapter):
    """Development adapter that forwards TrainBatch unchanged."""

    def to_trainer_batch(self, batch):
        return batch


__all__ = [
    "BatchBuilder",
    "FSDP2RLTrainerBridge",
    "FixedGlobalBatchBuilder",
    "GRPOBatchAdapter",
    "GRPOConfig",
    "GRPOPayload",
    "PassthroughTrainBatchAdapter",
    "TrainBatchAdapter",
]


def __getattr__(name):
    if name in {"BatchBuilder", "FixedGlobalBatchBuilder"}:
        from lmms_engine.train.rl import batch

        return getattr(batch, name)
    if name == "FSDP2RLTrainerBridge":
        from lmms_engine.train.rl import bridge

        return getattr(bridge, name)
    if name in {"GRPOBatchAdapter", "GRPOConfig", "GRPOPayload"}:
        from lmms_engine.train.rl import grpo

        return getattr(grpo, name)
    raise AttributeError(name)
