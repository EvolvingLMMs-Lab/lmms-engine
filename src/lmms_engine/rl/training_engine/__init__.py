"""Training Engine boundary for RL orchestration.

Algorithm implementations live in `lmms_engine.train.rl`; this package exposes
the training-engine-facing adapters plus policy weight synchronization.
"""

from lmms_engine.rl.training_engine.weight_sync import (
    RayActorWeightSyncClient,
    VLLMWeightSyncClient,
)

__all__ = [
    "BatchBuilder",
    "FSDP2RLTrainerBridge",
    "FixedGlobalBatchBuilder",
    "GRPOBatchAdapter",
    "GRPOConfig",
    "GRPOPayload",
    "PassthroughTrainBatchAdapter",
    "TrainBatchAdapter",
    "RayActorWeightSyncClient",
    "VLLMWeightSyncClient",
]


def __getattr__(name):
    if name in {
        "BatchBuilder",
        "FSDP2RLTrainerBridge",
        "FixedGlobalBatchBuilder",
        "GRPOBatchAdapter",
        "GRPOConfig",
        "GRPOPayload",
        "PassthroughTrainBatchAdapter",
        "TrainBatchAdapter",
    }:
        from lmms_engine.rl.training_engine import adapters

        return getattr(adapters, name)
    raise AttributeError(name)
