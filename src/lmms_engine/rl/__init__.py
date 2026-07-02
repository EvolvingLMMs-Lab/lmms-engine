"""RL orchestration scaffolding for LMMs-Engine.

The MVP keeps rollout execution in lmms-eval and owns only the distributed
orchestration, data buffering, training-batch boundary, and weight-sync boundary.
"""

from lmms_engine.rl.config import (
    DataBufferConfig,
    RLRunConfig,
    RolloutManagerConfig,
    TrainingEngineConfig,
    VLLMServerConfig,
)
from lmms_engine.rl.core.factory import DEFAULT_RL_COMPONENT_FACTORY, RLComponentFactory
from lmms_engine.rl.core.interfaces import (
    BatchBuilder as BatchBuilderBase,
    DataBuffer,
    RolloutManager,
    TrajectoryAdapter,
    TrainBatchAdapter as TrainBatchAdapterBase,
    TrainerBridge,
    WeightSyncClient,
)
from lmms_engine.rl.core.orchestrator import RLOrchestrator
from lmms_engine.rl.data_buffer import InMemoryDataBuffer
from lmms_engine.rl.lmms_eval import LMMSEvalTrajectoryAdapter, trajectory_from_lmms_eval_episode
from lmms_engine.rl.protocol import (
    BufferStats,
    ModelVersion,
    RewardedTrajectory,
    RolloutTask,
    TrainBatch,
    TrajectoryStep,
)
from lmms_engine.rl.rollout_manager import RayRolloutActor, RayRolloutManager, make_ray_rollout_actor
from lmms_engine.rl.training_engine import VLLMWeightSyncClient

__all__ = [
    "BatchBuilder",
    "BatchBuilderBase",
    "BufferStats",
    "DataBufferConfig",
    "DataBuffer",
    "DEFAULT_RL_COMPONENT_FACTORY",
    "FSDP2RLTrainerBridge",
    "FixedGlobalBatchBuilder",
    "GRPOBatchAdapter",
    "GRPOConfig",
    "GRPOPayload",
    "InMemoryDataBuffer",
    "LMMSEvalTrajectoryAdapter",
    "ModelVersion",
    "PassthroughTrainBatchAdapter",
    "RLRunConfig",
    "RayRolloutActor",
    "RayRolloutManager",
    "RLComponentFactory",
    "RLOrchestrator",
    "RewardedTrajectory",
    "RolloutManager",
    "RolloutManagerConfig",
    "RolloutTask",
    "TrainBatch",
    "TrainBatchAdapter",
    "TrainBatchAdapterBase",
    "TrajectoryAdapter",
    "TrainerBridge",
    "TrainingEngineConfig",
    "TrajectoryStep",
    "VLLMServerConfig",
    "VLLMWeightSyncClient",
    "WeightSyncClient",
    "make_ray_rollout_actor",
    "trajectory_from_lmms_eval_episode",
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
        from lmms_engine.rl import training_engine

        return getattr(training_engine, name)
    raise AttributeError(name)
