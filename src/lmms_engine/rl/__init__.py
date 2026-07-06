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
    resolve_train_batch_size_per_gpu,
)
from lmms_engine.rl.core.factory import DEFAULT_RL_COMPONENT_FACTORY, RLComponentFactory
from lmms_engine.rl.core.interfaces import BatchBuilder as BatchBuilderBase
from lmms_engine.rl.core.interfaces import DataBuffer, RolloutManager
from lmms_engine.rl.core.interfaces import TrainBatchAdapter as TrainBatchAdapterBase
from lmms_engine.rl.core.interfaces import (
    TrainerBridge,
    TrajectoryAdapter,
    WeightSyncClient,
)
from lmms_engine.rl.core.orchestrator import RLOrchestrator
from lmms_engine.rl.data_buffer import InMemoryDataBuffer
from lmms_engine.rl.lmms_eval import (
    LMMSEvalRolloutTaskConfig,
    LMMSEvalTrajectoryAdapter,
    build_rollout_episode_specs,
    clone_rollout_spec,
    trajectory_from_lmms_eval_episode,
)
from lmms_engine.rl.model_server import (
    RayModelServerPool,
    VLLMChatModelServer,
    start_ray_model_server_pool,
)
from lmms_engine.rl.protocol import (
    BufferStats,
    ModelVersion,
    RewardedTrajectory,
    RolloutTask,
    TrainBatch,
    TrajectoryStep,
)
from lmms_engine.rl.rollout_manager import (
    RayRolloutActor,
    RayRolloutManager,
    make_ray_rollout_actor,
)
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
    "LMMSEvalRolloutTaskConfig",
    "ModelVersion",
    "PassthroughTrainBatchAdapter",
    "RLRunConfig",
    "RayRolloutActor",
    "RayRolloutManager",
    "RayModelServerPool",
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
    "VLLMChatModelServer",
    "VLLMWeightSyncClient",
    "WeightSyncClient",
    "build_rollout_episode_specs",
    "clone_rollout_spec",
    "make_ray_rollout_actor",
    "start_ray_model_server_pool",
    "trajectory_from_lmms_eval_episode",
    "resolve_train_batch_size_per_gpu",
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
