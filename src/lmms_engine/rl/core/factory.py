from __future__ import annotations

from lmms_engine.rl.config import DataBufferConfig, RolloutManagerConfig, TrainingEngineConfig, VLLMServerConfig
from lmms_engine.rl.core.interfaces import BatchBuilder, DataBuffer, RolloutManager, WeightSyncClient
from lmms_engine.rl.core.registry import Registry
from lmms_engine.rl.data_buffer import InMemoryDataBuffer
from lmms_engine.rl.rollout_manager import RayRolloutManager
from lmms_engine.rl.training_engine import VLLMWeightSyncClient

DATA_BUFFER_REGISTRY: Registry[DataBuffer] = Registry("data_buffer")
ROLLOUT_MANAGER_REGISTRY: Registry[RolloutManager] = Registry("rollout_manager")
BATCH_BUILDER_REGISTRY: Registry[BatchBuilder] = Registry("batch_builder")
WEIGHT_SYNC_REGISTRY: Registry[WeightSyncClient] = Registry("weight_sync")

DATA_BUFFER_REGISTRY.register("in_memory", InMemoryDataBuffer)
ROLLOUT_MANAGER_REGISTRY.register("ray", RayRolloutManager)
BATCH_BUILDER_REGISTRY.register("fixed_global", lambda config: _build_fixed_global_batch_builder(config))
WEIGHT_SYNC_REGISTRY.register("vllm_http", VLLMWeightSyncClient)


def _build_fixed_global_batch_builder(config: TrainingEngineConfig) -> BatchBuilder:
    from lmms_engine.train.rl.batch import FixedGlobalBatchBuilder

    return FixedGlobalBatchBuilder(config)


class RLComponentFactory:
    """Builds RL components from config names.

    New backends can register themselves without changing the orchestration
    loop, for example a durable queue can register as another data_buffer.
    """

    def build_data_buffer(self, config: DataBufferConfig | None = None) -> DataBuffer:
        config = config or DataBufferConfig()
        return DATA_BUFFER_REGISTRY.build(config.backend, config)

    def build_rollout_manager(self, config: RolloutManagerConfig | None = None) -> RolloutManager:
        config = config or RolloutManagerConfig()
        return ROLLOUT_MANAGER_REGISTRY.build(config.backend, config)

    def build_batch_builder(self, config: TrainingEngineConfig | None = None) -> BatchBuilder:
        config = config or TrainingEngineConfig()
        return BATCH_BUILDER_REGISTRY.build(config.batch_builder, config)

    def build_weight_sync(self, config: VLLMServerConfig | None = None) -> WeightSyncClient:
        config = config or VLLMServerConfig()
        return WEIGHT_SYNC_REGISTRY.build(config.backend, config)


DEFAULT_RL_COMPONENT_FACTORY = RLComponentFactory()
