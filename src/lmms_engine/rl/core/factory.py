from __future__ import annotations

from lmms_engine.rl.config import (
    DataBufferConfig,
    RolloutManagerConfig,
    TrainingEngineConfig,
    VLLMServerConfig,
)
from lmms_engine.rl.core.interfaces import (
    BatchBuilder,
    DataBuffer,
    RolloutManager,
    WeightSyncClient,
)
from lmms_engine.rl.core.registry import Registry
from lmms_engine.rl.data_buffer import InMemoryDataBuffer
from lmms_engine.rl.rollout_manager import RayRolloutManager
from lmms_engine.rl.training_engine import (
    RayActorWeightSyncClient,
    VLLMWeightSyncClient,
)

DATA_BUFFER_REGISTRY: Registry[DataBuffer] = Registry("data_buffer")
ROLLOUT_MANAGER_REGISTRY: Registry[RolloutManager] = Registry("rollout_manager")
BATCH_BUILDER_REGISTRY: Registry[BatchBuilder] = Registry("batch_builder")
WEIGHT_SYNC_REGISTRY: Registry[WeightSyncClient] = Registry("weight_sync")

DATA_BUFFER_REGISTRY.register("in_memory", InMemoryDataBuffer)
ROLLOUT_MANAGER_REGISTRY.register("ray", RayRolloutManager)
BATCH_BUILDER_REGISTRY.register("fixed_global", lambda config: _build_fixed_global_batch_builder(config))
WEIGHT_SYNC_REGISTRY.register("vllm_http", VLLMWeightSyncClient)
WEIGHT_SYNC_REGISTRY.register("ray_actor_pool", lambda config: _build_ray_actor_weight_sync(config))


def _build_fixed_global_batch_builder(config: TrainingEngineConfig) -> BatchBuilder:
    from lmms_engine.train.rl.batch import FixedGlobalBatchBuilder

    return FixedGlobalBatchBuilder(config)


def _build_ray_actor_weight_sync(config: VLLMServerConfig) -> WeightSyncClient:
    extra_kwargs = dict(config.extra_kwargs or {})
    actor_names = extra_kwargs.pop("actor_names", [])
    if isinstance(actor_names, str):
        actor_names = [item.strip() for item in actor_names.split(",") if item.strip()]
    namespace = extra_kwargs.pop("namespace", None)
    timeout_s_value = extra_kwargs.pop("timeout_s", config.timeout_s)
    timeout_s = None if timeout_s_value is None else float(timeout_s_value)
    preflight_weight_path_value = extra_kwargs.pop("preflight_weight_path", True)
    require_hf_checkpoint_value = extra_kwargs.pop("require_hf_checkpoint", True)
    preflight_weight_path = _as_bool(preflight_weight_path_value, default=True)
    require_hf_checkpoint = _as_bool(require_hf_checkpoint_value, default=True)
    extra_kwargs.pop("shared_checkpoint_dir", None)
    extra_kwargs.pop("policy_sync_dir", None)
    extra_kwargs.pop("update_weight_mode", None)
    extra_kwargs.pop("update_weight_transport", None)
    extra_kwargs.pop("update_weight_disk_dir", None)
    if timeout_s is not None:
        extra_kwargs.setdefault("timeout_s", timeout_s)
    return RayActorWeightSyncClient(
        actor_names=list(actor_names),
        namespace=namespace,
        timeout_s=timeout_s,
        preflight_weight_path=preflight_weight_path,
        require_hf_checkpoint=require_hf_checkpoint,
        extra_kwargs=extra_kwargs,
    )


def _as_bool(value, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


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
