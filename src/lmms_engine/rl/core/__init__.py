"""Core RL interfaces, registries, factories, and orchestration.

The package exports are lazy on purpose: low-level modules such as buffers
import `core.interfaces`, and eager factory imports would create cycles.
"""

__all__ = [
    "BATCH_BUILDER_REGISTRY",
    "BatchBuilder",
    "DATA_BUFFER_REGISTRY",
    "DEFAULT_RL_COMPONENT_FACTORY",
    "DataBuffer",
    "RLComponentFactory",
    "RLOrchestrator",
    "ROLLOUT_MANAGER_REGISTRY",
    "Registry",
    "RolloutManager",
    "TrajectoryAdapter",
    "TrainBatchAdapter",
    "TrainerBridge",
    "WEIGHT_SYNC_REGISTRY",
    "WeightSyncClient",
]


def __getattr__(name):
    if name in {
        "BatchBuilder",
        "DataBuffer",
        "RolloutManager",
        "TrajectoryAdapter",
        "TrainBatchAdapter",
        "TrainerBridge",
        "WeightSyncClient",
    }:
        from lmms_engine.rl.core import interfaces

        return getattr(interfaces, name)
    if name == "Registry":
        from lmms_engine.rl.core.registry import Registry

        return Registry
    if name in {
        "BATCH_BUILDER_REGISTRY",
        "DATA_BUFFER_REGISTRY",
        "DEFAULT_RL_COMPONENT_FACTORY",
        "RLComponentFactory",
        "ROLLOUT_MANAGER_REGISTRY",
        "WEIGHT_SYNC_REGISTRY",
    }:
        from lmms_engine.rl.core import factory

        return getattr(factory, name)
    if name == "RLOrchestrator":
        from lmms_engine.rl.core.orchestrator import RLOrchestrator

        return RLOrchestrator
    raise AttributeError(name)
