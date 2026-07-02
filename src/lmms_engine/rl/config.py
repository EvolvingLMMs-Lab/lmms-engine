from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VLLMServerConfig:
    """OpenAI-compatible policy server used by lmms-eval rollout workers."""

    backend: str = "vllm_http"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "policy"
    api_key: str = "EMPTY"
    reload_endpoint: str = "/reload_weights"
    logprobs_endpoint: str = "/v1/completions"
    timeout_s: float = 600.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutManagerConfig:
    """Ray-side rollout scheduling knobs.

    Each Ray actor owns a synchronous lmms-eval loop worker. Concurrency comes
    from Ray actors plus optional thread-level batching inside lmms-eval.
    """

    backend: str = "ray"
    num_workers: int = 1
    max_inflight_per_worker: int = 1
    actor_options: dict[str, Any] = field(default_factory=dict)
    worker_config: dict[str, Any] = field(default_factory=dict)
    trajectory_adapter: Any = "lmms_engine.rl.lmms_eval.trajectory_adapter:LMMSEvalTrajectoryAdapter"
    task_queue_size: int = 1024
    pause_on_buffer_high_watermark: bool = True
    poll_timeout_s: float = 0.0


@dataclass(slots=True)
class DataBufferConfig:
    """Independent producer-consumer buffer between rollout and training."""

    backend: str = "in_memory"
    max_trajectories: int = 4096
    high_watermark: int = 3072
    low_watermark: int = 2048
    train_batch_size: int = 64
    min_trajectories_per_batch: int = 1
    preserve_model_version: bool = True


@dataclass(slots=True)
class TrainingEngineConfig:
    """Training-side MVP boundary.

    The first implementation is intentionally FSDP2-only. Algorithm-specific
    PPO/GRPO loss construction belongs behind TrainBatchAdapter.
    """

    trainer_type: str = "fsdp2_rl_trainer"
    global_batch_size: int = 64
    update_weights_every_steps: int = 1
    batch_builder: str = "fixed_global"
    batch_adapter: dict[str, Any] = field(default_factory=dict)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RLRunConfig:
    """Top-level RL MVP config object."""

    rollout: RolloutManagerConfig = field(default_factory=RolloutManagerConfig)
    data_buffer: DataBufferConfig = field(default_factory=DataBufferConfig)
    training: TrainingEngineConfig = field(default_factory=TrainingEngineConfig)
    vllm: VLLMServerConfig = field(default_factory=VLLMServerConfig)
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
