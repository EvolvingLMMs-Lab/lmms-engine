from __future__ import annotations

import copy
import os
import pathlib
import time
import uuid
from dataclasses import fields, is_dataclass
from functools import reduce
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger
from tqdm import tqdm

import lmms_engine.models.utils as model_utils
import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.datasets.processor import ProcessorConfig
from lmms_engine.mapping_func import DATAPROCESSOR_MAPPING
from lmms_engine.rl import (
    DataBufferConfig,
    ModelVersion,
    RLOrchestrator,
    RLRunConfig,
    RolloutManagerConfig,
    RolloutTask,
    TrainingEngineConfig,
    VLLMServerConfig,
)
from lmms_engine.rl.lmms_eval import (
    LMMSEvalRolloutTaskConfig,
    build_rollout_episode_specs,
)
from lmms_engine.rl.lmms_eval.paths import ensure_lmms_eval_importable
from lmms_engine.train.registry import TRAINER_REGISTER
from lmms_engine.train.rl.grpo import GRPOBatchAdapter, GRPOConfig, GRPOPayload
from lmms_engine.train.runner import TrainRunner
from lmms_engine.utils import ComputeTracker, TrainUtilities
from lmms_engine.utils.tracking import Tracking


class RLTrainRunner(TrainRunner):
    """Online RL runner that owns rollout scheduling and drives policy updates."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.rl_config = dict(getattr(config.trainer_args, "rl_config", None) or {})
        self.rl_run_config = _build_rl_run_config(self.rl_config)
        self.rollout_task_config = _build_dataclass(
            LMMSEvalRolloutTaskConfig,
            {
                **self.rl_config.get("task", {}),
                "model_server": self.rl_config.get("model_server", self.rl_config.get("task", {}).get("model_server")),
            },
        )
        self.sync_policy_weights = bool(self.rl_config.get("sync_policy_weights", False))
        self.rollout_poll_s = float(self.rl_config.get("rollout_poll_s", 0.1))
        self.rollout_sleep_s = float(self.rl_config.get("rollout_sleep_s", 0.05))
        self.rollout_seed = int(self.rl_config.get("rollout_seed", self.config.trainer_args.seed or 42))
        self.save_final_checkpoint = bool(self.rl_config.get("save_final_checkpoint", True))
        self.submitted_rollouts = 0
        self.ray_model_server_pool = None
        self.batch_adapter = None

    def build(self):
        if dist.is_initialized():
            self.create_sp_dis_group()
        self.model = self._build_model()
        self.eval_dataset = None
        self.train_dataset = None
        self.processing_class = self._build_processor()
        self._apply_monkey_patch()
        self.trainer = self._build_trainer()
        self.batch_adapter = self._build_batch_adapter()

    def _build_processor(self):
        processor_config = self.train_dataset_config.processor_config
        if isinstance(processor_config, dict):
            processor_config = ProcessorConfig(**processor_config)
        processor_cls = DATAPROCESSOR_MAPPING[processor_config.processor_type]
        processor = processor_cls(processor_config)
        processor.build()
        return processor

    def _build_trainer(self):
        trainer_cls = TRAINER_REGISTER[self.config.trainer_type]
        return trainer_cls(
            model=self.model,
            args=self.config.trainer_args,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            processing_class=self.processing_class,
        )

    def _build_batch_adapter(self):
        algorithm_config = dict(self.rl_config.get("algorithm", {}) or {})
        algorithm = str(algorithm_config.get("name", self.rl_config.get("algorithm_name", "grpo"))).lower()
        if algorithm != "grpo":
            raise ValueError(f"Unsupported RL algorithm {algorithm!r}; only 'grpo' is registered.")
        return GRPOBatchAdapter(
            config=_build_dataclass(GRPOConfig, algorithm_config),
            processor=self.processing_class,
        )

    def run(self, **kwargs):
        self._freeze_modules()
        resume_from_checkpoint = bool(list(pathlib.Path(self.config.trainer_args.output_dir).glob("checkpoint-*")))
        self._prepare_trainer(resume_from_checkpoint=resume_from_checkpoint)

        trainer = self.trainer
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        orchestrator = None
        rollout_specs = None
        if rank == 0:
            self._maybe_init_ray()
            self._maybe_start_ray_model_server_pool()
            rollout_specs = build_rollout_episode_specs(self.rollout_task_config)
            if not rollout_specs:
                raise ValueError("No rollout specs were built from the lmms-eval task config.")
            orchestrator = RLOrchestrator(config=self.rl_run_config)
            orchestrator.start()
            logger.info(f"Built {len(rollout_specs)} rollout episode spec(s) for RL training.")

        pbar = tqdm(total=trainer.total_steps, desc="RL Training", disable=rank != 0)
        if trainer.global_step:
            pbar.update(trainer.global_step)

        current_model_version = ModelVersion(
            version_id=trainer.global_step,
            checkpoint_path=getattr(trainer, "_last_policy_checkpoint", None),
        )
        while not trainer.should_stop():
            train_batch = None
            if rank == 0:
                train_batch = self._next_rollout_train_batch(orchestrator, rollout_specs, current_model_version)

            object_list = [train_batch]
            dist.broadcast_object_list(object_list, src=0)
            train_batch = object_list[0]
            if train_batch is None:
                continue

            payload = self.batch_adapter.to_trainer_batch(train_batch)
            tensors = trainer.prepare_rl_batch(payload)

            trainer.memory_snapshot_profiler.step(trainer.global_step)
            start_time = time.perf_counter()
            try:
                train_metrics = trainer.train_batch(tensors)
            except torch.OutOfMemoryError:
                trainer.memory_snapshot_profiler.dump_on_exception(f"oom_step{trainer.global_step}")
                raise
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    trainer.memory_snapshot_profiler.dump_on_exception(f"oom_step{trainer.global_step}")
                raise
            trainer.step_profiler.step()
            if trainer.step_profiler.should_save(trainer.global_step + 1):
                trainer.step_profiler.stop_and_save()
                trainer.step_profiler.stop_trace()

            delta_time = time.perf_counter() - start_time
            perf_metrics, trainer.total_tokens = self._rl_perf_metrics(tensors, delta_time, world_size)
            train_metrics.update(perf_metrics)
            train_metrics.update(_payload_metrics(payload))
            trainer.print_batch_input(tensors)

            is_accumulation_complete = trainer.accumulated_grad_steps == 0
            if rank == 0 and is_accumulation_complete:
                trainer.tracking.log(train_metrics, step=trainer.global_step)
            if not is_accumulation_complete:
                trainer._check_eval_results(rank)
                continue

            trainer.global_step += 1
            if trainer.should_save:
                output_dir = os.path.join(trainer.args.output_dir, f"checkpoint-{trainer.global_step}")
                self._save_policy_checkpoint(output_dir, trainer.global_step, total_limit=trainer.args.save_total_limit)
                trainer.validation_step(output_dir, trainer.global_step)

            if self.sync_policy_weights and self._should_sync_policy_weights():
                current_model_version = self._sync_policy_weights(orchestrator, rank)
            else:
                current_model_version = ModelVersion(
                    version_id=trainer.global_step,
                    checkpoint_path=getattr(trainer, "_last_policy_checkpoint", None),
                )
            pbar.update(1)
            trainer._check_eval_results(rank)

        pbar.close()
        trainer.memory_snapshot_profiler.stop_and_save(reason="rl_train_end")
        if self.save_final_checkpoint:
            output_dir = os.path.join(trainer.args.output_dir, f"checkpoint-{trainer.global_step}")
            self._save_policy_checkpoint(output_dir, trainer.global_step, total_limit=trainer.args.save_total_limit)
            trainer.validation_step(output_dir, trainer.global_step)
        else:
            logger.info("Skipping final RL checkpoint because rl_config.save_final_checkpoint=false.")
        if trainer.eval_backend is not None:
            trainer._check_eval_results(rank, wait_until_complete=True)
        if rank == 0:
            summary = trainer.compute_tracker.finish()
            trainer.compute_tracker.save_summary(trainer.args.output_dir, summary)
            logger.info(
                f"Compute Summary: Total FLOPS={summary.total_flops_formatted}, "
                f"Duration={summary.training_duration_formatted}, "
                f"Energy={summary.energy_kwh} kWh, CO2={summary.co2_formatted}"
            )
        trainer.cuda_event_profiler.close()

    def _prepare_trainer(self, *, resume_from_checkpoint: bool) -> None:
        trainer = self.trainer
        trainer.prepare_model()
        trainer.prepare_optimizer()
        trainer.prepare_and_validate_rl_config()
        warmup_steps = (
            int(trainer.total_steps * trainer.args.warmup_ratio)
            if trainer.args.warmup_ratio > 0
            else trainer.args.warmup_steps
        )
        trainer.prepare_scheduler(warmup_steps, trainer.total_steps)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            trainer.tracking = Tracking(
                project_name=os.environ.get("WANDB_PROJECT", trainer.args.project),
                experiment_name=os.environ.get("WANDB_NAME", trainer.args.run_name),
                default_backend=trainer.default_backend,
                config=trainer.args,
            )

        trainer.total_tokens = 0
        trainer.compute_tracker = ComputeTracker(
            num_gpus=world_size,
            carbon_intensity=getattr(trainer.args, "carbon_intensity", 0.475) or 0.475,
            gpu_tdp_watts=TrainUtilities.get_device_tdp(),
            gpu_name=torch.cuda.get_device_name(),
        )
        trainer.compute_tracker.start()

        loaded_checkpoint_dir = None
        if resume_from_checkpoint:
            loaded_checkpoint_dir = trainer._load_latest_checkpoint()
            self.submitted_rollouts = getattr(trainer, "_submitted_rollouts", 0)
        else:
            trainer.global_step = 0
            trainer._submitted_rollouts = 0

        trainer.ema.maybe_init(model=trainer.fsdp2_model, checkpoint_dir=loaded_checkpoint_dir)
        trainer.step_profiler.start()
        trainer.memory_snapshot_profiler.start()

    def _freeze_modules(self) -> None:
        if not self.config.trainer_args.freeze_modules:
            return
        for modules in self.config.trainer_args.freeze_modules:
            cls = reduce(lambda obj, key: getattr(obj, key, None), modules.split("."), self.model)
            if cls is not None:
                for param in cls.parameters():
                    param.requires_grad = False

    def _next_rollout_train_batch(self, orchestrator, rollout_specs, model_version: ModelVersion):
        while True:
            self._submit_available_rollouts(orchestrator, rollout_specs, model_version)
            orchestrator.drain_rollouts(timeout_s=self.rollout_poll_s)
            train_batch = orchestrator.next_train_batch()
            if train_batch is not None:
                return train_batch
            time.sleep(self.rollout_sleep_s)

    def _submit_available_rollouts(self, orchestrator, rollout_specs, model_version: ModelVersion) -> None:
        capacity = max(
            1,
            self.rl_run_config.rollout.num_workers
            * self.rl_run_config.rollout.max_inflight_per_worker
            * max(1, int(self.rl_run_config.rollout.batch_size)),
        )
        while orchestrator.rollout_manager.inflight < capacity and not orchestrator.data_buffer.should_pause_rollout():
            base_spec = rollout_specs[self.submitted_rollouts % len(rollout_specs)]
            spec = copy.copy(base_spec)
            spec.seed = self.rollout_seed + self.submitted_rollouts
            task = RolloutTask(
                task_id=f"rollout-{self.submitted_rollouts}-{uuid.uuid4().hex[:8]}",
                payload=spec,
                model_version=model_version,
                seed=spec.seed,
                metadata={"rollout_index": self.submitted_rollouts},
            )
            if not orchestrator.submit_rollout(task):
                break
            self.submitted_rollouts += 1

    def _save_policy_checkpoint(self, output_dir: str, step: int, total_limit: int | None = None) -> None:
        self.trainer._submitted_rollouts = self.submitted_rollouts
        self.trainer.save_checkpoints(output_dir, step, total_limit=total_limit)

    def _should_sync_policy_weights(self) -> bool:
        return self.trainer.global_step % max(1, self.rl_run_config.training.update_weights_every_steps) == 0

    def _sync_policy_weights(self, orchestrator, rank: int) -> ModelVersion:
        output_dir = os.path.join(self.trainer.args.output_dir, f"policy-sync-{self.trainer.global_step}")
        self._save_policy_checkpoint(output_dir, self.trainer.global_step, total_limit=None)
        model_version = ModelVersion(version_id=self.trainer.global_step, checkpoint_path=os.path.abspath(output_dir))
        if rank == 0:
            result = orchestrator.reload_policy_weights(model_version)
            logger.info(f"Reloaded policy weights for version {self.trainer.global_step}: {result}")
        return model_version

    def _maybe_init_ray(self) -> None:
        if self.rl_run_config.rollout.backend != "ray":
            return
        import ray

        ray_init_kwargs = _ray_init_kwargs_with_lmms_eval_path(self.rl_run_config.ray_init_kwargs)
        if not ray.is_initialized():
            ray.init(**ray_init_kwargs)

    def _maybe_start_ray_model_server_pool(self) -> None:
        model_server = self.rollout_task_config.model_server
        if not isinstance(model_server, dict):
            return
        backend = model_server.get("name") or model_server.get("backend")
        if backend != "ray_actor_pool":
            return

        from lmms_engine.rl.model_server import start_ray_model_server_pool

        self.ray_model_server_pool = start_ray_model_server_pool(model_server)
        self.rollout_task_config.model_server = self.ray_model_server_pool.client_spec(
            **dict(model_server.get("client", {}) or {})
        )
        logger.info(
            "Started Ray model server pool with "
            f"{len(self.ray_model_server_pool.actor_names)} replica(s); "
            f"load_balancer={self.ray_model_server_pool.load_balancer_name}"
        )

    def _rl_perf_metrics(self, batch: dict[str, Any], delta_time: float, world_size: int) -> tuple[dict, int]:
        seq_len = (
            batch.get("attention_mask", torch.zeros((1, 1), device=self.trainer.fsdp2_model.device))
            .sum(dim=1)
            .detach()
            .cpu()
            .tolist()
        )
        flops, promised_flops, raw_flops = model_utils.flops_counter.estimate_flops(seq_len, delta_time=delta_time)
        self.trainer.compute_tracker.accumulate_flops(raw_flops)
        parallel_size = pgm.process_group_manager.cp_world_size * pgm.process_group_manager.tp_world_size
        return self.trainer.calculate_training_metrics(
            flops=flops,
            parallel_size=parallel_size,
            promised_flops=promised_flops,
            device=self.trainer.fsdp2_model.device,
            seq_len=seq_len,
            total_tokens=self.trainer.total_tokens,
            delta_time=delta_time,
            world_size=world_size,
        )


def _build_rl_run_config(config: dict[str, Any]) -> RLRunConfig:
    return RLRunConfig(
        rollout=_build_dataclass(RolloutManagerConfig, config.get("rollout", {})),
        data_buffer=_build_dataclass(DataBufferConfig, config.get("data_buffer", {})),
        training=_build_dataclass(TrainingEngineConfig, config.get("training", {})),
        vllm=_build_dataclass(VLLMServerConfig, config.get("vllm", {})),
        ray_init_kwargs=dict(config.get("ray_init_kwargs", {}) or {}),
        extra_kwargs=dict(config.get("extra_kwargs", {}) or {}),
    )


def _build_dataclass(cls, values: dict[str, Any] | None):
    if values is None:
        return cls()
    if isinstance(values, cls):
        return values
    if not is_dataclass(cls):
        return cls(**(values or {}))
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in dict(values).items() if key in allowed})


def _ray_init_kwargs_with_lmms_eval_path(ray_init_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = copy.deepcopy(ray_init_kwargs or {})
    lmms_eval_root = ensure_lmms_eval_importable()
    engine_src = Path(__file__).resolve().parents[3]
    pythonpath = _prepend_pythonpath([engine_src, lmms_eval_root])
    os.environ["PYTHONPATH"] = pythonpath

    runtime_env = dict(kwargs.get("runtime_env") or {})
    env_vars = dict(runtime_env.get("env_vars") or {})
    env_vars["PYTHONPATH"] = _prepend_pythonpath([engine_src, lmms_eval_root], env_vars.get("PYTHONPATH"))
    runtime_env["env_vars"] = env_vars
    kwargs["runtime_env"] = runtime_env
    return kwargs


def _prepend_pythonpath(paths: list[Path], existing: str | None = None) -> str:
    entries = [str(path) for path in paths]
    current = existing if existing is not None else os.environ.get("PYTHONPATH", "")
    entries.extend(item for item in current.split(os.pathsep) if item)
    deduped = list(dict.fromkeys(entries))
    return os.pathsep.join(deduped)


def _payload_metrics(payload: GRPOPayload) -> dict[str, float]:
    rewards = payload.tensors.get("sample_rewards") if isinstance(payload.tensors, dict) else None
    if rewards is None:
        return {}
    return {
        "rl/batch_reward_mean_host": float(rewards.float().mean().item()),
        "rl/batch_size": float(rewards.numel()),
    }
