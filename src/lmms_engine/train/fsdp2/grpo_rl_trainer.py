from __future__ import annotations

import copy
import os
import shutil
import time
import uuid
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from accelerate.utils import send_to_device
from loguru import logger
from tqdm import tqdm

import lmms_engine.models.utils as model_utils
import lmms_engine.parallel.process_group_manager as pgm
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
from lmms_engine.train.config import TrainingArguments
from lmms_engine.train.fsdp2.fsdp2_trainer import FSDP2SFTTrainer
from lmms_engine.train.fsdp2.rl_policy_step import FSDP2RLPolicyStepMixin, RLPolicyLoss
from lmms_engine.train.registry import TRAINER_REGISTER
from lmms_engine.train.rl import GRPOBatchAdapter, GRPOConfig, GRPOPayload
from lmms_engine.utils import ComputeTracker, TrainUtilities
from lmms_engine.utils.tracking import Tracking


@TRAINER_REGISTER.register("fsdp2_grpo_rl_trainer")
class FSDP2GRPORLTrainer(FSDP2RLPolicyStepMixin, FSDP2SFTTrainer):
    """FSDP2-only RL trainer for the LMMs-Engine + lmms-eval MVP.

    Rollout is deliberately coordinated on rank 0 and broadcast to all FSDP
    ranks. That keeps the first runnable path simple while preserving the
    diagram's independent RolloutManager, DataBuffer, and TrainingEngine
    boundaries.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        data_collator=None,
    ) -> None:
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            data_collator=data_collator,
        )
        self.rl_config = dict(getattr(args, "rl_config", None) or {})
        self.grpo_config = _build_dataclass(GRPOConfig, self.rl_config.get("algorithm", {}))
        self.batch_adapter = GRPOBatchAdapter(config=self.grpo_config, processor=processing_class)
        self.rollout_task_config = _build_dataclass(
            LMMSEvalRolloutTaskConfig,
            {
                **self.rl_config.get("task", {}),
                "model_server": self.rl_config.get("model_server", self.rl_config.get("task", {}).get("model_server")),
            },
        )
        self.rl_run_config = _build_rl_run_config(self.rl_config)
        self.sync_policy_weights = bool(self.rl_config.get("sync_policy_weights", False))
        self.rollout_poll_s = float(self.rl_config.get("rollout_poll_s", 0.1))
        self.rollout_sleep_s = float(self.rl_config.get("rollout_sleep_s", 0.05))
        self.rollout_seed = int(self.rl_config.get("rollout_seed", self.args.seed or 42))
        self.save_final_checkpoint = bool(self.rl_config.get("save_final_checkpoint", True))
        self._submitted_rollouts = 0
        self._last_policy_checkpoint: str | None = None
        self._ray_model_server_pool = None

    def train(self, resume_from_checkpoint: bool = False):
        self.prepare_model()
        self.prepare_optimizer()
        self.prepare_and_validate_rl_config()
        warmup_steps = (
            int(self.total_steps * self.args.warmup_ratio) if self.args.warmup_ratio > 0 else self.args.warmup_steps
        )
        self.prepare_scheduler(warmup_steps, self.total_steps)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            self.tracking = Tracking(
                project_name=os.environ.get("WANDB_PROJECT", self.args.project),
                experiment_name=os.environ.get("WANDB_NAME", self.args.run_name),
                default_backend=self.default_backend,
                config=self.args,
            )

        self.total_tokens = 0
        self.compute_tracker = ComputeTracker(
            num_gpus=world_size,
            carbon_intensity=getattr(self.args, "carbon_intensity", 0.475) or 0.475,
            gpu_tdp_watts=TrainUtilities.get_device_tdp(),
            gpu_name=torch.cuda.get_device_name(),
        )
        self.compute_tracker.start()

        loaded_checkpoint_dir = None
        if resume_from_checkpoint:
            loaded_checkpoint_dir = self._load_latest_checkpoint()
        else:
            self.global_step = 0

        self.ema.maybe_init(model=self.fsdp2_model, checkpoint_dir=loaded_checkpoint_dir)
        self.step_profiler.start()
        self.memory_snapshot_profiler.start()

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

        pbar = tqdm(total=self.total_steps, desc="RL Training", disable=rank != 0)
        if self.global_step:
            pbar.update(self.global_step)

        current_model_version = ModelVersion(version_id=self.global_step, checkpoint_path=self._last_policy_checkpoint)
        while not self.should_stop():
            train_batch = None
            if rank == 0:
                train_batch = self._next_rollout_train_batch(orchestrator, rollout_specs, current_model_version)

            object_list = [train_batch]
            dist.broadcast_object_list(object_list, src=0)
            train_batch = object_list[0]
            if train_batch is None:
                continue

            payload = self.batch_adapter.to_trainer_batch(train_batch)
            tensors = send_to_device(payload.tensors, self.fsdp2_model.device, non_blocking=True)

            self.memory_snapshot_profiler.step(self.global_step)
            start_time = time.perf_counter()
            try:
                train_metrics = self.training_step(tensors)
            except torch.OutOfMemoryError:
                self.memory_snapshot_profiler.dump_on_exception(f"oom_step{self.global_step}")
                raise
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    self.memory_snapshot_profiler.dump_on_exception(f"oom_step{self.global_step}")
                raise
            self.step_profiler.step()
            if self.step_profiler.should_save(self.global_step + 1):
                self.step_profiler.stop_and_save()
                self.step_profiler.stop_trace()

            delta_time = time.perf_counter() - start_time
            perf_metrics, self.total_tokens = self._rl_perf_metrics(tensors, delta_time, world_size)
            train_metrics.update(perf_metrics)
            train_metrics.update(_payload_metrics(payload))
            self.print_batch_input(tensors)

            is_accumulation_complete = self.accumulated_grad_steps == 0
            if rank == 0 and is_accumulation_complete:
                self.tracking.log(train_metrics, step=self.global_step)
            if not is_accumulation_complete:
                self._check_eval_results(rank)
                continue

            self.global_step += 1
            if self.should_save:
                output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                self.save_checkpoints(output_dir, self.global_step, total_limit=self.args.save_total_limit)
                self.validation_step(output_dir, self.global_step)

            if self.sync_policy_weights and self._should_sync_policy_weights():
                current_model_version = self._sync_policy_weights(orchestrator, rank)
            else:
                current_model_version = ModelVersion(
                    version_id=self.global_step,
                    checkpoint_path=self._last_policy_checkpoint,
                )
            pbar.update(1)
            self._check_eval_results(rank)

        pbar.close()
        self.memory_snapshot_profiler.stop_and_save(reason="rl_train_end")
        if self.save_final_checkpoint:
            output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
            self.save_checkpoints(output_dir, self.global_step, total_limit=self.args.save_total_limit)
            self.validation_step(output_dir, self.global_step)
        else:
            logger.info("Skipping final RL checkpoint because rl_config.save_final_checkpoint=false.")
        if self.eval_backend is not None:
            self._check_eval_results(rank, wait_until_complete=True)
        if rank == 0:
            summary = self.compute_tracker.finish()
            self.compute_tracker.save_summary(self.args.output_dir, summary)
            logger.info(
                f"Compute Summary: Total FLOPS={summary.total_flops_formatted}, "
                f"Duration={summary.training_duration_formatted}, "
                f"Energy={summary.energy_kwh} kWh, CO2={summary.co2_formatted}"
            )
        self.cuda_event_profiler.close()

    def compute_policy_loss(self, batch: dict[str, Any]) -> RLPolicyLoss:
        loss, metrics = self.compute_grpo_loss(batch)
        return RLPolicyLoss(loss=loss, metrics=metrics)

    def compute_grpo_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        labels = batch["labels"]
        advantages = batch["sample_advantages"].to(dtype=torch.float32, device=labels.device).view(-1)
        model_inputs = {
            key: value
            for key, value in batch.items()
            if key not in {"labels", "sample_advantages", "sample_rewards", "sample_metadata"}
        }

        cast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16
        with torch.autocast(device_type="cuda", dtype=cast_dtype):
            outputs = self.model(**model_inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        response_mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~response_mask, 0)
        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * response_mask

        token_counts = response_mask.sum(dim=-1).clamp_min(1)
        sample_log_probs = token_log_probs.sum(dim=-1) / token_counts
        loss = -(advantages * sample_log_probs).mean()

        reward = batch["sample_rewards"].to(dtype=torch.float32, device=labels.device).view(-1)
        metrics = {
            "rl/reward_mean": _distributed_mean(reward.mean()),
            "rl/advantage_mean": _distributed_mean(advantages.mean()),
            "rl/response_tokens_mean": _distributed_mean(token_counts.float().mean()),
            "rl/policy_logprob_mean": _distributed_mean(sample_log_probs.detach().mean()),
        }
        return loss, metrics

    def prepare_and_validate_rl_config(self) -> None:
        if self.processing_class is None:
            raise ValueError("fsdp2_grpo_rl_trainer requires a built processor via dataset_config.processor_config.")
        if self.args.max_steps <= 0:
            raise ValueError("trainer_args.max_steps must be positive for RL training.")
        self.steps_per_epoch = self.args.max_steps
        self.total_steps = self.args.max_steps

    def save_checkpoints(self, output_path: str, step: int, total_limit: int = None):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        os.makedirs(output_path, exist_ok=True)
        dist.barrier()
        model_dir = os.path.join(output_path, "pytorch_model_fsdp_0")
        optim_dir = os.path.join(output_path, "optimizer")
        extra_dir = os.path.join(output_path, "extra_state")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(optim_dir, exist_ok=True)
        os.makedirs(extra_dir, exist_ok=True)
        if self.ema.is_enabled():
            os.makedirs(os.path.join(output_path, "pytorch_ema_model_fsdp_0"), exist_ok=True)
        dist.barrier()

        torch.save(
            self.fsdp2_model.state_dict(),
            os.path.join(model_dir, f"model_world_size_{world_size}_rank_{rank}.pt"),
        )
        if self.ema.is_enabled() and self.ema.initialized:
            torch.save(
                self.ema.state_dict_for_save(self.fsdp2_model),
                os.path.join(
                    output_path,
                    "pytorch_ema_model_fsdp_0",
                    f"model_world_size_{world_size}_rank_{rank}.pt",
                ),
            )
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(optim_dir, f"optimizer_world_size_{world_size}_rank_{rank}.pt"),
        )
        extra_state = {
            "lr_scheduler_state": self.scheduler.state_dict(),
            "rng": self.get_rng_state(),
            "total_tokens": self.total_tokens,
            "accumulated_grad_steps": self.accumulated_grad_steps,
            "compute_tracker": self.compute_tracker.state_dict(),
            "submitted_rollouts": self._submitted_rollouts,
        }
        torch.save(
            extra_state,
            os.path.join(extra_dir, f"extra_state_world_size_{world_size}_rank_{rank}.pt"),
        )
        logger.info(f"Saved RL checkpoint to {output_path} at step {step}")
        if rank == 0:
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_path)
            self.model.config.save_pretrained(output_path)
            self.remove_old_checkpoints(self.args.output_dir, total_limit=total_limit)
            self._last_policy_checkpoint = os.path.abspath(output_path)
        dist.barrier()

    def load_checkpoints(self, output_path: str, step: int):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self.fsdp2_model.load_state_dict(
            torch.load(
                os.path.join(
                    output_path,
                    "pytorch_model_fsdp_0",
                    f"model_world_size_{world_size}_rank_{rank}.pt",
                ),
                weights_only=False,
            )
        )
        self.optimizer.load_state_dict(
            torch.load(
                os.path.join(output_path, "optimizer", f"optimizer_world_size_{world_size}_rank_{rank}.pt"),
                weights_only=False,
            )
        )
        extra_state = torch.load(
            os.path.join(output_path, "extra_state", f"extra_state_world_size_{world_size}_rank_{rank}.pt"),
            weights_only=False,
        )
        self.total_tokens = extra_state["total_tokens"]
        self.accumulated_grad_steps = extra_state.get("accumulated_grad_steps", 0)
        self._submitted_rollouts = extra_state.get("submitted_rollouts", 0)
        if "compute_tracker" in extra_state and hasattr(self, "compute_tracker"):
            self.compute_tracker.load_state_dict(extra_state["compute_tracker"])
        self.load_rng_state(extra_state["rng"])
        self.scheduler.load_state_dict(extra_state["lr_scheduler_state"])
        self.global_step = step
        self._last_policy_checkpoint = os.path.abspath(output_path)
        logger.info(f"Loaded RL checkpoint from {output_path} at step {step}")

    def _load_latest_checkpoint(self) -> str | None:
        checkpoints = [item for item in os.listdir(self.args.output_dir) if item.startswith("checkpoint")]
        if not checkpoints:
            self.global_step = 0
            return None
        checkpoints.sort(key=lambda item: int(item.split("-")[1]))
        latest = checkpoints[-1]
        checkpoint_dir = os.path.join(self.args.output_dir, latest)
        self.load_checkpoints(checkpoint_dir, int(latest.split("-")[1]))
        return checkpoint_dir

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
            self.rl_run_config.rollout.num_workers * self.rl_run_config.rollout.max_inflight_per_worker,
        )
        while orchestrator.rollout_manager.inflight < capacity and not orchestrator.data_buffer.should_pause_rollout():
            base_spec = rollout_specs[self._submitted_rollouts % len(rollout_specs)]
            spec = copy.copy(base_spec)
            spec.seed = self.rollout_seed + self._submitted_rollouts
            task = RolloutTask(
                task_id=f"rollout-{self._submitted_rollouts}-{uuid.uuid4().hex[:8]}",
                payload=spec,
                model_version=model_version,
                seed=spec.seed,
                metadata={"rollout_index": self._submitted_rollouts},
            )
            if not orchestrator.submit_rollout(task):
                break
            self._submitted_rollouts += 1

    def _should_sync_policy_weights(self) -> bool:
        return self.global_step % max(1, self.rl_run_config.training.update_weights_every_steps) == 0

    def _sync_policy_weights(self, orchestrator, rank: int) -> ModelVersion:
        output_dir = os.path.join(self.args.output_dir, f"policy-sync-{self.global_step}")
        self.save_checkpoints(output_dir, self.global_step, total_limit=None)
        model_version = ModelVersion(version_id=self.global_step, checkpoint_path=os.path.abspath(output_dir))
        if rank == 0:
            result = orchestrator.reload_policy_weights(model_version)
            logger.info(f"Reloaded policy weights for version {self.global_step}: {result}")
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

        self._ray_model_server_pool = start_ray_model_server_pool(model_server)
        self.rollout_task_config.model_server = self._ray_model_server_pool.client_spec(
            **dict(model_server.get("client", {}) or {})
        )
        logger.info(
            "Started Ray model server pool with "
            f"{len(self._ray_model_server_pool.actor_names)} replica(s); "
            f"load_balancer={self._ray_model_server_pool.load_balancer_name}"
        )

    def _rl_perf_metrics(self, batch: dict[str, Any], delta_time: float, world_size: int) -> tuple[dict, int]:
        seq_len = (
            batch.get("attention_mask", torch.zeros((1, 1), device=self.fsdp2_model.device))
            .sum(dim=1)
            .detach()
            .cpu()
            .tolist()
        )
        flops, promised_flops, raw_flops = model_utils.flops_counter.estimate_flops(seq_len, delta_time=delta_time)
        self.compute_tracker.accumulate_flops(raw_flops)
        parallel_size = pgm.process_group_manager.cp_world_size * pgm.process_group_manager.tp_world_size
        return self.calculate_training_metrics(
            flops=flops,
            parallel_size=parallel_size,
            promised_flops=promised_flops,
            device=self.fsdp2_model.device,
            seq_len=seq_len,
            total_tokens=self.total_tokens,
            delta_time=delta_time,
            world_size=world_size,
        )

    def remove_old_checkpoints(self, output_path: str, total_limit: int = None):
        if total_limit is None:
            return
        checkpoints = [item for item in os.listdir(output_path) if item.startswith("checkpoint")]
        checkpoints.sort(key=lambda item: int(item.split("-")[1]))
        if len(checkpoints) > total_limit:
            for checkpoint in checkpoints[:-total_limit]:
                shutil.rmtree(os.path.join(output_path, checkpoint))


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


def _distributed_mean(value: torch.Tensor) -> float:
    reduced = value.detach().float()
    dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
    return float(reduced.item())


def _payload_metrics(payload: GRPOPayload) -> dict[str, float]:
    rewards = payload.tensors.get("sample_rewards") if isinstance(payload.tensors, dict) else None
    if rewards is None:
        return {}
    return {
        "rl/batch_reward_mean_host": float(rewards.float().mean().item()),
        "rl/batch_size": float(rewards.numel()),
    }
