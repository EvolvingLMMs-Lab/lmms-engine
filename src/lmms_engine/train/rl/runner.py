from __future__ import annotations

import copy
import os
import pathlib
import shutil
import sys
import time
import traceback
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
    RayActorWeightSyncClient,
    resolve_train_batch_size_per_gpu,
)
from lmms_engine.rl.lmms_eval import (
    LMMSEvalRolloutTaskConfig,
    build_rollout_episode_specs,
)
from lmms_engine.rl.lmms_eval.paths import ensure_lmms_eval_importable
from lmms_engine.rl.protocol import TrainBatch
from lmms_engine.train.registry import TRAINER_REGISTER
from lmms_engine.train.rl.grpo import GRPOBatchAdapter, GRPOConfig, GRPOPayload
from lmms_engine.train.runner import TrainRunner
from lmms_engine.utils import ComputeTracker, TrainUtilities
from lmms_engine.utils.tracking import Tracking

_VLLM_SERVER_FACTORY = "lmms_engine.rl.model_server.vllm:VLLMChatModelServer"


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
        _validate_ray_vllm_model_server(self.rollout_task_config.model_server)
        self.sync_policy_weights = bool(self.rl_config.get("sync_policy_weights", True))
        self.rollout_poll_s = float(self.rl_config.get("rollout_poll_s", 0.1))
        self.rollout_sleep_s = float(self.rl_config.get("rollout_sleep_s", 0.05))
        self.sync_initial_policy_weights = bool(self.rl_config.get("sync_initial_policy_weights", True))
        self.sync_drain_inflight_rollouts = bool(self.rl_config.get("sync_drain_inflight_rollouts", True))
        self.sync_drain_timeout_s = float(self.rl_config.get("sync_drain_timeout_s", 1800.0))
        self.sync_drain_poll_s = float(self.rl_config.get("sync_drain_poll_s", self.rollout_poll_s))
        self.policy_sync_merge_to_hf = bool(self.rl_config.get("policy_sync_merge_to_hf", True))
        self.policy_sync_keep_last = int(self.rl_config.get("policy_sync_keep_last", 2))
        weight_sync_config = dict(self.rl_config.get("weight_sync", {}) or {})
        self.update_weight_mode = str(
            self.rl_config.get("update_weight_mode", weight_sync_config.get("update_weight_mode", "full"))
        ).lower()
        self.update_weight_transport = str(
            self.rl_config.get(
                "update_weight_transport",
                weight_sync_config.get("update_weight_transport", "disk"),
            )
        ).lower()
        self.update_weight_disk_dir = _coalesce_update_weight_disk_dir(
            self.rl_config,
            weight_sync_config,
            env_value=os.environ.get("LMMS_ENGINE_POLICY_SYNC_DIR"),
        )
        self.policy_sync_dir = self.update_weight_disk_dir
        if self.sync_policy_weights:
            self._validate_update_weight_config()
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
        self._log_train_rank_placement(rank, world_size)
        global_train_batch_size = resolve_train_batch_size_per_gpu(
            self.rl_run_config,
            train_world_size=world_size,
        )
        if rank == 0:
            logger.info(
                "Resolved RL train batch: "
                f"train_batch_size_per_gpu={self.rl_run_config.data_buffer.train_batch_size_per_gpu}, "
                f"train_world_size={world_size}, global_train_batch_size={global_train_batch_size}"
            )

        orchestrator = None
        rollout_specs = None
        if rank == 0:
            self._maybe_init_ray()
            self._maybe_start_ray_model_server_pool()
            rollout_specs = build_rollout_episode_specs(self.rollout_task_config)
            if not rollout_specs:
                raise ValueError("No rollout specs were built from the lmms-eval task config.")
            orchestrator = RLOrchestrator(config=self.rl_run_config, weight_sync=self._build_weight_sync_client())
            orchestrator.start()
            logger.info(f"Built {len(rollout_specs)} rollout episode spec(s) for RL training.")

        pbar = tqdm(total=trainer.total_steps, desc="RL Training", disable=rank != 0)
        if trainer.global_step:
            pbar.update(trainer.global_step)

        current_model_version = ModelVersion(
            version_id=trainer.global_step,
            checkpoint_path=getattr(trainer, "_last_policy_checkpoint", None),
        )
        if self.sync_policy_weights and self.sync_initial_policy_weights:
            current_model_version = self._sync_policy_weights(orchestrator, rank)

        while not trainer.should_stop():
            train_batch = None
            if rank == 0:
                train_batch = self._next_rollout_train_batch(orchestrator, rollout_specs, current_model_version)

            object_list = [train_batch]
            dist.broadcast_object_list(object_list, src=0)
            train_batch = object_list[0]
            if train_batch is None:
                continue
            train_batch = _shard_train_batch(
                train_batch,
                rank=rank,
                world_size=world_size,
                max_steps_per_trajectory=getattr(self.batch_adapter.config, "max_steps_per_trajectory", None),
            )

            payload = self.batch_adapter.to_trainer_batch(train_batch)
            tensors = trainer.prepare_rl_batch(payload)
            self._maybe_log_train_shard_balance(train_batch, payload, tensors, rank, world_size)

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

    def _log_train_rank_placement(self, rank: int, world_size: int) -> None:
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
        else:
            current_device = None
            device_name = "cpu"
        logger.info(
            "RL train rank placement: "
            f"rank={rank}/{world_size}, "
            f"node={os.uname().nodename}, "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"current_device={current_device}, "
            f"device_name={device_name}"
        )

    def _maybe_log_train_shard_balance(
        self,
        train_batch: TrainBatch,
        payload: GRPOPayload,
        tensors: dict[str, Any],
        rank: int,
        world_size: int,
    ) -> None:
        log_steps = int(self.rl_config.get("train_shard_log_steps", 3))
        if log_steps <= 0 or self.trainer.global_step >= log_steps:
            return

        attention_mask = tensors.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.numel() > 0:
            seq_lens = attention_mask.sum(dim=1).detach().to(dtype=torch.float32)
            seq_min = float(seq_lens.min().item())
            seq_max = float(seq_lens.max().item())
            seq_mean = float(seq_lens.mean().item())
        else:
            seq_min = seq_max = seq_mean = 0.0

        sample_metadata = payload.tensors.get("sample_metadata", []) if isinstance(payload.tensors, dict) else []
        local_metrics = {
            "rank": rank,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "trajectories": len(train_batch.trajectories),
            "trajectory_steps": sum(len(trajectory.steps) for trajectory in train_batch.trajectories),
            "estimated_train_steps": train_batch.metadata.get("local_estimated_train_steps"),
            "samples": len(sample_metadata),
            "seq_min": seq_min,
            "seq_max": seq_max,
            "seq_mean": seq_mean,
        }
        gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_metrics)
        if rank != 0:
            return

        summaries = []
        for item in sorted((metric for metric in gathered if metric is not None), key=lambda metric: metric["rank"]):
            summaries.append(
                "rank={rank} cuda={cuda_visible_devices} dev={current_device} "
                "traj={trajectories} steps={trajectory_steps} est={estimated_train_steps} "
                "samples={samples} seq=({seq_min:.0f},{seq_mean:.0f},{seq_max:.0f})".format(**item)
            )
        logger.info(f"RL train shard balance step={self.trainer.global_step}: " + " | ".join(summaries))

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
            task_id = f"rollout-{self.submitted_rollouts}-{uuid.uuid4().hex[:8]}"
            spec.request_metadata = {
                **dict(getattr(spec, "request_metadata", {}) or {}),
                "request_id": task_id,
                "rollout_index": self.submitted_rollouts,
            }
            task = RolloutTask(
                task_id=task_id,
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

    def _build_weight_sync_client(self):
        if self.ray_model_server_pool is None:
            return None

        extra_kwargs = {
            **dict(self.rl_run_config.vllm.extra_kwargs or {}),
            **dict(self.rl_config.get("weight_sync", {}) or {}),
        }
        timeout_s_value = extra_kwargs.pop("timeout_s", self.rl_run_config.vllm.timeout_s)
        timeout_s = None if timeout_s_value is None else float(timeout_s_value)
        preflight_weight_path_value = extra_kwargs.pop("preflight_weight_path", True)
        require_hf_checkpoint_value = extra_kwargs.pop("require_hf_checkpoint", None)
        preflight_weight_path = _as_bool(preflight_weight_path_value, default=True)
        require_hf_checkpoint = (
            self.policy_sync_merge_to_hf
            if require_hf_checkpoint_value is None
            else _as_bool(require_hf_checkpoint_value, default=self.policy_sync_merge_to_hf)
        )
        extra_kwargs.pop("shared_checkpoint_dir", None)
        extra_kwargs.pop("policy_sync_dir", None)
        extra_kwargs.pop("update_weight_mode", None)
        extra_kwargs.pop("update_weight_transport", None)
        extra_kwargs.pop("update_weight_disk_dir", None)
        if timeout_s is not None:
            extra_kwargs.setdefault("timeout_s", timeout_s)
        return RayActorWeightSyncClient(
            actor_handles=list(self.ray_model_server_pool.actor_handles),
            actor_names=list(self.ray_model_server_pool.actor_names),
            namespace=self.ray_model_server_pool.namespace,
            timeout_s=timeout_s,
            preflight_weight_path=preflight_weight_path,
            require_hf_checkpoint=require_hf_checkpoint,
            extra_kwargs=extra_kwargs,
        )

    def _should_sync_policy_weights(self) -> bool:
        return self.trainer.global_step % max(1, self.rl_run_config.training.update_weights_every_steps) == 0

    def _sync_policy_weights(self, orchestrator, rank: int) -> ModelVersion:
        self._validate_update_weight_config()
        if rank == 0 and self.sync_drain_inflight_rollouts:
            self._drain_rollouts_for_policy_sync(orchestrator)

        fsdp_output_dir = self._policy_sync_fsdp_checkpoint_dir(self.trainer.global_step)
        if rank == 0 and fsdp_output_dir.exists():
            shutil.rmtree(fsdp_output_dir)
        _dist_barrier()
        self._save_policy_checkpoint(str(fsdp_output_dir), self.trainer.global_step, total_limit=None)
        _dist_barrier()

        model_version = None
        sync_error = None
        if rank == 0:
            try:
                checkpoint_path = self._prepare_policy_sync_checkpoint(fsdp_output_dir)
                model_version = ModelVersion(
                    version_id=self.trainer.global_step,
                    checkpoint_path=str(checkpoint_path.resolve()),
                    metadata={
                        "fsdp_checkpoint_path": str(fsdp_output_dir.resolve()),
                        "policy_sync_dir": str(self._policy_sync_root().resolve()),
                        "update_weight_mode": self.update_weight_mode,
                        "update_weight_transport": self.update_weight_transport,
                        "update_weight_path": str(checkpoint_path.resolve()),
                    },
                )
                result = orchestrator.reload_policy_weights(model_version)
                logger.info(f"Reloaded policy weights for version {self.trainer.global_step}: {result}")
                self._cleanup_policy_sync_checkpoints()
            except Exception:
                sync_error = traceback.format_exc()

        object_list = [(model_version, sync_error)]
        dist.broadcast_object_list(object_list, src=0)
        model_version, sync_error = object_list[0]
        if sync_error is not None:
            raise RuntimeError(f"Policy weight sync failed on rank 0:\n{sync_error}")
        return model_version

    def _drain_rollouts_for_policy_sync(self, orchestrator) -> None:
        if orchestrator is None:
            return

        orchestrator.rollout_manager.pause()
        started = time.perf_counter()
        accepted = 0
        while orchestrator.rollout_manager.inflight > 0:
            accepted += orchestrator.drain_rollouts(timeout_s=self.sync_drain_poll_s)
            if orchestrator.rollout_manager.inflight == 0:
                break
            if time.perf_counter() - started > self.sync_drain_timeout_s:
                stats = orchestrator.data_buffer.stats()
                raise TimeoutError(
                    "Timed out waiting for in-flight rollouts before policy weight sync: "
                    f"inflight={orchestrator.rollout_manager.inflight}, "
                    f"pending_trajectories={stats.pending_trajectories}, "
                    f"max_trajectories={self.rl_run_config.data_buffer.max_trajectories}. "
                    "Increase data_buffer.max_trajectories or reduce rollout concurrency if the buffer is full."
                )
            time.sleep(max(0.01, self.sync_drain_poll_s))
        accepted += orchestrator.drain_rollouts(timeout_s=0.0)
        logger.info(
            "Drained rollout work before policy sync: "
            f"accepted={accepted}, pending_trajectories={orchestrator.data_buffer.stats().pending_trajectories}"
        )

    def _prepare_policy_sync_checkpoint(self, fsdp_checkpoint_dir: Path) -> Path:
        if not self.policy_sync_merge_to_hf:
            return fsdp_checkpoint_dir

        output_dir = self._policy_sync_root() / _weight_sync_version_name(self.trainer.global_step)
        if output_dir.exists():
            shutil.rmtree(output_dir)

        from lmms_engine.merger.fsdp2 import FSDP2Merger

        model_general_type = getattr(self.config.model_config, "model_general_type", None)
        FSDP2Merger().merge(
            fsdp_checkpoint_dir,
            output_path=output_dir,
            model_general_type=model_general_type,
        )
        (output_dir / ".lmms_weight_sync_ready").write_text(
            f"version_id={self.trainer.global_step}\n",
            encoding="utf-8",
        )
        return output_dir

    def _policy_sync_fsdp_checkpoint_dir(self, step: int) -> Path:
        root = self._policy_sync_root()
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{_weight_sync_version_name(step)}-fsdp"

    def _policy_sync_root(self) -> Path:
        configured = self.policy_sync_dir
        if configured:
            root = Path(os.path.expandvars(os.path.expanduser(str(configured))))
            if not root.is_absolute():
                raise ValueError(
                    "trainer_args.rl_config.update_weight_disk_dir must be an absolute shared filesystem path. "
                    f"Got {configured!r}. Relative paths can resolve inside Ray trial working_dirs."
                )
            root = root.resolve()
        else:
            output_dir = Path(self.trainer.args.output_dir)
            if not output_dir.is_absolute():
                raise ValueError(
                    "Policy weight sync is enabled but trainer_args.rl_config.update_weight_disk_dir is not set. "
                    "Set it to an absolute shared filesystem path visible from all training and rollout nodes. "
                    f"trainer_args.output_dir={self.trainer.args.output_dir!r} is relative and unsafe under Ray."
                )
            root = (output_dir / "policy-sync").resolve()

        if _is_ray_session_path(root):
            raise ValueError(
                "Policy weight sync directory resolves inside a Ray session/artifact working directory, "
                f"which rollout actors may not see: {root}. Set trainer_args.rl_config.update_weight_disk_dir "
                "to an absolute shared filesystem path, for example /mnt/shared/<run>/policy-sync."
            )
        return root

    def _validate_update_weight_config(self) -> None:
        if self.update_weight_mode not in {"full", "delta"}:
            raise ValueError(
                "trainer_args.rl_config.update_weight_mode must be one of {'full', 'delta'}, "
                f"got {self.update_weight_mode!r}."
            )
        if self.update_weight_transport not in {"disk", "nccl"}:
            raise ValueError(
                "trainer_args.rl_config.update_weight_transport must be one of {'disk', 'nccl'}, "
                f"got {self.update_weight_transport!r}."
            )
        if (self.update_weight_mode, self.update_weight_transport) != ("full", "disk"):
            raise NotImplementedError(
                "Only slime-style full/disk weight update is implemented in lmms-engine today. "
                f"Requested update_weight_mode={self.update_weight_mode!r}, "
                f"update_weight_transport={self.update_weight_transport!r}. "
                "NCCL and delta disk support require a tensor/delta sender and are intentionally not silently emulated."
            )
        if not self.policy_sync_merge_to_hf:
            raise ValueError(
                "slime-style full/disk weight update requires policy_sync_merge_to_hf=true so vLLM reloads "
                "a complete Hugging Face checkpoint."
            )

    def _cleanup_policy_sync_checkpoints(self) -> None:
        keep_last = self.policy_sync_keep_last
        if keep_last < 1:
            return

        output_dir = self._policy_sync_root()
        sync_dirs = [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and _is_weight_sync_source_dir(path)
        ]
        sync_dirs.sort(key=_policy_sync_step)
        stale = sync_dirs[: max(0, len(sync_dirs) - keep_last)]
        for fsdp_dir in stale:
            hf_dir = _weight_sync_output_dir_for_source(fsdp_dir)
            shutil.rmtree(fsdp_dir, ignore_errors=True)
            shutil.rmtree(hf_dir, ignore_errors=True)

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
    data_buffer_config = dict(config.get("data_buffer", {}) or {})
    if "train_batch_size" in data_buffer_config:
        raise ValueError(
            "trainer_args.rl_config.data_buffer.train_batch_size has been removed. "
            "Use trainer_args.rl_config.data_buffer.train_batch_size_per_gpu instead."
        )
    return RLRunConfig(
        rollout=_build_dataclass(RolloutManagerConfig, config.get("rollout", {})),
        data_buffer=_build_dataclass(DataBufferConfig, data_buffer_config),
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


def _shard_train_batch(
    batch: TrainBatch,
    *,
    rank: int,
    world_size: int,
    max_steps_per_trajectory: int | None = None,
) -> TrainBatch:
    if world_size <= 1:
        return batch
    if len(batch.trajectories) < world_size:
        raise ValueError(
            "RL global train batch must contain at least one trajectory per FSDP rank: "
            f"got {len(batch.trajectories)} trajectories for world_size={world_size}. "
            "Increase trainer_args.rl_config.data_buffer.train_batch_size_per_gpu or reduce ray_train.num_workers."
        )

    local_trajectories, local_cost, global_costs = _balanced_trajectory_shard(
        batch.trajectories,
        rank=rank,
        world_size=world_size,
        max_steps_per_trajectory=max_steps_per_trajectory,
    )
    if not local_trajectories:
        raise ValueError(
            "RL train batch sharding produced an empty local batch: "
            f"rank={rank}, world_size={world_size}, global_trajectories={len(batch.trajectories)}."
        )

    return TrainBatch(
        batch_id=f"{batch.batch_id}-rank{rank}",
        model_version=batch.model_version,
        trajectories=local_trajectories,
        global_batch_size=batch.global_batch_size,
        payload=batch.payload,
        metadata={
            **batch.metadata,
            "global_batch_id": batch.batch_id,
            "global_num_trajectories": len(batch.trajectories),
            "local_num_trajectories": len(local_trajectories),
            "local_estimated_train_steps": local_cost,
            "global_estimated_train_steps_by_rank": global_costs,
            "sharding_strategy": "balanced_trajectory_steps",
            "rank": rank,
            "world_size": world_size,
        },
    )


def _balanced_trajectory_shard(
    trajectories,
    *,
    rank: int,
    world_size: int,
    max_steps_per_trajectory: int | None = None,
):
    base_quota = len(trajectories) // world_size
    remainder = len(trajectories) % world_size
    quotas = [base_quota + (1 if shard_rank < remainder else 0) for shard_rank in range(world_size)]
    buckets = [[] for _ in range(world_size)]
    costs = [0 for _ in range(world_size)]

    indexed_trajectories = list(enumerate(trajectories))
    indexed_trajectories.sort(
        key=lambda item: (
            -_estimated_trajectory_train_steps(item[1], max_steps_per_trajectory),
            item[0],
        )
    )
    for _index, trajectory in indexed_trajectories:
        eligible_ranks = [shard_rank for shard_rank, quota in enumerate(quotas) if len(buckets[shard_rank]) < quota]
        shard_rank = min(eligible_ranks, key=lambda candidate: (costs[candidate], len(buckets[candidate]), candidate))
        buckets[shard_rank].append(trajectory)
        costs[shard_rank] += _estimated_trajectory_train_steps(trajectory, max_steps_per_trajectory)

    return buckets[rank], costs[rank], costs


def _estimated_trajectory_train_steps(trajectory, max_steps_per_trajectory: int | None = None) -> int:
    num_steps = len(getattr(trajectory, "steps", []) or [])
    if max_steps_per_trajectory is not None and int(max_steps_per_trajectory) > 0:
        num_steps = min(num_steps, int(max_steps_per_trajectory))
    return max(1, num_steps)


def _coalesce_update_weight_disk_dir(
    rl_config: dict[str, Any],
    weight_sync_config: dict[str, Any],
    *,
    env_value: str | None = None,
) -> str | None:
    candidates = [
        ("update_weight_disk_dir", rl_config.get("update_weight_disk_dir")),
        ("weight_sync.update_weight_disk_dir", weight_sync_config.get("update_weight_disk_dir")),
        ("policy_sync_dir", rl_config.get("policy_sync_dir")),
        ("weight_sync.shared_checkpoint_dir", weight_sync_config.get("shared_checkpoint_dir")),
    ]
    values = [(name, str(value)) for name, value in candidates if value not in (None, "")]
    if not values:
        return env_value

    canonical = {
        str(Path(os.path.expandvars(os.path.expanduser(value))).resolve()) for _name, value in values
    }
    if len(canonical) > 1:
        details = ", ".join(f"{name}={value!r}" for name, value in values)
        raise ValueError(
            "Conflicting policy weight sync disk directories were configured. "
            "Use only update_weight_disk_dir, or make all aliases identical. "
            f"Got: {details}"
        )
    return values[0][1]


def _as_bool(value: Any, *, default: bool) -> bool:
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


def _weight_sync_version_name(step: int) -> str:
    return f"weight_v{int(step):06d}"


def _is_weight_sync_source_dir(path: Path) -> bool:
    name = path.name
    return (name.startswith("weight_v") and name.endswith("-fsdp")) or (
        name.startswith("policy-sync-") and not name.endswith("-hf")
    )


def _weight_sync_output_dir_for_source(source_dir: Path) -> Path:
    name = source_dir.name
    if name.startswith("weight_v") and name.endswith("-fsdp"):
        return source_dir.with_name(name.removesuffix("-fsdp"))
    if name.startswith("policy-sync-"):
        return source_dir.with_name(f"{name}-hf")
    return source_dir.with_name(f"{name}-hf")


def _policy_sync_step(path: Path) -> int:
    name = path.name
    if name.startswith("weight_v"):
        version = name.removeprefix("weight_v").removesuffix("-fsdp")
        try:
            return int(version)
        except ValueError:
            return -1
    try:
        return int(name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _dist_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _is_ray_session_path(path: Path) -> bool:
    value = path.as_posix()
    return "/tmp/ray/session_" in value or (
        "/ray/session_" in value and ("/working_dirs/" in value or "/artifacts/" in value)
    )


def _ray_init_kwargs_with_lmms_eval_path(ray_init_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = copy.deepcopy(ray_init_kwargs or {})
    lmms_eval_root = ensure_lmms_eval_importable()
    engine_src = Path(__file__).resolve().parents[3]
    pythonpath = _prepend_pythonpath([engine_src, lmms_eval_root])
    os.environ["PYTHONPATH"] = pythonpath

    runtime_env = dict(kwargs.get("runtime_env") or {})
    env_vars = dict(runtime_env.get("env_vars") or {})
    env_vars["PYTHONPATH"] = _prepend_pythonpath([engine_src, lmms_eval_root], env_vars.get("PYTHONPATH"))
    env_vars["PATH"] = _prepend_path([_python_bin_dir()], env_vars.get("PATH"))
    runtime_env["env_vars"] = env_vars
    kwargs["runtime_env"] = runtime_env
    return kwargs


def _prepend_pythonpath(paths: list[Path], existing: str | None = None) -> str:
    entries = [str(path) for path in paths]
    current = existing if existing is not None else os.environ.get("PYTHONPATH", "")
    entries.extend(item for item in current.split(os.pathsep) if item)
    deduped = list(dict.fromkeys(entries))
    return os.pathsep.join(deduped)


def _prepend_path(paths: list[Path], existing: str | None = None) -> str:
    entries = [str(path) for path in paths]
    current = existing if existing is not None else os.environ.get("PATH", "")
    entries.extend(item for item in current.split(os.pathsep) if item)
    return os.pathsep.join(dict.fromkeys(entries))


def _python_bin_dir() -> Path:
    bin_name = "Scripts" if os.name == "nt" else "bin"
    venv_bin = Path(sys.prefix) / bin_name
    if venv_bin.exists():
        return venv_bin
    return Path(sys.executable).parent


def _payload_metrics(payload: GRPOPayload) -> dict[str, float]:
    rewards = payload.tensors.get("sample_rewards") if isinstance(payload.tensors, dict) else None
    if rewards is None:
        return {}
    return {
        "rl/batch_reward_mean_host": float(rewards.float().mean().item()),
        "rl/batch_size": float(rewards.numel()),
    }


def _validate_ray_vllm_model_server(model_server: Any) -> None:
    if not isinstance(model_server, dict):
        raise ValueError("rl_config.model_server must be a Ray actor pool hosting vLLM.")
    backend = model_server.get("name") or model_server.get("backend")
    if backend != "ray_actor_pool":
        raise ValueError(f"rl_config.model_server.name must be 'ray_actor_pool', got {backend!r}.")

    server = model_server.get("server") or model_server.get("server_spec")
    if not isinstance(server, dict):
        raise ValueError("rl_config.model_server.server must be a dict vLLM server spec.")
    factory = server.get("factory")
    if factory != _VLLM_SERVER_FACTORY:
        raise ValueError(
            "rl_config.model_server.server.factory must be "
            f"{_VLLM_SERVER_FACTORY!r}; no non-vLLM rollout backend is allowed."
        )
