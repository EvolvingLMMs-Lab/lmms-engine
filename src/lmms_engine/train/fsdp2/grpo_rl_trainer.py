from __future__ import annotations

import os
import shutil
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.utils import send_to_device
from loguru import logger

from lmms_engine.train.config import TrainingArguments
from lmms_engine.train.fsdp2.fsdp2_trainer import FSDP2SFTTrainer
from lmms_engine.train.fsdp2.rl_policy_step import FSDP2RLPolicyStepMixin, RLPolicyLoss
from lmms_engine.train.registry import TRAINER_REGISTER
from lmms_engine.train.rl.grpo import GRPOPayload


@TRAINER_REGISTER.register("fsdp2_grpo_rl_trainer")
class FSDP2GRPORLTrainer(FSDP2RLPolicyStepMixin, FSDP2SFTTrainer):
    """FSDP2-only GRPO policy trainer.

    Online rollout scheduling is owned by `lmms_engine.train.rl.RLTrainRunner`.
    This class only owns model preparation, policy optimization, loss,
    checkpointing, and validation hooks.
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
        self._submitted_rollouts = 0
        self._last_policy_checkpoint: str | None = None

    def train(self, resume_from_checkpoint: bool = False):
        raise RuntimeError(
            "FSDP2GRPORLTrainer is driven by RLTrainRunner. "
            "Launch with `python -m lmms_engine.launch.rl ...` or the generic launch CLI."
        )

    def prepare_rl_batch(self, payload: GRPOPayload | dict[str, Any]) -> dict[str, Any]:
        tensors = payload.tensors if isinstance(payload, GRPOPayload) else payload
        if tensors is None:
            raise ValueError("GRPO payload does not contain tensors. Build it with GRPOBatchAdapter and a processor.")
        return send_to_device(tensors, self.fsdp2_model.device, non_blocking=True)

    def train_batch(self, batch: dict[str, Any]) -> dict[str, float]:
        return self.training_step(batch)

    def compute_policy_loss(self, batch: dict[str, Any]) -> RLPolicyLoss:
        loss, metrics = self.compute_grpo_loss(batch)
        return RLPolicyLoss(loss=loss, metrics=metrics)

    def compute_grpo_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        labels = batch["labels"]
        advantages = batch["sample_advantages"].to(dtype=torch.float32, device=labels.device).view(-1)
        model_inputs = {
            key: value
            for key, value in batch.items()
            if key
            not in {
                "labels",
                "sample_advantages",
                "sample_rewards",
                "sample_metadata",
                "sample_ref_logprobs",
                "sample_old_logprobs",
            }
        }

        cast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16
        with torch.autocast(device_type="cuda", dtype=cast_dtype):
            outputs = self.model(**model_inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        response_mask = shift_labels.ne(-100)
        token_log_probs = -F.cross_entropy(
            shift_logits.transpose(1, 2),
            shift_labels,
            ignore_index=-100,
            reduction="none",
        )
        token_log_probs = token_log_probs * response_mask

        token_counts = response_mask.sum(dim=-1).clamp_min(1)
        sample_log_probs = token_log_probs.sum(dim=-1) / token_counts
        beta = float((self.rl_config.get("algorithm") or {}).get("beta", 0.0))
        loss = -(advantages * sample_log_probs).mean()
        kl_mean = None
        if "sample_ref_logprobs" in batch:
            ref_log_probs = batch["sample_ref_logprobs"].to(dtype=torch.float32, device=labels.device).view(-1)
            if ref_log_probs.numel() != sample_log_probs.numel():
                raise ValueError(
                    "sample_ref_logprobs must have one value per GRPO sample: "
                    f"got {ref_log_probs.numel()} ref logprobs for {sample_log_probs.numel()} samples."
                )
            kl = sample_log_probs.float() - ref_log_probs
            kl_mean = kl.mean()
            if beta:
                loss = loss + beta * kl_mean
        elif beta:
            raise RuntimeError(
                "GRPO beta > 0 requires reference logprob annotations, but batch has no sample_ref_logprobs. "
                "Start a reference model role and annotate trajectories before training, or set algorithm.beta=0."
            )

        reward = batch["sample_rewards"].to(dtype=torch.float32, device=labels.device).view(-1)
        metrics = {
            "rl/reward_mean": _distributed_mean(reward.mean()),
            "rl/advantage_mean": _distributed_mean(advantages.mean()),
            "rl/response_tokens_mean": _distributed_mean(token_counts.float().mean()),
            "rl/policy_logprob_mean": _distributed_mean(sample_log_probs.detach().mean()),
        }
        if kl_mean is not None:
            metrics["rl/ref_kl_mean"] = _distributed_mean(kl_mean.detach())
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

    def remove_old_checkpoints(self, output_path: str, total_limit: int = None):
        if total_limit is None:
            return
        checkpoints = [item for item in os.listdir(output_path) if item.startswith("checkpoint")]
        checkpoints.sort(key=lambda item: int(item.split("-")[1]))
        if len(checkpoints) > total_limit:
            for checkpoint in checkpoints[:-total_limit]:
                shutil.rmtree(os.path.join(output_path, checkpoint))


def _distributed_mean(value: torch.Tensor) -> float:
    reduced = value.detach().float()
    dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
    return float(reduced.item())
