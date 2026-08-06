from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger

from lmms_engine.utils.fsdp2_utils import fsdp2_clip_grad_norm_


@dataclass(slots=True)
class RLPolicyLoss:
    """Loss package returned by an RL algorithm implementation."""

    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


class FSDP2RLPolicyStepMixin:
    """Shared FSDP2 policy optimization step for RL algorithms.

    Subclasses implement `compute_policy_loss`; this mixin owns the model-side
    update mechanics: grad accumulation, backward, clipping, optimizer step,
    EMA, scheduler, and distributed train/loss reduction.
    """

    def training_step(self, batch: Any) -> dict[str, float]:
        self.fsdp2_model.train()
        if self.accumulated_grad_steps == 0:
            self.optimizer.zero_grad()

        policy_loss = self.compute_policy_loss(batch)
        loss = self.prepare_policy_loss_for_backward(policy_loss.loss)
        loss_item = loss.detach().item() * self.args.gradient_accumulation_steps
        self.backward_policy_loss(loss)

        grad_norm = self.maybe_step_policy_optimizer()
        return self.policy_step_metrics(
            loss_item=loss_item,
            algorithm_metrics=policy_loss.metrics,
            grad_norm=grad_norm,
        )

    def compute_policy_loss(self, batch: Any) -> RLPolicyLoss:
        raise NotImplementedError

    def prepare_policy_loss_for_backward(self, loss: torch.Tensor) -> torch.Tensor:
        if dist.get_world_size() > 1:
            loss = loss.mean()
        return loss / self.args.gradient_accumulation_steps

    def backward_policy_loss(self, loss: torch.Tensor) -> None:
        loss.backward()

    def maybe_step_policy_optimizer(self) -> torch.Tensor | None:
        self.accumulated_grad_steps += 1
        should_update = self.accumulated_grad_steps >= self.args.gradient_accumulation_steps
        if not should_update:
            return None

        grad_norm = self.clip_policy_grad_norm()
        if not torch.isfinite(grad_norm):
            logger.warning(f"grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.step_policy_optimizer()

        self.scheduler.step()
        self.accumulated_grad_steps = 0
        return grad_norm

    def clip_policy_grad_norm(self) -> torch.Tensor:
        return fsdp2_clip_grad_norm_(self.fsdp2_model.parameters(), self.args.max_grad_norm)

    def step_policy_optimizer(self) -> None:
        self.optimizer.step()
        self.ema.update(step=self.global_step + 1)

    def policy_step_metrics(
        self,
        *,
        loss_item: float,
        algorithm_metrics: dict[str, float],
        grad_norm: torch.Tensor | None,
    ) -> dict[str, float]:
        reduced_loss = torch.tensor(loss_item, device=self.args.device)
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)
        metrics = {
            "train/loss": reduced_loss.item(),
            "train/lr": self.scheduler.get_last_lr()[0],
            **algorithm_metrics,
        }
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm.item()
        return metrics
