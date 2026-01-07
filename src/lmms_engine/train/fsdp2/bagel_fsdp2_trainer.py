from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from torch.utils.data import Dataset, IterableDataset

from lmms_engine.train.config import TrainingArguments
from lmms_engine.train.fsdp2.fsdp2_trainer import FSDP2SFTTrainer
from lmms_engine.train.registry import TRAINER_REGISTER

DatasetType = Union[Dataset, IterableDataset]


def _as_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if torch.is_tensor(x):
        if x.numel() == 0:
            return None
        return float(x.detach().float().mean().item())
    try:
        return float(x)
    except Exception:
        return None


@TRAINER_REGISTER.register("bagel_fsdp2_trainer")
class BagelFSDP2Trainer(FSDP2SFTTrainer):
    """
    Bagel-specific FSDP2 SFT trainer that logs Bagel loss components (CE/MSE) to W&B/console via Tracking.

    It intentionally reuses the base FSDP2SFTTrainer training loop behavior (optimizer, scheduler, EMA),
    but extracts Bagel's `loss_dict` / token counts from model outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: DatasetType,
        eval_dataset: DatasetType = None,
        processing_class=None,
        data_collator=None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            data_collator=data_collator,
        )

    def training_step(self, batch):
        self.fsdp2_model.train()
        if self.accumulated_grad_steps == 0:
            self.optimizer.zero_grad()

        if self.args.bf16:
            cast_dtype = torch.bfloat16
        else:
            cast_dtype = torch.float16

        # Forward (keep outputs so we can log Bagel CE/MSE)
        with torch.autocast(device_type="cuda", dtype=cast_dtype):
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Normalize loss for grad accumulation
        if dist.get_world_size() > 1:
            loss = loss.mean()
        loss = loss / self.args.gradient_accumulation_steps
        loss_item = loss.item() * self.args.gradient_accumulation_steps

        loss.backward()
        self.accumulated_grad_steps += 1
        should_update = self.accumulated_grad_steps >= self.args.gradient_accumulation_steps

        grad_norm = None
        if should_update:
            from lmms_engine.utils.fsdp2_utils import fsdp2_clip_grad_norm_

            grad_norm = fsdp2_clip_grad_norm_(self.fsdp2_model.parameters(), self.args.max_grad_norm)
            did_step = False
            if not torch.isfinite(grad_norm):
                logger.warning(f"grad_norm is not finite: {grad_norm}. Skip optimizer step.")
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()
                did_step = True

            self.scheduler.step()
            self.accumulated_grad_steps = 0
            if did_step:
                # global_step is incremented by the outer train loop after accumulation completes.
                self.ema.update(step=self.global_step + 1)

        # Prepare metrics (DP-averaged scalars; tokens are summed)
        lr = self.scheduler.get_last_lr()[0]
        device = self.args.device

        loss_tensor = torch.tensor(loss_item, device=device)
        if dist.get_world_size() > 1:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        metrics = {
            "train/loss": loss_tensor.item(),
            "train/lr": lr,
        }
        if grad_norm is not None:
            metrics["train/grad_norm"] = float(grad_norm.detach().float().item())

        if isinstance(outputs, dict):
            loss_dict = outputs.get("loss_dict", None)
            ce = None
            mse = None
            if isinstance(loss_dict, dict):
                ce = loss_dict.get("ce", None)
                mse = loss_dict.get("mse", None)

            ce_val = _as_float(ce)
            mse_val = _as_float(mse)

            if ce_val is not None:
                ce_tensor = torch.tensor(ce_val, device=device)
                if dist.get_world_size() > 1:
                    dist.all_reduce(ce_tensor, op=dist.ReduceOp.AVG)
                metrics["train/ce"] = ce_tensor.item()

            if mse_val is not None:
                mse_tensor = torch.tensor(mse_val, device=device)
                if dist.get_world_size() > 1:
                    dist.all_reduce(mse_tensor, op=dist.ReduceOp.AVG)
                metrics["train/mse"] = mse_tensor.item()

            # Also log weighted components (what actually contributes to the total loss)
            ce_w = _as_float(getattr(self.model.config, "ce_weight", 1.0))
            mse_w = _as_float(getattr(self.model.config, "mse_weight", 1.0))
            if ce_w is not None:
                metrics["train/ce_weight"] = ce_w
            if mse_w is not None:
                metrics["train/mse_weight"] = mse_w
            if ce_val is not None and ce_w is not None:
                metrics["train/ce_weighted"] = ce_val * ce_w
            if mse_val is not None and mse_w is not None:
                metrics["train/mse_weighted"] = mse_val * mse_w

            # Token stats (sum across ranks)
            ce_tokens = outputs.get("total_ce_tokens", None)
            mse_tokens = outputs.get("total_mse_tokens", None)
            if torch.is_tensor(ce_tokens):
                ce_tokens_t = ce_tokens.detach().float().to(device=device)
                if dist.get_world_size() > 1:
                    dist.all_reduce(ce_tokens_t, op=dist.ReduceOp.SUM)
                metrics["train/ce_tokens"] = ce_tokens_t.item()
            if torch.is_tensor(mse_tokens):
                mse_tokens_t = mse_tokens.detach().float().to(device=device)
                if dist.get_world_size() > 1:
                    dist.all_reduce(mse_tokens_t, op=dist.ReduceOp.SUM)
                metrics["train/mse_tokens"] = mse_tokens_t.item()

        return metrics
