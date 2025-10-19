import contextlib
import os
import pdb
import shutil
import time
from typing import Any, Literal, Optional, Union

import torch
import torch.nn as nn
from transformers.utils import is_torch_xla_available

import lmms_engine.models.utils as model_utils
import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.train.hf.trainer import Trainer as HFTrainer
from lmms_engine.train.registry import TRAINER_REGISTER


@TRAINER_REGISTER.register("dllm_trainer")
class DLLMTrainer(HFTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon: float = 0.1
        self.ignore_index: int = -100
        self.tr_nll = None
        self.do_log_nll_step = -1
        self.step = -1

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        assert "labels" in inputs, "labels must be in inputs"
        assert "mlm_prob" in inputs, "mlm_prob must be in inputs"
        assert "attention_mask" in inputs, "attention_mask must be in inputs"
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}

        zero_config = (
            getattr(model.config, "zero_optimization", None)
            if hasattr(model, "config")
            else None
        )
        if zero_config:
            zero_stage = (
                zero_config.get("stage", None)
                if isinstance(zero_config, dict)
                else getattr(zero_config, "stage", None)
            )
            inputs["zero_stage"] = zero_stage

        if self.state.global_step == 0 or getattr(self, "cur_time", None) is None:
            self.cur_time = time.perf_counter()
            self.mfu = 0.0
            self.total_seq_len = []

        self.total_seq_len.extend(
            inputs.get("attention_mask", torch.tensor(0))
            .sum(dim=1)
            .detach()
            .cpu()
            .tolist()
        )

        self.step += 1
        do_log_nll_step = (self.step + 1) % self.args.gradient_accumulation_steps == 0
        outputs = model(**inputs)
        d_loss, nll = outputs.loss, outputs.nll

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            d_loss *= self.accelerator.num_processes
            nll *= self.accelerator.num_processes

        tr_nll_step = nll.mean() if self.args.n_gpu > 1 else nll
        self.tr_nll = tr_nll_step if self.tr_nll is None else self.tr_nll + tr_nll_step
        return d_loss

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_nll_scalar = self._nested_gather(self.tr_nll).mean().item()
            tr_loss -= tr_loss
            self.tr_nll -= self.tr_nll

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["nll"] = round(
                tr_nll_scalar / (self.state.global_step - self._globalstep_last_logged),
                4,
            )

            prev_time = self.cur_time
            self.cur_time = time.perf_counter()
            device = self.args.local_rank

            flops, promised_flops = model_utils.flops_counter.estimate_flops(
                self.total_seq_len, delta_time=self.cur_time - prev_time
            )
            flops_tensor = torch.tensor(flops, device=device)
            torch.distributed.all_reduce(
                flops_tensor, op=torch.distributed.ReduceOp.SUM
            )
            sp_size = pgm.process_group_manager.cp_world_size
            mfu = flops_tensor.item() / self.args.world_size / sp_size / promised_flops
            logs["mfu"] = round(mfu, 2)
            self.flops = 0
            self.total_seq_len.clear()

            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)
