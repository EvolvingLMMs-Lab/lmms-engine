import os
import time
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import send_to_device
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
from transformers.trainer_utils import seed_worker

import lmms_engine.models.utils as model_utils
import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.utils.fsdp2_utils import (
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)
from lmms_engine.utils.logging_utils import Logging
from lmms_engine.utils.tracking import Tracking

from .config import TrainingArguments


class FSDP2SFTTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        processing_class=None,
        data_collator=None,
    ) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.data_collator = data_collator

    def prepare_dataloader(self, dataset: Dataset, is_training: bool = True):
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self.args.train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if self.args.group_by_length:
            sampler = DistributedLengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=dataset,
                lengths=dataset.modality_length,
                model_input_name=None,
                num_replicas=pgm.process_group_manager.dp_world_size,
                rank=pgm.process_group_manager.dp_rank,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=pgm.process_group_manager.dp_world_size,
                rank=pgm.process_group_manager.dp_rank,
            )
        dataloader_params["sampler"] = sampler
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        if is_training:
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=pgm.process_group_manager.dp_rank,
            )
        dataloader = DataLoader(dataset, **dataloader_params)
        return dataloader

    def prepare_model(self):
        if self.args.bf16:
            param_dtype = torch.bfloat16
        else:
            param_dtype = torch.float16

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        reduce_dtype = getattr(torch, self.args.reduce_dtype)
        output_dtype = getattr(torch, self.args.output_dtype)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=output_dtype,
        )

        fsdp_kwargs = {
            "reshard_after_forward": getattr(self.args, "fsdp_config", {}).get(
                "reshard_after_forward", True
            ),
            "mp_policy": mp_policy,
        }

        transformer_cls_names_to_wrap = self.args.fsdp_config.get(
            "transformer_layer_cls_to_wrap", None
        )
        full_state = self.model.state_dict()
        Logging.info(f"Applying FSDP2 to model")
        apply_fsdp2(self.model, fsdp_kwargs, transformer_cls_names_to_wrap)
        Logging.info(f"Loading full state dict to model")
        fsdp2_load_full_state_dict(self.model, full_state)
        Logging.info(f"FSDP2 applied to model")
        self.fsdp2_model = self.model

    def prepare_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.fsdp2_model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

    def prepare_scheduler(
        self,
        num_warmup_steps: int,
        num_training_steps: int,
    ):
        if self.args.lr_scheduler_type:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.args.lr_scheduler_type == "wsd":
            self.scheduler = get_wsd_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {self.args.lr_scheduler_type}"
            )

    def compute_loss(self, batch):
        if self.args.bf16:
            cast_dtype = torch.bfloat16
        else:
            cast_dtype = torch.float16
        with torch.autocast(device_type="cuda", dtype=cast_dtype):
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return loss

    def training_step(self, batch):
        self.fsdp2_model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss_item = loss.item()
        loss.backward()
        grad_norm = fsdp2_clip_grad_norm_(
            self.fsdp2_model.parameters(), self.args.max_grad_norm
        )

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        self.scheduler.step()

        # reduce loss across dp ranks
        lr = self.scheduler.get_last_lr()[0]
        loss_item = torch.tensor(loss_item, device=self.args.device)
        torch.distributed.all_reduce(loss_item, op=torch.distributed.ReduceOp.AVG)
        return {
            "loss": loss_item.item(),
            "lr": lr,
            "grad_norm": grad_norm.item(),
        }

    def validation_step(self):
        pass

    def train(self, resume_from_checkpoint: bool = False):
        self.prepare_model()
        train_dataloader = self.prepare_dataloader(self.train_dataset, is_training=True)
        if self.eval_dataset is not None:
            raise NotImplementedError("Evaluation is not implemented")
        self.prepare_optimizer()
        self.steps_per_epoch = len(train_dataloader)
        self.total_steps = self.steps_per_epoch * self.args.num_train_epochs
        self.prepare_scheduler(self.args.warmup_steps, self.total_steps)
        rank = dist.get_rank()
        if rank == 0:
            self.tracking = Tracking(
                project_name=os.environ.get("WANDB_PROJECT", "lmms-engine"),
                experiment_name=self.args.run_name,
                config=self.args,
            )

        if resume_from_checkpoint:
            raise NotImplementedError("Resume from checkpoint is not implemented")

        Logging.info(f"Training with {self.args.num_train_epochs} epochs")

        start_epoch = 0

        for epoch in range(start_epoch, self.args.num_train_epochs):
            train_dataloader.sampler.set_epoch(epoch)
            pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=dist.get_rank() != 0,
            )
            for step, batch in enumerate(pbar):
                # send batch to device
                batch = send_to_device(batch, self.fsdp2_model.device)
                start_time = time.perf_counter()
                train_metrics = self.training_step(batch)
                end_time = time.perf_counter()
                delta_time = end_time - start_time
                seq_len = (
                    batch.get("attention_mask", torch.tensor(0))
                    .sum(dim=1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                flops, promised_flops = model_utils.flops_counter.estimate_flops(
                    seq_len, delta_time=delta_time
                )
                device = self.fsdp2_model.device
                flops_tensor = torch.tensor(flops, device=device)
                torch.distributed.all_reduce(
                    flops_tensor, op=torch.distributed.ReduceOp.SUM
                )
                sp_size = pgm.process_group_manager.cp_world_size
                mfu = (
                    flops_tensor.item()
                    / self.args.world_size
                    / sp_size
                    / promised_flops
                )
                train_metrics["mfu"] = round(mfu, 2)

                epoch_progress = f"{step / len(train_dataloader):.2f}"
                train_metrics["epoch"] = epoch_progress
                if rank == 0:
                    self.tracking.log(train_metrics)

            if self.eval_dataset is not None:
                raise NotImplementedError("Evaluation is not implemented")

    def evaluate(self):
        raise NotImplementedError("Evaluation is not implemented")
