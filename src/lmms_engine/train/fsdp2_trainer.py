import os
import random
import shutil
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import send_to_device
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
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
        self.default_backend = []
        if "wandb" in self.args.report_to:
            self.default_backend.append("wandb")
        self.default_backend.append("console")

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
        dataloader = StatefulDataLoader(dataset, **dataloader_params)
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
        self.train_dataloader = train_dataloader
        if self.eval_dataset is not None:
            raise NotImplementedError("Evaluation is not implemented")
        self.prepare_optimizer()
        self.steps_per_epoch = len(train_dataloader)
        self.total_steps = self.steps_per_epoch * self.args.num_train_epochs
        warmup_steps = (
            int(self.total_steps * self.args.warmup_ratio)
            if self.args.warmup_ratio > 0
            else self.args.warmup_steps
        )
        self.prepare_scheduler(warmup_steps, self.total_steps)
        rank = dist.get_rank()
        if rank == 0:
            self.tracking = Tracking(
                project_name=os.environ.get("WANDB_PROJECT", "lmms-engine"),
                experiment_name=self.args.run_name,
                default_backend=self.default_backend,
                config=self.args,
            )

        if resume_from_checkpoint:
            # Search for the latest checkpoint in the output_dir
            checkpoints = [
                f
                for f in os.listdir(self.args.output_dir)
                if f.startswith("checkpoint")
            ]
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = checkpoints[-1]
            self.load_checkpoints(
                os.path.join(self.args.output_dir, latest_checkpoint),
                int(latest_checkpoint.split("-")[1]),
            )
            start_epoch = int(latest_checkpoint.split("-")[1]) / self.steps_per_epoch
            # start_epoch is a float, we need to convert it to an integer
            start_epoch = int(start_epoch)
            self.global_step = int(latest_checkpoint.split("-")[1])
            need_update_pbar = True
        else:
            start_epoch = 0
            self.global_step = 0
            need_update_pbar = False
        Logging.info(f"Training with {self.args.num_train_epochs} epochs")

        for epoch in range(start_epoch, self.args.num_train_epochs):
            self.train_dataloader.sampler.set_epoch(epoch)
            pbar = tqdm(
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}",
                disable=dist.get_rank() != 0,
            )
            # if the checkpoint is loaded, we need to update the pbar
            # but we only need to update the pbar once
            if need_update_pbar:
                pbar.update(self.global_step)
                need_update_pbar = False
            for step, batch in enumerate(self.train_dataloader):
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

                epoch_progress = f"{self.global_step / self.steps_per_epoch:.2f}"
                train_metrics["epoch"] = float(epoch_progress)
                if rank == 0:
                    self.tracking.log(train_metrics)
                self.global_step += 1
                if self.should_save:
                    output_dir = os.path.join(
                        self.args.output_dir, f"checkpoint-{self.global_step}"
                    )
                    self.save_checkpoints(
                        output_dir,
                        self.global_step,
                        total_limit=self.args.save_total_limit,
                    )
                if self.global_step >= self.args.max_steps and self.args.max_steps > 0:
                    break
                pbar.update(1)
            pbar.close()

            if self.eval_dataset is not None:
                raise NotImplementedError("Evaluation is not implemented")

        # Save the final checkpoint
        output_dir = os.path.join(
            self.args.output_dir, f"checkpoint-{self.global_step}"
        )
        self.save_checkpoints(
            output_dir, self.global_step, total_limit=self.args.save_total_limit
        )

    def evaluate(self):
        raise NotImplementedError("Evaluation is not implemented")

    def remove_old_checkpoints(self, output_path: str, total_limit: int = None):
        if total_limit is None:
            return
        # get all checkpoints in output_path
        checkpoints = [f for f in os.listdir(output_path) if f.startswith("checkpoint")]
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        if len(checkpoints) > total_limit:
            for checkpoint in checkpoints[:-total_limit]:
                Logging.info(f"Removing checkpoint {checkpoint}")
                shutil.rmtree(os.path.join(output_path, checkpoint))

    def save_checkpoints(self, output_path: str, step: int, total_limit: int = None):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            os.makedirs(output_path, exist_ok=True)

        dist.barrier()
        model_path = os.path.join(
            output_path,
            "pytorch_model_fsdp_0",
            f"model_world_size_{world_size}_rank_{rank}.pt",
        )
        optim_path = os.path.join(
            output_path,
            "optimizer",
            f"optimizer_world_size_{world_size}_rank_{rank}.pt",
        )
        extra_state_path = os.path.join(
            output_path,
            "extra_state",
            f"extra_state_world_size_{world_size}_rank_{rank}.pt",
        )
        dataloader_state_path = os.path.join(
            output_path,
            "dataloader_state",
            f"dataloader_state_world_size_{world_size}_rank_{rank}.pt",
        )
        if rank == 0:
            os.makedirs(
                os.path.join(output_path, "pytorch_model_fsdp_0"), exist_ok=True
            )
            os.makedirs(os.path.join(output_path, "optimizer"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "extra_state"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "dataloader_state"), exist_ok=True)

        dist.barrier()

        torch.save(self.fsdp2_model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)
        extra_state = {
            "lr_scheduler_state": self.scheduler.state_dict(),
            "rng": self.get_rng_state(),
        }
        torch.save(extra_state, extra_state_path)
        torch.save(self.train_dataloader.state_dict(), dataloader_state_path)
        Logging.info(f"Saved checkpoint to {output_path} at step {step}")

        if rank == 0:
            self.processing_class.save_pretrained(output_path)
            self.model.config.save_pretrained(output_path)
            self.remove_old_checkpoints(
                self.args.output_dir, total_limit=self.args.save_total_limit
            )

        dist.barrier()

    @property
    def should_save(self):
        return self.global_step % self.args.save_steps == 0 and self.global_step > 0

    def load_checkpoints(self, output_path: str, step: int):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        model_path = os.path.join(
            output_path,
            "pytorch_model_fsdp_0",
            f"model_world_size_{world_size}_rank_{rank}.pt",
        )
        optim_path = os.path.join(
            output_path,
            "optimizer",
            f"optimizer_world_size_{world_size}_rank_{rank}.pt",
        )
        extra_state_path = os.path.join(
            output_path,
            "extra_state",
            f"extra_state_world_size_{world_size}_rank_{rank}.pt",
        )
        dataloader_state_path = os.path.join(
            output_path,
            "dataloader_state",
            f"dataloader_state_world_size_{world_size}_rank_{rank}.pt",
        )

        model_state_dict = torch.load(model_path, weights_only=False)
        self.fsdp2_model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(torch.load(optim_path, weights_only=False))
        extra_state = torch.load(extra_state_path, weights_only=False)
        self.load_rng_state(extra_state["rng"])
        self.scheduler.load_state_dict(extra_state["lr_scheduler_state"])
        self.train_dataloader.load_state_dict(
            torch.load(dataloader_state_path, weights_only=False)
        )
        Logging.info(f"Loaded checkpoint from {output_path} at step {step}")

    def get_rng_state(self):
        return {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

    def load_rng_state(self, rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])
