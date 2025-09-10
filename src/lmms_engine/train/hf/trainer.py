import functools
import importlib.metadata
import inspect
import os
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.data_loader import DataLoaderShard
from accelerate.utils import DataLoaderConfiguration, send_to_device
from packaging import version
from peft import PeftModel
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    RandomSampler,
    Sampler,
)
from transformers import Trainer as HFTrainer
from transformers.trainer import logger
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    LengthGroupedSampler,
    RandomSampler,
)
from transformers.trainer_utils import has_length, seed_worker
from transformers.utils import is_datasets_available, is_peft_available

import lmms_engine.models.utils as model_utils
import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.parallel.sequence_parallel.ulysses import (
    get_ulysses_sequence_parallel_world_size,
)
from lmms_engine.train.registry import TRAINER_REGISTER
from lmms_engine.utils.train_utils import TrainUtilities


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


TRAINER_STATE_NAME = "trainer_state.json"


@TRAINER_REGISTER.register("hf_trainer")
class Trainer(HFTrainer):
    def create_accelerator_and_postprocess(self):
        if self.args.fsdp2:
            if self.args.bf16:
                torch_dtype = torch.bfloat16
            elif self.args.fp16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            fsdp_plugin = FullyShardedDataParallelPlugin(
                fsdp_version=2,
                mixed_precision_policy={
                    "param_dtype": torch_dtype,
                    "reduce_dtype": torch_dtype,
                    "output_dtype": torch_dtype,
                },
                auto_wrap_policy="transformer_based_wrap",
                transformer_cls_names_to_wrap=self.args.fsdp_config.get(
                    "transformer_layer_cls_to_wrap", []
                ),
                activation_checkpointing=self.args.gradient_checkpointing,
                reshard_after_forward=self.args.fsdp_config.get(
                    "reshard_after_forward", True
                ),
            )
            accelerator_config = self.args.accelerator_config.to_dict()
            dataloader_params = [
                "split_batches",
                "dispatch_batches",
                "even_batches",
                "use_seedable_sampler",
            ]
            dataloader_config = DataLoaderConfiguration(
                **{param: accelerator_config.pop(param) for param in dataloader_params}
            )
            dataloader_config.data_seed = self.args.data_seed
            non_blocking = accelerator_config.pop("non_blocking")
            if non_blocking and not self.args.dataloader_pin_memory:
                logger.warning(
                    "`non_blocking` is enabled but `dataloader_pin_memory` is not. For the best performance, it's recommended to enable both."
                )
            dataloader_config.non_blocking = non_blocking
            # this would have been updated above, no need for it anymore
            accelerator_config.pop("gradient_accumulation_kwargs")

            args = {"fsdp_plugin": fsdp_plugin}
            args["dataloader_config"] = dataloader_config
            # create accelerator object
            self.accelerator = Accelerator(**args)
            # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
            self.gather_function = self.accelerator.gather_for_metrics

            if (
                "use_gather_object"
                in inspect.signature(self.gather_function).parameters.keys()
            ):
                self.gather_function = functools.partial(
                    self.gather_function,
                    use_gather_object=self.args.eval_use_gather_object,
                )

            self.is_deepspeed_enabled = (
                getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
            )
            self.is_fsdp_enabled = (
                getattr(self.accelerator.state, "fsdp_plugin", None) is not None
            )
            self.is_tp_enabled = (
                getattr(self.accelerator.state, "torch_tp_plugin", None) is not None
            )

            # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
            if (
                self.args.save_only_model
                and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
                and self.args.load_best_model_at_end
            ):
                wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
                raise ValueError(
                    f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`."
                )

            # `auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3
            if (
                self.is_deepspeed_enabled
                and self.accelerator.state.deepspeed_plugin.zero_stage == 3
                and self.args.auto_find_batch_size
            ):
                raise ValueError(
                    "`auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3. Please consider using Zero-2, Zero-1, or FSDP"
                )
            if (
                self.args.save_only_model
                and self.is_fsdp_enabled
                and "SHARDED_STATE_DICT"
                in str(self.accelerator.state.fsdp_plugin.state_dict_type)
            ):
                raise ValueError(
                    "save_only_model option is not compatible with FSDP state dict type 'SHARDED_STATE_DICT'"
                )
        else:
            return super().create_accelerator_and_postprocess()

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(
                self.train_dataset, datasets.Dataset
            ):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            # Hard code here because we use our own processing class
            model_input_name = None
            if self.args.sp_ulysses_degree > 1:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=self.train_dataset.modality_length,
                    model_input_name=model_input_name,
                    num_replicas=pgm.process_group_manager.dp_world_size,
                    rank=pgm.process_group_manager.dp_rank,
                )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=self.train_dataset.modality_length,
                model_input_name=model_input_name,
            )

        else:
            if self.args.sp_ulysses_degree > 1:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=pgm.process_group_manager.dp_world_size,
                    rank=pgm.process_group_manager.dp_rank,
                )
            else:
                return RandomSampler(self.train_dataset)

    def _get_dataloader(
        self,
        dataset,
        description,
        batch_size,
        sampler_fn=None,
        is_training=False,
        dataloader_key=None,
    ):
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(
                self.data_collator, description=description
            )

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )

        dataloader = DataLoader(dataset, **dataloader_params)

        # Accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version for eval dataloaders.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        # If using the Ulysses SP, we skip to prepare the dataloader
        if self.args.sp_ulysses_degree > 1:
            return dataloader
        else:
            return self.accelerator.prepare(dataloader)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        if self.args.sp_ulysses_degree > 1:
            batch_samples = []
            num_items_in_batch = None

            for _ in range(num_batches):
                try:
                    # Because we use the original pytorch dataloader, send the data to the device manually
                    data_sample = next(epoch_iterator)
                    data_sample = send_to_device(
                        data_sample,
                        self.accelerator.device,
                        non_blocking=self.accelerator.non_blocking,
                    )
                    batch_samples.append(data_sample)
                except StopIteration:
                    break

            count_num_items_in_batch = (
                len(batch_samples) > 0
                and "labels" in batch_samples[0]
                and (
                    # num_items_in_batch is passed to model forward
                    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3757
                    self.model_accepts_loss_kwargs
                    # num_items_in_batch is passed to compute_loss_func
                    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3773
                    or self.compute_loss_func is not None
                    # num_items_in_batch is also verified if (self.model_accepts_loss_kwargs or self.compute_loss_func)
                    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3790
                )
            )

            if count_num_items_in_batch:
                # For now we don't support object detection
                try:
                    num_items_in_batch = sum(
                        [(batch["labels"].ne(-100)).sum() for batch in batch_samples]
                    )
                except (TypeError, AttributeError):
                    pass

            if num_items_in_batch is not None:
                if self.args.average_tokens_across_devices:
                    num_items_in_batch = self.accelerator.gather(
                        num_items_in_batch
                    ).sum()

                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(device)

                    if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                        # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                        num_items_in_batch = num_items_in_batch.unsqueeze(0)

            return batch_samples, num_items_in_batch
        return super().get_batch_samples(epoch_iterator, num_batches, device)

    def _get_eval_sampler(self, eval_dataset: Optional[Dataset] = None):
        if eval_dataset is None or not has_length(eval_dataset):
            return None

        return RandomSampler(eval_dataset)

    def get_memory(self):
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
        mem = torch.cuda.memory_allocated()
        return peak_mem / 1e9, mem / 1e9

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "only_save_mm_adapter", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["multi_modal_projector", "audio_modal_projector"]

            weight_to_save = TrainUtilities.get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        else:
            if self.args.fsdp2:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(output_dir)
            super(Trainer, self)._save_checkpoint(model, trial)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if self.state.global_step == 0 or getattr(self, "cur_time", None) is None:
            self.cur_time = time.perf_counter()
            self.mfu = 0.0
            self.total_seq_len = []
        if self.state.global_step % 10 == 0 and self.state.global_step > 0:
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
            # Divide by the number of processes and the number of sequence parallel processes
            # Thus the mfu is within every dp group
            sp_size = pgm.process_group_manager.cp_world_size
            self.mfu = (
                flops_tensor.item() / self.args.world_size / sp_size / promised_flops
            )
            self.log({"mfu": round(self.mfu, 2)})
            self.flops = 0
            self.total_seq_len.clear()

        # Calculate the total number of tokens, sum at the row dimension
        self.total_seq_len.extend(
            inputs.get("attention_mask", torch.tensor(0))
            .sum(dim=1)
            .detach()
            .cpu()
            .tolist()
        )
        loss, outputs = super().compute_loss(
            model=model,
            inputs=inputs,
            num_items_in_batch=num_items_in_batch,
            return_outputs=True,
        )
        # Hf avg across every process, we scale the loss first such that the mean is over dp group
        loss = loss * get_ulysses_sequence_parallel_world_size()
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        args = self.args
        if args.use_muon:
            if self.optimizer is None:
                model = self.model

                def params_grouping(opt_model):
                    decay_parameters = self.get_decay_parameter_names(opt_model)
                    decay_muon_params_list = []
                    decay_adam_params_list = []
                    no_decay_muon_params_list = []
                    no_decay_adam_params_list = []
                    for name, param in opt_model.named_parameters():
                        if not param.requires_grad:
                            continue
                        patterns = [
                            "emb",
                            "norm",
                            "lm_head",
                            "bias",
                            "wte",
                            "wpe",
                            "output",
                            "a_proj",
                            "b_proj",
                            "conv1d",
                            "rotary",
                        ]
                        weight_decay = (
                            args.weight_decay if name not in decay_parameters else 0.0
                        )
                        use_muon = (param.ndim == 2) and (
                            not (any(pattern in name for pattern in patterns))
                        )
                        param.use_muon = use_muon
                        if use_muon and weight_decay > 0.0:
                            decay_muon_params_list.append(param)
                        elif use_muon and weight_decay == 0.0:
                            no_decay_muon_params_list.append(param)
                        elif not use_muon and weight_decay > 0.0:
                            decay_adam_params_list.append(param)
                        elif not use_muon and weight_decay == 0.0:
                            no_decay_adam_params_list.append(param)

                    params_groups = [
                        {
                            "params": decay_muon_params_list,
                            "use_muon": True,
                            "lr": args.learning_rate,
                            "momentum": 0.95,
                            "weight_decay": args.weight_decay,
                        },
                        {
                            "params": no_decay_muon_params_list,
                            "use_muon": True,
                            "lr": args.learning_rate,
                            "momentum": 0.95,
                            "weight_decay": 0.0,
                        },
                        {
                            "params": decay_adam_params_list,
                            "use_muon": False,
                            "lr": args.learning_rate,
                            "betas": (args.adam_beta1, args.adam_beta2),
                            "eps": args.adam_epsilon,
                            "weight_decay": args.weight_decay,
                        },
                        {
                            "params": no_decay_adam_params_list,
                            "use_muon": False,
                            "lr": args.learning_rate,
                            "betas": (args.adam_beta1, args.adam_beta2),
                            "eps": args.adam_epsilon,
                            "weight_decay": 0.0,
                        },
                    ]
                    return params_groups

                if self.is_fsdp_enabled and not self.args.fsdp2:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                    with FSDP.summon_full_params(model, writeback=False):
                        params_groups = params_grouping(model)
                else:
                    params_groups = params_grouping(model)

                from lmms_engine.utils.muon_utils import Muon

                self.optimizer = Muon(
                    params_groups,
                    defaults=dict(lr=args.learning_rate),
                    is_deepspeed_enabled=self.is_deepspeed_enabled,
                )
                return self.optimizer
        else:
            return super().create_optimizer()
