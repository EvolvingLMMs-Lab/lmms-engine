import functools
import importlib.metadata
import inspect
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DataLoaderConfiguration
from packaging import version
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler
from transformers import Trainer as HFTrainer
from transformers.trainer import logger
from transformers.trainer_pt_utils import LengthGroupedSampler, RandomSampler
from transformers.trainer_utils import has_length
from transformers.utils import is_datasets_available, is_peft_available

from ..utils.train_utils import TrainUtilities


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


TRAINER_STATE_NAME = "trainer_state.json"


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
            # model_input_name = (
            # self.processing_class.model_input_names[0]
            # if self.processing_class is not None
            # else None
            # )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=self.train_dataset.modality_length,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.train_dataset)

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
            self.flops = 0
        if (
            self.state.global_step % 10 == 0
            and self.flops > 0  # No flops logging for this model
        ):
            prev_time = self.cur_time
            self.cur_time = time.perf_counter()
            device = self.args.local_rank
            flops_tensor = torch.tensor(self.flops, device=device)
            torch.distributed.all_reduce(
                flops_tensor, op=torch.distributed.ReduceOp.SUM
            )
            self.mfu = (
                flops_tensor.item()
                / (self.cur_time - prev_time)
                / self.args.world_size
                / TrainUtilities.get_device_flops("B")
            )
            self.log({"mfu": round(self.mfu, 2)})
            self.flops = 0
        loss, outputs = super().compute_loss(
            model=model,
            inputs=inputs,
            num_items_in_batch=num_items_in_batch,
            return_outputs=True,
        )
        self.flops += outputs.get("flops", 0)
        return (loss, outputs) if return_outputs else loss
