import os
import pathlib

import torch
import torch.distributed as dist
from transformers import Trainer

from lmms_engine.models.sequence_parallel.ulysses import (
    set_ulysses_sequence_parallel_group,
)

from ..utils import Logging
from ..utils.train_utils import TrainUtilities
from .base_trainer import BaseTrainer
from .config import TrainerConfig
from .custom import LLaVATrainer


class Hf_Trainer(BaseTrainer):
    def __init__(self, config: TrainerConfig) -> None:
        self.set_random_seed()
        super().__init__(config)

    def build(self):
        super().build()
        self.create_sp_dis_group()
        self.trainer = self._build_trainer()

    def create_sp_dis_group(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        sp_ulysses_degree = self.config.trainer_args.sp_ulysses_degree
        sp_degree = sp_ulysses_degree * 1  # ring attn always 1, kept for clarity

        total_group_size = sp_degree * sp_ulysses_degree
        assert (
            world_size % total_group_size == 0
        ), f"world_size={world_size} must be divisible by total_group_size={total_group_size}"

        dp_degree = world_size // total_group_size

        for dp_idx in range(dp_degree):
            for sp_idx in range(sp_degree):
                # Each group contains sp_ulysses_degree ranks
                group_ranks = []
                for i in range(sp_ulysses_degree):
                    global_rank = (
                        dp_idx * total_group_size + sp_idx * sp_ulysses_degree + i
                    )
                    group_ranks.append(global_rank)

                group = torch.distributed.new_group(group_ranks)

                if rank in group_ranks:
                    set_ulysses_sequence_parallel_group(group)

    def _build_trainer(self):
        if self.config.trainer_args_type == "sft":
            trainer = LLaVATrainer(
                model=self.model,
                args=self.config.trainer_args,
                data_collator=self.train_dataset.get_collator(),
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.train_dataset.processor,
            )
        else:
            raise NotImplementedError(
                f"Unknown trainer args type: {self.config.trainer_args_type}"
            )
        return trainer

    def run(self, **kwargs):
        self.save_config()
        if self.config.trainer_args.freeze_modules:
            for modules in self.config.trainer_args.freeze_modules:
                cls = getattr(self.model, modules, None)
                if cls is not None:
                    for param in cls.parameters():
                        param.requires_grad = False

        if list(pathlib.Path(self.config.trainer_args.output_dir).glob("checkpoint-*")):
            self.trainer.train(resume_from_checkpoint=True)
        else:
            self.trainer.train()
        self.trainer.save_state()
        self.safe_save_model_for_hf_trainer(
            self.trainer, self.config.trainer_args.output_dir
        )

    def safe_save_model_for_hf_trainer(self, trainer: Trainer, output_dir: str):
        """Collects the state dict and dump to disk."""
        trainer.accelerator.wait_for_everyone()
        torch.cuda.synchronize()
        check_only_save_mm_adapter = self.config.trainer_args.only_save_mm_adapter
        Logging.info(f"Only save projectors: {check_only_save_mm_adapter}")

        if check_only_save_mm_adapter:
            # Only save Adapter
            keys_to_match = ["multi_modal_projector", "audio_modal_projector"]

            weight_to_save = TrainUtilities.get_mm_adapter_state_maybe_zero_3(
                trainer.model.named_parameters(), keys_to_match
            )
            trainer.model.config.save_pretrained(output_dir)

            current_folder = output_dir.split("/")[-1]
            parent_folder = os.path.dirname(output_dir)
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith("checkpoint-"):
                    mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                    os.makedirs(mm_projector_folder, exist_ok=True)
                    torch.save(
                        weight_to_save,
                        os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                    )
                else:
                    torch.save(
                        weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                    )
            return
        if trainer.deepspeed:
            trainer.save_model(output_dir)
            return
        if self.config.trainer_args.fsdp2:
            # For fsdp we merge the shards into a single checkpoint after the training is done
            if trainer.processing_class is not None:
                trainer.processing_class.save_pretrained(output_dir)
            return

        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
