import json
import os
import pathlib
import random
import shutil
from copy import deepcopy
from functools import reduce

import numpy as np
import torch
import torch.distributed as dist
import yaml
from loguru import logger

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.mapping_func import (
    DATASET_MAPPING,
    create_model_from_config,
    create_model_from_pretrained,
)
from lmms_engine.models import MONKEY_PATCHER
from lmms_engine.models.utils import setup_flops_counter
from lmms_engine.parallel.parallelize import apply_parallelize
from lmms_engine.parallel.sequence_parallel.ulysses import (
    set_ulysses_sequence_parallel_group,
)
from lmms_engine.train.hf import Trainer

from ..utils.train_utils import TrainUtilities
from .config import TrainerConfig
from .registry import TRAINER_REGISTER

# from transformers import Trainer


class TrainRunner:
    """
    This is a base train runner to wrap all other trainer or your training logic
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.set_random_seed()
        self.train_dataset_config = config.dataset_config
        if config.dataset_config.eval_dataset_path is not None:
            self.eval_dataset_config = deepcopy(config.dataset_config)
            # Never use packing for eval dataset
            self.eval_dataset_config.packing = False
            self.eval_dataset_config.dataset_path = config.dataset_config.eval_dataset_path
        self.model_config = config.model_config
        self.config = config

    def build(self):
        if dist.is_initialized():
            self.create_sp_dis_group()
        self.model = self._build_model()
        if self.config.dataset_config.eval_dataset_path is not None:
            self.eval_dataset = self._build_eval_dataset()
        else:
            self.eval_dataset = None
        self.train_dataset = self._build_train_dataset()
        self._apply_monkey_patch()
        self.trainer = self._build_trainer()

    def _build_model(self):
        load_from_pretrained_path = self.model_config.load_from_pretrained_path
        load_from_config = self.model_config.load_from_config
        model_kwargs = self.model_config.extra_kwargs
        if load_from_pretrained_path is not None:
            model_class = create_model_from_pretrained(load_from_pretrained_path)
            model = model_class.from_pretrained(
                load_from_pretrained_path,
                attn_implementation=self.model_config.attn_implementation,
                torch_dtype=(torch.bfloat16 if self.config.trainer_args.bf16 else None),
                **model_kwargs,
            )
        elif load_from_config is not None:
            model_type = load_from_config.get("model_type", None)
            # Handle both nested and flat config structures
            init_config = load_from_config.get("config", None)
            if init_config is None:
                # If no nested config, use the load_from_config dict directly (excluding model_type)
                init_config = {k: v for k, v in load_from_config.items() if k != "model_type"}
            model_class, m_config = create_model_from_config(model_type, init_config)
            model = model_class.from_config(m_config, **model_kwargs)
        else:
            raise ValueError("No model name or pretrained path provided. Please provide one of them.")

        if self.model_config.overwrite_config:
            for key, value in self.model_config.overwrite_config.items():
                setattr(model.config, key, value)
                if getattr(model, key, None) is not None:
                    setattr(model, key, value)
                logger.info(f"Overwrite {key} to {value}")

        setup_flops_counter(model.config)
        logger.info(f"Model Structure: {model}")
        logger.info(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9} B")
        return model

    def _apply_monkey_patch(self):
        kwargs = {"use_rmpad": self.config.trainer_args.use_rmpad, "patch_type": []}
        if self.config.trainer_args.use_liger_kernel:
            kwargs["patch_type"].append("liger")
            # Overwrite the use_liger_kernel to False as we already apply the liger kernel by ourselves
            self.config.trainer_args.use_liger_kernel = False

        if self.model_config.monkey_patch_kwargs:
            patch_type = getattr(self.model_config.monkey_patch_kwargs, "patch_type", [])
            kwargs["patch_type"].extend(patch_type)
            kwargs.update(self.model_config.monkey_patch_kwargs)
        try:
            MONKEY_PATCHER.apply_monkey_patch_to_instance(self.model, **kwargs)
        except Exception as e:
            logger.error(f"Error applying monkey patch: {e}. Skip monkey patch.")

    def _build_train_dataset(self):
        dataset_cls = DATASET_MAPPING[self.train_dataset_config.dataset_type]
        dataset = dataset_cls(self.train_dataset_config)
        dataset.build()
        return dataset

    def _build_eval_dataset(self):
        dataset_cls = DATASET_MAPPING[self.eval_dataset_config.dataset_type]
        dataset = dataset_cls(self.eval_dataset_config)
        dataset.build()
        return dataset

    def set_random_seed(self, random_seed: int = 42):
        # Setting random seed for all
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        logger.info(f"Set random seed to {random_seed}")
        return random_seed

    def create_sp_dis_group(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        sp_ulysses_degree = self.config.trainer_args.sp_ulysses_degree
        sp_degree = sp_ulysses_degree * 1  # ring attn always 1, kept for clarity

        total_group_size = sp_degree
        assert (
            world_size % total_group_size == 0
        ), f"world_size={world_size} must be divisible by total_group_size={total_group_size}"

        set_ulysses_sequence_parallel_group(pgm.process_group_manager.cp_group)

    def _build_trainer(self):
        trainer_cls = TRAINER_REGISTER[self.config.trainer_type]
        trainer = trainer_cls(
            model=self.model,
            args=self.config.trainer_args,
            data_collator=self.train_dataset.get_collator(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.train_dataset.processor,
        )
        return trainer

    def run(self, **kwargs):
        if self.config.trainer_args.freeze_modules:
            for modules in self.config.trainer_args.freeze_modules:
                cls = reduce(lambda o, k: getattr(o, k, None), modules.split("."), self.model)
                if cls is not None:
                    for param in cls.parameters():
                        param.requires_grad = False

        if list(pathlib.Path(self.config.trainer_args.output_dir).glob("checkpoint-*")):
            self.trainer.train(resume_from_checkpoint=True)
        else:
            self.trainer.train()
        # Save the state for hf_trainer
        if hasattr(self.trainer, "save_state"):
            self.trainer.save_state()
            self.safe_save_model_for_hf_trainer(self.trainer, self.config.trainer_args.output_dir)

    def safe_save_model_for_hf_trainer(self, trainer: Trainer, output_dir: str):
        """Collects the state dict and dump to disk."""
        trainer.accelerator.wait_for_everyone()
        torch.cuda.synchronize()
        check_only_save_mm_adapter = self.config.trainer_args.only_save_mm_adapter
        logger.info(f"Only save projectors: {check_only_save_mm_adapter}")

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
                    torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
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
