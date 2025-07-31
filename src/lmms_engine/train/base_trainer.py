import json
import os
import random
import shutil
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch
import yaml

from lmms_engine.mapping_func import (
    DATASET_MAPPING,
    create_model_from_config,
    create_model_from_pretrained,
)

from ..models.kernels import CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN
from ..models.kernels import (
    _apply_liger_kernel_to_instance as _apply_liger_kernel_to_custom_instance,
)
from ..utils import Logging
from ..utils.train_utils import TrainUtilities
from .config import TrainerConfig


class BaseTrainer(ABC):
    """
    This is a base trainer wrapper to wrap all other trainer or your training logic
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.train_dataset_config = config.dataset_config
        if config.dataset_config.eval_dataset_path is not None:
            self.eval_dataset_config = deepcopy(config.dataset_config)
            # Never use packing for eval dataset
            self.eval_dataset_config.packing = False
            self.eval_dataset_config.dataset_path = (
                config.dataset_config.eval_dataset_path
            )
        self.model_config = config.model_config
        self.config = config

    def build(self):
        self.model = self._build_model()
        if self.config.dataset_config.eval_dataset_path is not None:
            self.eval_dataset = self._build_eval_dataset()
        else:
            self.eval_dataset = None
        self.train_dataset = self._build_train_dataset()
        if self.model_config.pretrain_mm_mlp_adapter is not None:
            self._load_mm_projector()
        if self.config.trainer_args.use_liger_kernel:
            self._apply_liger_kernel()
            # Set to False as we already apply the liger kernel by ourselves
            self.config.trainer_args.use_liger_kernel = False

    def _build_model(self):
        load_from_pretrained_path = self.model_config.load_from_pretrained_path
        load_from_config = self.model_config.load_from_config
        if load_from_pretrained_path is not None:
            model_class = create_model_from_pretrained(load_from_pretrained_path)
            model = model_class.from_pretrained(
                load_from_pretrained_path,
                attn_implementation=self.model_config.attn_implementation,
                torch_dtype=(torch.bfloat16 if self.config.trainer_args.bf16 else None),
            )
        elif load_from_config is not None:
            model_type = load_from_config.get("model_type", None)
            init_config = load_from_config.get("config", None)
            model_class, m_config = create_model_from_config(model_type, init_config)
            model = model_class.from_config(m_config)
        else:
            raise ValueError(
                "No model name or pretrained path provided. Please provide one of them."
            )

        if self.model_config.overwrite_config:
            for key, value in self.model_config.overwrite_config.items():
                setattr(model.config, key, value)
                Logging.info(f"Overwrite {key} to {value}")

        Logging.info(f"Model Structure: {model}")
        Logging.info(
            f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9} GB"
        )
        return model

    def _apply_liger_kernel(self):
        kwargs = {"use_rmpad": self.config.trainer_args.use_rmpad}
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance
            from liger_kernel.transformers.monkey_patch import (
                MODEL_TYPE_TO_APPLY_LIGER_FN,
            )
        except ImportError as e:
            Logging.error(
                "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 is not available. "
                "Please install it with `pip install liger-kernel`"
            )

        model_type = getattr(self.model, "config", None) and getattr(
            self.model.config, "model_type", None
        )
        if model_type in CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN:
            Logging.info(f"Try to apply liger kernel on the model {model_type}")
            _apply_liger_kernel_to_custom_instance(self.model, **kwargs)
        # If the model itself is already in liger kernel,
        # we should not apply the liger kernel again
        elif model_type in MODEL_TYPE_TO_APPLY_LIGER_FN:
            Logging.info(f"Try to apply liger kernel on the model {model_type}")
            _apply_liger_kernel_to_instance(self.model)
        else:
            # If not, we probe whether lm can apply
            Logging.info(
                f"Not found model class, Try to apply liger kernel on the language model of the model {model_type}"
            )
            try:
                model_type = getattr(
                    self.model.language_model, "config", None
                ) and getattr(self.model.language_model.config, "model_type", None)
                _apply_liger_kernel_to_instance(self.model.language_model)
                if model_type and model_type in MODEL_TYPE_TO_APPLY_LIGER_FN:
                    Logging.info(
                        f"Successfully apply liger kernels to model type {model_type}"
                    )
                else:
                    Logging.info(
                        f"Cannot find model type {model_type} in MODEL_TYPE_TO_APPLY_LIGER_FN, skip applying liger kernels"
                    )
            except Exception as e:
                Logging.error(
                    f"Try to apply liger kernel on the language model of the model, but failed with exceptions : \n {e}"
                )

    def _load_mm_projector(self):
        pretrain_mm_mlp_adapter = self.config.model_config.pretrain_mm_mlp_adapter
        mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

        def get_w(weights, keyword):
            return {
                k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k
            }

        deepspeed3_enabled = hasattr(
            [p for p in self.model.multi_modal_projector.parameters()][0], "ds_id"
        )

        TrainUtilities.load_zero_partitions(
            self.model.multi_modal_projector,
            get_w(mm_projector_weights, "multi_modal_projector"),
            deepspeed3_enabled,
            pretrain_mm_mlp_adapter,
        )
        TrainUtilities.load_zero_partitions(
            self.model.audio_modal_projector,
            get_w(mm_projector_weights, "audio_modal_projector"),
            deepspeed3_enabled,
            pretrain_mm_mlp_adapter,
        )

        Logging.info(
            f"Loaded multi_modal_projector,audio_modal_projector weights from {pretrain_mm_mlp_adapter}."
        )

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

    @abstractmethod
    def run(self, **kwargs):
        pass

    def save_config(self):
        output_dir = self.config.trainer_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/training_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)
        if self.config.dataset_config.dataset_format == "yaml":
            if self.config.dataset_config.dataset_path:
                # Copy the external yaml to output dir
                yaml_path = self.config.dataset_config.dataset_path
                shutil.copy(yaml_path, f"{output_dir}/dataset.yaml")
            elif self.config.dataset_config.datasets:
                # For inline datasets, save them to a yaml file
                with open(f"{output_dir}/dataset.yaml", "w") as f:
                    yaml.dump({"datasets": self.config.dataset_config.datasets}, f)

    def set_random_seed(self, random_seed: int = 42):
        # Setting random seed for all
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        Logging.info(f"Set random seed to {random_seed}")
        return random_seed
