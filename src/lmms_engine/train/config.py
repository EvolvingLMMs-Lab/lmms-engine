from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

import transformers

from ..datasets import DatasetConfig
from ..models import ModelConfig


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_muon: Optional[bool] = False
    freeze_modules: Optional[List[str]] = None
    only_save_mm_adapter: Optional[bool] = False
    use_rmpad: Optional[bool] = False
    fsdp2: Optional[bool] = False
    sp_ulysses_degree: Optional[int] = 1
    reduce_dtype: Optional[str] = "bfloat16"
    output_dtype: Optional[str] = "bfloat16"


@dataclass
class TrainerConfig:
    trainer_type: Literal["hf_trainer", "fsdp2_trainer"]
    dataset_config: DatasetConfig
    trainer_args: TrainingArguments
    model_config: ModelConfig
    extra_kwargs: Dict[str, Any] = None

    def to_dict(self):
        trainer_args_dict = self.trainer_args.to_dict()
        model_config_dict = self.model_config.to_dict()
        dataset_config_dict = self.dataset_config.to_dict()
        final_dict = asdict(self)
        final_dict["trainer_args"] = trainer_args_dict
        final_dict["model_config"] = model_config_dict
        final_dict["dataset_config"] = dataset_config_dict
        return final_dict
