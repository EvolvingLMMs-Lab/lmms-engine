from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

import transformers

from ..datasets import DatasetConfig
from ..models import ModelConfig


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_muon: Optional[bool] = False
    freeze_modules: Optional[List[str]] = None
    use_rmpad: Optional[bool] = False
    fsdp2: Optional[bool] = False
    reduce_dtype: Optional[str] = "bfloat16"
    output_dtype: Optional[str] = "bfloat16"
    print_batch_input_steps: Optional[int] = -1
    enable_profiler: Optional[bool] = False
    profiler_config: Optional[Dict[str, Any]] = None

    # Parallelism
    ep_degree: Optional[int] = 1
    sp_ulysses_degree: Optional[int] = 1

    # --- EMA (Exponential Moving Average) ---
    ema_enabled: Optional[bool] = False
    ema_decay: Optional[float] = 0.9999
    ema_update_every: Optional[int] = 1
    ema_start_step: Optional[int] = 0
    ema_requires_grad_only: Optional[bool] = True
    # Optional name-based filtering for which parameters participate in EMA.
    # Example:
    #   ema_param_filter:
    #     mode: "substring"  # or "regex"
    #     include: ["language_model"]   # only include matching params
    #     exclude: ["lm_head"]          # exclude matching params
    ema_param_filter: Optional[Dict[str, Any]] = None
    ema_resume_from_ema: Optional[bool] = False

    # --- Eval Server Configuration ---
    eval_config: Optional[Dict[str, Any]] = None


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
