from .bagel_grpo_trainer import BagelGRPOTrainer  # noqa: F401 - ensures registration
from .fsdp2_trainer import FSDP2SFTTrainer
from .rae_trainer import RaeTrainer
from .sit_trainer import SitTrainer

__all__ = [
    "FSDP2SFTTrainer",
    "SitTrainer",
    "RaeTrainer",
    "BagelGRPOTrainer",
]
