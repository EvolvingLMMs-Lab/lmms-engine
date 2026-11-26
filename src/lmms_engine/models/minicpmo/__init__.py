from lmms_engine.mapping_func import register_model

from .configuration_minicpmo import MiniCPMOConfig
from .modeling_minicpmo import MiniCPMO
from .modeling_minicpmo import MiniCPMOForTraining
from .processing_minicpmo import MiniCPMOProcessor

register_model(
    model_type="minicpmo",
    model_config=MiniCPMOConfig,
    model_class=MiniCPMOForTraining,
    model_general_type="causal_lm",
)

__all__ = [
    "MiniCPMOConfig",
    "MiniCPMO",
    "MiniCPMOForTraining",
    "MiniCPMOProcessor",
]
