from lmms_engine.mapping_func import register_model

from .configuration_llavaonevision1_5 import Llavaonevision1_5Config
from .modeling_llavaonevision1_5 import LLaVAOneVision1_5_ForConditionalGeneration
from .monkey_patch import apply_liger_kernel_to_llava_onevision1_5

register_model(
    "llavaonevision1_5",
    Llavaonevision1_5Config,
    LLaVAOneVision1_5_ForConditionalGeneration,
    model_general_type="image_text_to_text",
)

__all__ = [
    "Llavaonevision1_5Config",
    "LLaVAOneVision1_5_ForConditionalGeneration",
    "apply_liger_kernel_to_llava_onevision1_5",
]
