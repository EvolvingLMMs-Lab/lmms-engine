from lmms_engine.mapping_func import register_model

from .configuration_aero import AeroConfig
from .modeling_aero import AeroForConditionalGeneration
from .monkey_patch import apply_liger_kernel_to_aero
from .processing_aero import AeroProcessor

register_model(
    "aero",
    AeroConfig,
    AeroForConditionalGeneration,
)

__all__ = [
    "AeroConfig",
    "AeroForConditionalGeneration",
    "AeroProcessor",
    "apply_liger_kernel_to_aero",
]
