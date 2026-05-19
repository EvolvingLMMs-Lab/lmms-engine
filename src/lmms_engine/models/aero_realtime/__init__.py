from lmms_engine.mapping_func import register_model

from .configuration_aero_realtime import AeroRealtimeConfig
from .modeling_aero_realtime import AeroRealtimeForConditionalGeneration
from .monkey_patch import apply_liger_kernel_to_aero_realtime
from .processing_aero_realtime import AeroRealtimeProcessor

register_model(
    "aero_realtime",
    AeroRealtimeConfig,
    AeroRealtimeForConditionalGeneration,
)

__all__ = [
    "AeroRealtimeConfig",
    "AeroRealtimeForConditionalGeneration",
    "AeroRealtimeProcessor",
    "apply_liger_kernel_to_aero_realtime",
]
