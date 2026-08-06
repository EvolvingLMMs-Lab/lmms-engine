from transformers import AutoConfig, AutoModel

from lmms_engine.mapping_func import register_model

from .configuration_aero_realtime import (
    AeroRealtimeAudioEncoderConfig,
    AeroRealtimeConfig,
)
from .modeling_aero_realtime import (
    AeroRealtimeAudioEncoder,
    AeroRealtimeForConditionalGeneration,
)
from .monkey_patch import apply_liger_kernel_to_aero_realtime
from .processing_aero_realtime import AeroRealtimeProcessor

register_model(
    "aero_realtime",
    AeroRealtimeConfig,
    AeroRealtimeForConditionalGeneration,
)

AutoConfig.register("aero_realtime_audio_encoder", AeroRealtimeAudioEncoderConfig, exist_ok=True)
AutoModel.register(AeroRealtimeAudioEncoderConfig, AeroRealtimeAudioEncoder, exist_ok=True)

__all__ = [
    "AeroRealtimeConfig",
    "AeroRealtimeAudioEncoderConfig",
    "AeroRealtimeForConditionalGeneration",
    "AeroRealtimeAudioEncoder",
    "AeroRealtimeProcessor",
    "apply_liger_kernel_to_aero_realtime",
]
