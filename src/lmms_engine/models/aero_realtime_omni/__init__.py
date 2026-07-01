"""AeroRealtime Omni (thinker + talker) wrapper package.

Imports the thinker classes from the existing ``aero_realtime`` package
unchanged; adds the talker + omni wrapper classes and registers the new
``aero_realtime_omni`` model type. Also registers monkey-patchers (Task 12).
"""

from transformers import AutoConfig

from lmms_engine.mapping_func import register_model
from lmms_engine.models.aero_realtime import (
    AeroRealtimeConfig,
    AeroRealtimeForConditionalGeneration,
)

from .configuration_aero_realtime_omni import AeroRealtimeOmniConfig
from .configuration_aero_realtime_talker import (
    AeroRealtimeTalkerCodePredictorConfig,
    AeroRealtimeTalkerConfig,
)
from .modeling_aero_realtime_omni import (
    AeroRealtimeOmniCausalLMOutputWithPast,
    AeroRealtimeOmniForConditionalGeneration,
)
from .modeling_aero_realtime_talker import AeroRealtimeTalkerForConditionalGeneration

AutoConfig.register("aero_realtime_talker", AeroRealtimeTalkerConfig, exist_ok=True)
AutoConfig.register(
    "aero_realtime_talker_code_predictor",
    AeroRealtimeTalkerCodePredictorConfig,
    exist_ok=True,
)

register_model(
    "aero_realtime_omni",
    AeroRealtimeOmniConfig,
    AeroRealtimeOmniForConditionalGeneration,
    model_general_type="general",
)

from . import monkey_patch  # noqa: F401, E402

__all__ = [
    "AeroRealtimeOmniConfig",
    "AeroRealtimeOmniForConditionalGeneration",
    "AeroRealtimeOmniCausalLMOutputWithPast",
    "AeroRealtimeTalkerConfig",
    "AeroRealtimeTalkerCodePredictorConfig",
    "AeroRealtimeTalkerForConditionalGeneration",
]
