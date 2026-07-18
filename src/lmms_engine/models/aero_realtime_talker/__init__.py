"""Standalone AeroRealtime talker trained from aligned text-stream IDs."""

from transformers import AutoConfig

from lmms_engine.mapping_func import register_model

from .configuration_aero_realtime_talker import (
    AeroRealtimeTalkerCodePredictorConfig,
    AeroRealtimeTalkerConfig,
)
from .modeling_aero_realtime_talker import (
    AeroRealtimeTalkerForConditionalGeneration,
    AeroRealtimeTalkerOutputWithPast,
)

AutoConfig.register(
    "aero_realtime_talker_code_predictor",
    AeroRealtimeTalkerCodePredictorConfig,
    exist_ok=True,
)
register_model(
    "aero_realtime_talker",
    AeroRealtimeTalkerConfig,
    AeroRealtimeTalkerForConditionalGeneration,
    model_general_type="general",
)

from . import monkey_patch  # noqa: E402, F401

__all__ = [
    "AeroRealtimeTalkerCodePredictorConfig",
    "AeroRealtimeTalkerConfig",
    "AeroRealtimeTalkerForConditionalGeneration",
    "AeroRealtimeTalkerOutputWithPast",
]
