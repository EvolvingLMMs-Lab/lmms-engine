# src/lmms_engine/models/aero_realtime_omni/configuration_aero_realtime_omni.py
"""AeroRealtime Omni wrapper config (thinker + talker)."""

from transformers.configuration_utils import PretrainedConfig

# thinker config lives in the SIBLING package ``aero_realtime``; do NOT use a
# relative import (would resolve inside ``aero_realtime_omni`` and fail).
from lmms_engine.models.aero_realtime.configuration_aero_realtime import (
    AeroRealtimeConfig,
)

from .configuration_aero_realtime_talker import AeroRealtimeTalkerConfig


class AeroRealtimeOmniConfig(PretrainedConfig):
    model_type = "aero_realtime_omni"

    sub_configs = {
        "thinker_config": AeroRealtimeConfig,
        "talker_config": AeroRealtimeTalkerConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        talker_config=None,
        codec_loss_weight: float = 1.0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if thinker_config is None:
            thinker_config = AeroRealtimeConfig()
        elif isinstance(thinker_config, dict):
            thinker_config = AeroRealtimeConfig(**thinker_config)
        self.thinker_config = thinker_config

        if talker_config is None:
            talker_config = AeroRealtimeTalkerConfig()
        elif isinstance(talker_config, dict):
            talker_config = AeroRealtimeTalkerConfig(**talker_config)
        self.talker_config = talker_config

        self.codec_loss_weight = codec_loss_weight
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
