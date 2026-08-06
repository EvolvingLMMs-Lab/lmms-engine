from loguru import logger
from transformers import PreTrainedModel

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("voxtral_realtime_encoder", "rmpad")
def apply_rmpad_to_voxtral_realtime(model: PreTrainedModel = None) -> None:
    from transformers.models.voxtral_realtime.modeling_voxtral_realtime import (
        VoxtralRealtimeAttention,
        VoxtralRealtimeEncoder,
    )

    from .voxtral_realtime_ops import (
        voxtral_realtime_attention_forward,
        voxtral_realtime_encoder_forward,
    )

    if model is not None and not isinstance(model, VoxtralRealtimeEncoder):
        raise TypeError(f"Expected VoxtralRealtimeEncoder, got {type(model)}")

    VoxtralRealtimeEncoder.forward = voxtral_realtime_encoder_forward
    VoxtralRealtimeAttention.forward = voxtral_realtime_attention_forward
    logger.info("voxtral_realtime rmpad: patched encoder and attention forwards")
