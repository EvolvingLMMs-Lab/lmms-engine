from functools import wraps

from transformers import PreTrainedModel

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("qwen2_audio", "liger")
def apply_liger_kernel_to_qwen2_audio(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = True,
):
    from transformers import Qwen2AudioEncoder
    from transformers.models.qwen2_audio.modeling_qwen2_audio import (
        Qwen2AudioAttention,
        Qwen2AudioEncoderLayer,
    )

    if use_rmpad:
        from .qwen2_audio_ops import encoder_forward as qwen2_audio_encoder_forward
        from .qwen2_audio_ops import (
            encoder_layer_forward as qwen2_audio_encoder_layer_forward,
        )
        from .qwen2_audio_ops import (
            flash_attn_forward as qwen2_audio_flash_attn_forward,
        )

        Qwen2AudioEncoder.forward = qwen2_audio_encoder_forward
        Qwen2AudioEncoderLayer.forward = qwen2_audio_encoder_layer_forward
        Qwen2AudioAttention.forward = qwen2_audio_flash_attn_forward
