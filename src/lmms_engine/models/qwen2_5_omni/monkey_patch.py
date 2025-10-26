import inspect
from functools import partial, wraps
from typing import Callable

from packaging import version

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.functional import liger_cross_entropy
    from liger_kernel.transformers.geglu import LigerGEGLUMLP
    from liger_kernel.transformers.layer_norm import LigerLayerNorm
    from liger_kernel.transformers.model.qwen2 import (
        lce_forward_deprecated as qwen2_lce_forward_deprecated,
    )
    from liger_kernel.transformers.monkey_patch import (
        _patch_layer_norm_module,
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except:
    print("liger kernel not installed, please install it with `pip install liger-kernel`")

import transformers
from transformers import PreTrainedModel
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
    Qwen2_5OmniVisionEncoder,
)

from lmms_engine.parallel.sequence_parallel.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    patch_vlm_for_ulysses_input_slicing,
)

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"

from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.utils.logging_utils import Logging


@MONKEY_PATCHER.register("qwen2_5_omni", "liger")
@MONKEY_PATCHER.register("qwen2_5_omni_thinker", "liger")
def apply_liger_kernel_to_qwen2_5_omni(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    layer_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-Omni models.
    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        use_rmpad (bool): Whether to use remove padding optimization. Default is False.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni

    from .qwen2_5_omni_liger import lce_forward as qwen2_5_omni_lce_forward

    def wrap_forward(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.setdefault("use_rmpad", use_rmpad)
            return func(*args, **kwargs)

        return wrapper

    qwen2_5_omni_lce_forward = wrap_forward(qwen2_5_omni_lce_forward)
    if rope:
        Logging.warning("RoPE optimization not supported for Qwen2.5-Omni, skipping")
    if rms_norm:
        modeling_qwen2_5_omni.Qwen2RMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_qwen2_5_omni.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_qwen2_5_omni.Qwen2_5OmniThinkerForConditionalGeneration.forward = qwen2_5_omni_lce_forward
    if swiglu:
        modeling_qwen2_5_omni.Qwen2MLP = LigerSwiGLUMLP
    if use_rmpad:
        from .qwen2_5_omni_ops import attn_forward as qwen2_5_omni_attn_forward
        from .qwen2_5_omni_ops import (
            decoder_layer_forward as qwen2_5_omni_decoder_layer_forward,
        )
        from .qwen2_5_omni_ops import (
            text_model_forward as qwen2_5_omni_text_model_forward,
        )

        modeling_qwen2_5_omni.Qwen2_5OmniThinkerTextModel.forward = qwen2_5_omni_text_model_forward
        modeling_qwen2_5_omni.Qwen2_5OmniDecoderLayer.forward = qwen2_5_omni_decoder_layer_forward
        modeling_qwen2_5_omni.Qwen2_5OmniAttention.forward = qwen2_5_omni_attn_forward

    if get_ulysses_sequence_parallel_world_size() > 1:
        patch_vlm_for_ulysses_input_slicing(modeling_qwen2_5_omni.Qwen2_5OmniThinkerTextModel)

    if model is not None:
        if isinstance(model, Qwen2_5OmniThinkerForConditionalGeneration):
            text_model: Qwen2_5OmniThinkerTextModel = model.model
            vision_model: Qwen2_5OmniVisionEncoder = model.visual
            audio_model: Qwen2_5OmniAudioEncoder = model.audio_tower
        elif isinstance(model, Qwen2_5OmniThinkerTextModel):
            text_model: Qwen2_5OmniThinkerTextModel = model
            vision_model = None
            audio_model = None
        else:
            raise TypeError(
                f"Unsupported Qwen2.5-Omni model type. `model` must be "
                f"`Qwen2_5OmniThinkerForConditionalGeneration` or `Qwen2_5OmniThinkerTextModel`. "
                f"Got: {type(model)}. "
                f"If you have the full model, extract the thinker using scripts/extract_qwen_omni_thinker.py"
            )

        if vision_model is not None and rms_norm:
            for vision_block in vision_model.blocks:
                _patch_rms_norm_module(vision_block.norm1)
                _patch_rms_norm_module(vision_block.norm2)
        if audio_model is not None and layer_norm:
            if hasattr(audio_model, "layers"):
                for audio_layer in audio_model.layers:
                    _patch_layer_norm_module(audio_layer.self_attn_layer_norm)
                    _patch_layer_norm_module(audio_layer.final_layer_norm)
        if text_model is not None:
            if rms_norm:
                _patch_rms_norm_module(text_model.norm)
            for decoder_layer in text_model.layers:
                if swiglu:
                    _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
                if rms_norm:
                    _patch_rms_norm_module(decoder_layer.input_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
