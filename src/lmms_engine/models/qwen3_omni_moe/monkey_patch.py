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
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
    Qwen3OmniMoeVisionEncoder,
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


@MONKEY_PATCHER.register("qwen3_omni_moe", "liger")
@MONKEY_PATCHER.register("qwen3_omni_moe_thinker", "liger")
def apply_liger_kernel_to_qwen3_omni_moe(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    layer_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = True,
) -> None:
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe

    from .qwen3_omni_moe_liger import lce_forward as qwen3_omni_moe_lce_forward
    from .qwen3_omni_moe_ops import (
        moe_sparse_layer_forward as qwen3_omni_moe_moe_sparse_layer_forward,
    )

    def wrap_forward(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.setdefault("use_rmpad", use_rmpad)
            return func(*args, **kwargs)

        return wrapper

    qwen3_omni_moe_lce_forward = wrap_forward(qwen3_omni_moe_lce_forward)
    if rope:
        Logging.warning("RoPE optimization not supported for Qwen3-Omni MoE, skipping")
    if rms_norm:
        modeling_qwen3_omni_moe.Qwen3OmniMoeRMSNorm = LigerRMSNorm

    if cross_entropy:
        modeling_qwen3_omni_moe.CrossEntropyLoss = LigerCrossEntropyLoss
        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextRMSNorm = LigerRMSNorm
    if fused_linear_cross_entropy:
        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration.forward = qwen3_omni_moe_lce_forward
    if swiglu:
        modeling_qwen3_omni_moe.Qwen3OmniMoeMLP = LigerSwiGLUMLP
    if use_rmpad:
        from .qwen3_omni_moe_ops import attn_forward as qwen3_omni_moe_attn_forward
        from .qwen3_omni_moe_ops import (
            decoder_layer_forward as qwen3_omni_moe_decoder_layer_forward,
        )
        from .qwen3_omni_moe_ops import (
            text_model_forward as qwen3_omni_moe_text_model_forward,
        )

        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextModel.forward = qwen3_omni_moe_text_model_forward
        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextDecoderLayer.forward = qwen3_omni_moe_decoder_layer_forward
        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextAttention.forward = qwen3_omni_moe_attn_forward

    if get_ulysses_sequence_parallel_world_size() > 1:
        patch_vlm_for_ulysses_input_slicing(modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextModel)

    if model is not None:
        if isinstance(model, Qwen3OmniMoeThinkerForConditionalGeneration):
            text_model: Qwen3OmniMoeThinkerTextModel = model.model
            vision_model: Qwen3OmniMoeVisionEncoder = model.visual
            audio_model: Qwen3OmniMoeAudioEncoder = model.audio_tower
        elif isinstance(model, Qwen3OmniMoeThinkerTextModel):
            text_model: Qwen3OmniMoeThinkerTextModel = model
            vision_model = None
            audio_model = None
        else:
            raise TypeError(
                f"Unsupported Qwen3-Omni MoE model type. `model` must be "
                f"`Qwen3OmniMoeThinkerForConditionalGeneration` or `Qwen3OmniMoeThinkerTextModel`. "
                f"Got: {type(model)}. "
                f"If you have the full model, extract the thinker using scripts/extract_qwen_omni_thinker.py"
            )

        if vision_model is not None and rms_norm:
            for vision_block in vision_model.blocks:
                _patch_layer_norm_module(vision_block.norm1)
                _patch_layer_norm_module(vision_block.norm2)
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
                    if hasattr(decoder_layer.mlp, "experts"):
                        for expert in decoder_layer.mlp.experts:
                            _patch_swiglu_module(expert, LigerSwiGLUMLP)
                    else:
                        _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
                if rms_norm:
                    _patch_rms_norm_module(decoder_layer.input_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_attention_layernorm)

    modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock.forward = qwen3_omni_moe_moe_sparse_layer_forward
