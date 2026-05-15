"""Monkey patches for transformers.models.qwen3_5_moe.

Two independent registrations:
- `liger`: rope/rmsnorm/swiglu + fused-LCE forward on the CausalLM class.
- `rmpad`: text model/decoder/attention/gated-delta-net/MoE/experts forwards,
   plus rmpad-flavoured CausalLM forward.

The trainer runner applies them in order ["liger", "rmpad"] when both are
requested. SP is intentionally not supported.
"""

from functools import partial, wraps
from types import MethodType

from loguru import logger
from transformers import PreTrainedModel

try:
    from liger_kernel.transformers.monkey_patch import (
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except ImportError:
    _patch_rms_norm_module = None
    _patch_swiglu_module = None
    LigerRMSNorm = None
    liger_rotary_pos_emb = None
    LigerSwiGLUMLP = None
    logger.warning("liger kernel not installed; qwen3_5_moe liger patch will be a no-op.")

from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.utils.import_utils import is_transformers_version_greater_or_equal_to

_IS_TRANSFORMERS_5 = is_transformers_version_greater_or_equal_to("5.0")


@MONKEY_PATCHER.register("qwen3_5_moe", "liger")
def apply_liger_kernel_to_qwen3_5_moe(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

    from .qwen3_5_moe_liger import lce_forward

    if rope and liger_rotary_pos_emb is not None:
        modeling_qwen3_5_moe.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm and LigerRMSNorm is not None:
        modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm = LigerRMSNorm

    if cross_entropy:
        from liger_kernel.transformers.functional import liger_cross_entropy
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(lce_forward, model)
        else:
            modeling_qwen3_5_moe.Qwen3_5MoeForCausalLM.forward = lce_forward

    if swiglu and LigerSwiGLUMLP is not None:
        # qwen3_5_moe MLP (the inner Qwen3_5MoeMLP used as shared_expert) is swiglu-style
        modeling_qwen3_5_moe.Qwen3_5MoeMLP = LigerSwiGLUMLP

    if model is not None:
        base_model = getattr(model, model.base_model_prefix, model)
        # base_model is Qwen3_5MoeTextModel for ForCausalLM, Qwen3_5MoeModel for ConditionalGeneration
        language_model = getattr(base_model, "language_model", base_model)
        if rms_norm and _patch_rms_norm_module is not None:
            if hasattr(language_model, "norm"):
                _patch_rms_norm_module(language_model.norm)
            for decoder_layer in getattr(language_model, "layers", []):
                if hasattr(decoder_layer, "input_layernorm"):
                    _patch_rms_norm_module(decoder_layer.input_layernorm)
                if hasattr(decoder_layer, "post_attention_layernorm"):
                    _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


@MONKEY_PATCHER.register("qwen3_5_moe", "rmpad")
def apply_rmpad_to_qwen3_5_moe(model: PreTrainedModel = None) -> None:
    """Replace the qwen3_5_moe text model / decoder / attention / linear-attn /
    MoE / experts forwards with rmpad-aware versions. If `model` is None we
    patch class-level only (future construction); otherwise we also rebind
    `model.forward` to the rmpad-flavoured lce_forward.
    """
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

    from .qwen3_5_moe_liger import lce_forward
    from .qwen3_5_moe_ops import (
        attn_forward,
        decoder_layer_forward,
        experts_forward,
        gated_delta_net_forward,
        model_forward,
        moe_sparse_layer_forward,
        text_model_forward,
    )

    modeling_qwen3_5_moe.Qwen3_5MoeTextModel.forward = text_model_forward
    modeling_qwen3_5_moe.Qwen3_5MoeModel.forward = model_forward
    modeling_qwen3_5_moe.Qwen3_5MoeDecoderLayer.forward = decoder_layer_forward
    modeling_qwen3_5_moe.Qwen3_5MoeAttention.forward = attn_forward
    modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet.forward = gated_delta_net_forward
    modeling_qwen3_5_moe.Qwen3_5MoeSparseMoeBlock.forward = moe_sparse_layer_forward
    if _IS_TRANSFORMERS_5:
        modeling_qwen3_5_moe.Qwen3_5MoeExperts.forward = experts_forward

    if model is not None:
        # rebind CausalLM forward to lce_forward with use_rmpad=True
        bound = partial(lce_forward, use_rmpad=True)

        @wraps(lce_forward)
        def _forward(self, *args, **kwargs):
            return bound(self, *args, **kwargs)

        model.forward = MethodType(_forward, model)
