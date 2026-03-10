from functools import partial, wraps

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.monkey_patch import (
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except Exception:
    print("liger kernel not installed, please install it with `pip install liger-kernel`")

from loguru import logger
from transformers import PreTrainedModel

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("qwen3_5_text", "liger")
def apply_liger_kernel_to_qwen3_5(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
) -> None:
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen3_5 import modeling_qwen3_5

    if rope:
        modeling_qwen3_5.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_qwen3_5.Qwen3_5RMSNorm = LigerRMSNorm

    if fused_linear_cross_entropy:
        from .qwen3_5_liger import qwen3_5_lce_forward

        if use_rmpad:

            def wrap_forward(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(use_rmpad=use_rmpad, *args, **kwargs)

                return wrapper

            qwen3_5_lce_forward = wrap_forward(qwen3_5_lce_forward)
        modeling_qwen3_5.Qwen3_5ForCausalLM.forward = qwen3_5_lce_forward

    if swiglu:
        modeling_qwen3_5.Qwen3_5MLP = LigerSwiGLUMLP

    if use_rmpad:
        from .qwen3_5_ops import attn_forward as qwen3_5_ops_attn_forward
        from .qwen3_5_ops import (
            decoder_layer_forward as qwen3_5_ops_decoder_layer_forward,
        )
        from .qwen3_5_ops import model_forward as qwen3_5_ops_model_forward

        modeling_qwen3_5.Qwen3_5TextModel.forward = qwen3_5_ops_model_forward
        modeling_qwen3_5.Qwen3_5DecoderLayer.forward = qwen3_5_ops_decoder_layer_forward
        modeling_qwen3_5.Qwen3_5Attention.forward = qwen3_5_ops_attn_forward

    if model is not None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForCausalLM,
            Qwen3_5TextModel,
        )

        if isinstance(model, Qwen3_5ForCausalLM):
            base_model: Qwen3_5TextModel = model.model
        elif isinstance(model, Qwen3_5TextModel):
            base_model: Qwen3_5TextModel = model
        elif hasattr(model, "language_model"):
            base_model = getattr(
                model.language_model,
                model.language_model.base_model_prefix,
                model.language_model,
            )
        else:
            base_model = getattr(model, "model", model)

        _patch_qwen3_5_rms_norm = partial(_patch_rms_norm_module, offset=1.0, casting_mode="llama")

        if rms_norm:
            _patch_qwen3_5_rms_norm(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_qwen3_5_rms_norm(decoder_layer.input_layernorm)
                _patch_qwen3_5_rms_norm(decoder_layer.post_attention_layernorm)
                self_attn = getattr(decoder_layer, "self_attn", None)
                if self_attn is not None:
                    if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                        _patch_qwen3_5_rms_norm(self_attn.q_norm)
                    if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                        _patch_qwen3_5_rms_norm(self_attn.k_norm)
