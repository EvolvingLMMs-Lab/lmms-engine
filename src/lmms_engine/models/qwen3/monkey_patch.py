from functools import wraps

from packaging import version

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.functional import liger_cross_entropy
    from liger_kernel.transformers.monkey_patch import (
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

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"

from loguru import logger

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("qwen3", "liger")
def apply_liger_kernel_to_qwen3(
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

    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    if rope:
        modeling_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_qwen3.Qwen3RMSNorm = LigerRMSNorm

    if cross_entropy:
        if transformer_version >= version.parse(SUPPORTED_TRANSFORMER_VERSION):
            from transformers.loss.loss_utils import nn

            nn.functional.cross_entropy = liger_cross_entropy
        else:
            logger.warning(TRANSFORMER_DEPRECATION_WARNING)
            modeling_qwen3.CrossEntropyLoss = LigerCrossEntropyLoss

    if fused_linear_cross_entropy:
        from lmms_engine.models.qwen3.qwen3_liger import qwen3_lce_forward

        if use_rmpad:

            def wrap_forward(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(use_rmpad=use_rmpad, *args, **kwargs)

                return wrapper

            qwen3_lce_forward = wrap_forward(qwen3_lce_forward)
        modeling_qwen3.Qwen3ForCausalLM.forward = qwen3_lce_forward

    if swiglu:
        modeling_qwen3.Qwen3MLP = LigerSwiGLUMLP

    if use_rmpad:
        from lmms_engine.models.qwen3.qwen3_ops import (
            attn_forward as qwen3_ops_attn_forward,
        )
        from lmms_engine.models.qwen3.qwen3_ops import (
            decoder_layer_forward as qwen3_ops_decoder_layer_forward,
        )
        from lmms_engine.models.qwen3.qwen3_ops import (
            model_forward as qwen3_ops_model_forward,
        )

        modeling_qwen3.Qwen3Model.forward = qwen3_ops_model_forward
        modeling_qwen3.Qwen3DecoderLayer.forward = qwen3_ops_decoder_layer_forward
        modeling_qwen3.Qwen3Attention.forward = qwen3_ops_attn_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        if hasattr(model, "language_model"):
            base_model: Qwen3Model = getattr(
                model.language_model,
                model.language_model.base_model_prefix,
                model.language_model,
            )
        elif isinstance(model, Qwen3Model):
            base_model: Qwen3Model = model
        else:
            base_model: Qwen3Model = model.model

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
