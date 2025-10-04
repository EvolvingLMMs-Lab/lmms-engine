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
    print(
        "liger kernel not installed, please install it with `pip install liger-kernel`"
    )

import transformers
from transformers import PreTrainedModel

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"

from loguru import logger

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("qwen2", "liger")
def apply_liger_kernel_to_qwen2(
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

    from transformers.models.qwen2 import modeling_qwen2
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    if rope:
        modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm

    if cross_entropy:
        if transformer_version >= version.parse(SUPPORTED_TRANSFORMER_VERSION):
            from transformers.loss.loss_utils import nn

            nn.functional.cross_entropy = liger_cross_entropy
        else:
            logger.warning(TRANSFORMER_DEPRECATION_WARNING)
            modeling_qwen2.CrossEntropyLoss = LigerCrossEntropyLoss

    if fused_linear_cross_entropy:
        from lmms_engine.models.qwen2.qwen2_liger import qwen2_lce_forward

        if use_rmpad:

            def wrap_forward(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(use_rmpad=use_rmpad, *args, **kwargs)

                return wrapper

            qwen2_lce_forward = wrap_forward(qwen2_lce_forward)
        modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_lce_forward

    if swiglu:
        modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP

    if use_rmpad:
        from lmms_engine.models.qwen2.qwen2_ops import (
            attn_forward as qwen2_ops_attn_forward,
        )
        from lmms_engine.models.qwen2.qwen2_ops import (
            decoder_layer_forward as qwen2_ops_decoder_layer_forward,
        )
        from lmms_engine.models.qwen2.qwen2_ops import (
            model_forward as qwen2_ops_model_forward,
        )

        modeling_qwen2.Qwen2Model.forward = qwen2_ops_model_forward
        modeling_qwen2.Qwen2DecoderLayer.forward = qwen2_ops_decoder_layer_forward
        modeling_qwen2.Qwen2Attention.forward = qwen2_ops_attn_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        if hasattr(model, "language_model"):
            base_model: Qwen2Model = getattr(
                model.language_model,
                model.language_model.base_model_prefix,
                model.language_model,
            )
        elif isinstance(model, Qwen2Model):
            base_model: Qwen2Model = model
        else:
            base_model: Qwen2Model = model.model

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
