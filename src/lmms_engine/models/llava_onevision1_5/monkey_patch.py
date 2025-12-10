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
from loguru import logger
from transformers import PreTrainedModel

from lmms_engine.models.aero.monkey_patch import apply_liger_kernel_to_aero
from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.models.qwen3.monkey_patch import apply_liger_kernel_to_qwen3

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"


@MONKEY_PATCHER.register("llavatext", "liger")
def apply_liger_kernel_to_llavatext(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
) -> None:
    """
    Apply Liger kernels to LLaVAOneVision1_5_TextModel.
    This function patches the text model components with optimized Liger implementations.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from . import modeling_llavaonevision1_5
    from .modeling_llavaonevision1_5 import LLaVAOneVision1_5_TextModel

    if rope:
        modeling_llavaonevision1_5.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_llavaonevision1_5.LLaVAOneVision1_5_RMSNorm = LigerRMSNorm

    if cross_entropy:
        if transformer_version >= version.parse(SUPPORTED_TRANSFORMER_VERSION):
            from transformers.loss.loss_utils import nn

            nn.functional.cross_entropy = liger_cross_entropy
        else:
            logger.warning(TRANSFORMER_DEPRECATION_WARNING)
            modeling_llavaonevision1_5.CrossEntropyLoss = LigerCrossEntropyLoss

    if swiglu:
        modeling_llavaonevision1_5.LLaVAOneVision1_5_MLP = LigerSwiGLUMLP

    if use_rmpad:
        from .llava_onevision1_5_ops import attn_forward as llavatext_ops_attn_forward
        from .llava_onevision1_5_ops import (
            decoder_layer_forward as llavatext_ops_decoder_layer_forward,
        )
        from .llava_onevision1_5_ops import model_forward as llavatext_ops_model_forward

        modeling_llavaonevision1_5.LLaVAOneVision1_5_TextModel.forward = llavatext_ops_model_forward
        modeling_llavaonevision1_5.LLaVAOneVision1_5_DecoderLayer.forward = llavatext_ops_decoder_layer_forward
        modeling_llavaonevision1_5.LLaVAOneVision1_5_FlashAttention2.forward = llavatext_ops_attn_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the text model from the model instance
        if hasattr(model, "language_model"):
            base_model: LLaVAOneVision1_5_TextModel = getattr(
                model.language_model,
                model.language_model.base_model_prefix,
                model.language_model,
            )
        elif isinstance(model, LLaVAOneVision1_5_TextModel):
            base_model: LLaVAOneVision1_5_TextModel = model
        else:
            base_model: LLaVAOneVision1_5_TextModel = model.model

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


@MONKEY_PATCHER.register("llavaonevision1_5", "liger")
def apply_liger_kernel_to_llava_onevision1_5(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = True,
) -> None:
    from .modeling_llavaonevision1_5 import LLaVAOneVision1_5_ForConditionalGeneration

    if fused_linear_cross_entropy:
        from .llava_onevision1_5_liger import forward as llava_onevision1_5_forward

        if use_rmpad:

            def wrap_forward(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(use_rmpad=use_rmpad, *args, **kwargs)

                return wrapper

            llava_onevision1_5_forward = wrap_forward(llava_onevision1_5_forward)

        LLaVAOneVision1_5_ForConditionalGeneration.forward = llava_onevision1_5_forward

    language_model = getattr(model, "language_model", None) if model is not None else None

    # Apply liger kernel to the text model (language_model)
    apply_liger_kernel_to_qwen3(
        rope=rope,
        cross_entropy=cross_entropy,
        fused_linear_cross_entropy=False,  # Already handled at the top level
        rms_norm=rms_norm,
        swiglu=swiglu,
        model=language_model,
        use_rmpad=use_rmpad,
    )
