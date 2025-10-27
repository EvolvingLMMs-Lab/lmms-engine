from functools import wraps
from types import MethodType

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
except ImportError:
    print("liger kernel not installed, please install it with `pip install liger-kernel`")

import transformers
from transformers import PreTrainedModel

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"

from loguru import logger

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("qwen3_moe", "liger")
def apply_liger_kernel_to_qwen3_moe(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3 models.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP
    from transformers.models.qwen3_moe import modeling_qwen3_moe
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

    from .qwen3_moe_liger import lce_forward as qwen3_lce_forward

    if rope:
        modeling_qwen3_moe.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_qwen3_moe.Qwen3MoeRMSNorm = LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if use_rmpad:

            def wrap_forward(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(use_rmpad=use_rmpad, *args, **kwargs)

                return wrapper

            qwen3_lce_forward = wrap_forward(qwen3_lce_forward)

        if model is not None:
            model.forward = MethodType(qwen3_lce_forward, model)
        else:
            modeling_qwen3_moe.Qwen3MoeForCausalLM.forward = qwen3_lce_forward

    if swiglu:
        modeling_qwen3_moe.Qwen3MoeMLP = LigerQwen3MoeSwiGLUMLP

    if use_rmpad:
        from .qwen3_moe_ops import attn_forward as qwen3_moe_ops_attn_forward
        from .qwen3_moe_ops import (
            decoder_layer_forward as qwen3_moe_ops_decoder_layer_forward,
        )
        from .qwen3_moe_ops import model_forward as qwen3_moe_ops_model_forward

        modeling_qwen3_moe.Qwen3MoeModel.forward = qwen3_moe_ops_model_forward
        modeling_qwen3_moe.Qwen3MoeDecoderLayer.forward = qwen3_moe_ops_decoder_layer_forward
        modeling_qwen3_moe.Qwen3MoeAttention.forward = qwen3_moe_ops_attn_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Qwen3MoeModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            if swiglu:
                for mlp_expert in decoder_layer.mlp.experts:
                    _patch_swiglu_module(mlp_expert, LigerQwen3MoeSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)

    # Apply patch for sparse layer
    from .qwen3_moe_ops import (
        moe_sparse_layer_forward as qwen3_moe_ops_moe_sparse_layer_forward,
    )

    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward = qwen3_moe_ops_moe_sparse_layer_forward
