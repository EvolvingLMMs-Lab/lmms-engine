import inspect
from functools import partial, wraps
from typing import Callable

from packaging import version

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.monkey_patch import (
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except:
    print(
        "liger kernel not installed, please install it with `pip install liger-kernel`"
    )

import transformers
from transformers import (
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLTextModel,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)

from lmms_engine.parallel.sequence_parallel.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    patch_vlm_for_ulysses_input_slicing,
)

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("qwen2_5_vl", "liger")
def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2-VL models.
    NOTE: Qwen2.5-VL is not available in transformers<4.48.2
    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    from .qwen2_5_vl_liger import lce_forward as qwen2_5_vl_lce_forward

    if use_rmpad:

        def wrap_forward(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(use_rmpad=use_rmpad, *args, **kwargs)

            return wrapper

        qwen2_5_vl_lce_forward = wrap_forward(qwen2_5_vl_lce_forward)

    if rope:
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = (
            liger_multimodal_rotary_pos_emb
        )
    if rms_norm:
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_qwen2_5_vl.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = (
            qwen2_5_vl_lce_forward
        )
    if swiglu:
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP

    if use_rmpad:
        from .qwen2_5_vl_ops import attn_forward as qwen2_ops_attn_forward
        from .qwen2_5_vl_ops import (
            decoder_layer_forward as qwen2_ops_decoder_layer_forward,
        )
        from .qwen2_5_vl_ops import text_model_forward as qwen2_ops_text_model_forward
        from .qwen2_5_vl_ops import vl_model_forward as qwen2_ops_vl_model_forward

        modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen2_ops_vl_model_forward
        modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = qwen2_ops_text_model_forward
        modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = (
            qwen2_ops_decoder_layer_forward
        )
        modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_ops_attn_forward

    if get_ulysses_sequence_parallel_world_size() > 1:
        patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, Qwen2_5_VLForConditionalGeneration):
            text_model: Qwen2_5_VLTextModel = model.model.language_model
            vision_model: Qwen2_5_VisionTransformerPretrainedModel = model.model.visual
        elif isinstance(model, Qwen2_5_VLModel):
            # Note: language_model and visual properties can be accessed throught conditional class for BC.
            # Not sure if it is subject to changes in the future.
            # Reference: https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1823
            text_model: Qwen2_5_VLTextModel = model.language_model
            vision_model: Qwen2_5_VisionTransformerPretrainedModel = model.visual
        elif isinstance(model, Qwen2_5_VLTextModel):
            text_model: Qwen2_5_VLTextModel = model
            vision_model = None
        else:
            # Note: Currently there's no support for patching vision model only. Feel free to raise an issue if needed.
            raise TypeError(
                f"Unsupported Qwen2VL model type. `model` must be `Qwen2VLForConditionalGeneration`, `Qwen2VLModel` or `Qwen2VLTextModel`. Got: {type(model)}"
            )

        if vision_model is not None:
            # Patch Qwen2_5_VisionTransformerPretrainedModel
            for vision_block in model.visual.blocks:
                if rms_norm:
                    _patch_rms_norm_module(vision_block.norm1)
                    _patch_rms_norm_module(vision_block.norm2)

        if text_model is not None:
            if rms_norm:
                _patch_rms_norm_module(text_model.norm)
            for decoder_layer in text_model.layers:
                if swiglu:
                    _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
                if rms_norm:
                    _patch_rms_norm_module(decoder_layer.input_layernorm)
                    _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
