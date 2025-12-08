from functools import partial, wraps

from packaging import version

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.monkey_patch import (
        _patch_layer_norm_module,
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except:
    print("liger kernel not installed, please install it with `pip install liger-kernel`")

import transformers
from transformers import PreTrainedModel
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModel,
    Qwen3VLMoeTextModel,
    Qwen3VLMoeTextSparseMoeBlock,
    Qwen3VLMoeVisionModel,
)

from lmms_engine.parallel.sequence_parallel.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    patch_vlm_for_ulysses_input_slicing,
)

transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.utils.logging_utils import Logging


@MONKEY_PATCHER.register("qwen3_vl_moe", "liger")
def apply_liger_kernel_to_qwen3_vl_moe(
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

    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    from .qwen3_vl_moe_liger import lce_forward as qwen3_vl_moe_lce_forward
    from .qwen3_vl_moe_ops import (
        moe_sparse_layer_forward as qwen3_vl_moe_moe_sparse_layer_forward,
    )

    def wrap_forward(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.setdefault("use_rmpad", use_rmpad)
            return func(*args, **kwargs)

        return wrapper

    qwen3_vl_moe_lce_forward = wrap_forward(qwen3_vl_moe_lce_forward)
    if rope:
        Logging.warning("RoPE optimization not supported for Qwen3-VL MoE, skipping")
    if rms_norm:
        modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm = LigerRMSNorm

    if cross_entropy:
        modeling_qwen3_vl_moe.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_qwen3_vl_moe.Qwen3VLMoeForConditionalGeneration.forward = qwen3_vl_moe_lce_forward
    if swiglu:
        modeling_qwen3_vl_moe.Qwen3VLMoeTextMLP = LigerSwiGLUMLP
    if use_rmpad:
        from .qwen3_vl_moe_ops import attn_forward as qwen3_vl_moe_attn_forward
        from .qwen3_vl_moe_ops import (
            decoder_layer_forward as qwen3_vl_moe_decoder_layer_forward,
        )
        from .qwen3_vl_moe_ops import experts_forward as qwen3_vl_moe_experts_forward
        from .qwen3_vl_moe_ops import model_forward as qwen3_vl_moe_model_forward
        from .qwen3_vl_moe_ops import (
            text_model_forward as qwen3_vl_moe_text_model_forward,
        )

        modeling_qwen3_vl_moe.Qwen3VLMoeModel.forward = qwen3_vl_moe_model_forward
        modeling_qwen3_vl_moe.Qwen3VLMoeTextModel.forward = qwen3_vl_moe_text_model_forward
        modeling_qwen3_vl_moe.Qwen3VLMoeTextDecoderLayer.forward = qwen3_vl_moe_decoder_layer_forward
        modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = qwen3_vl_moe_attn_forward

    if get_ulysses_sequence_parallel_world_size() > 1:
        patch_vlm_for_ulysses_input_slicing(modeling_qwen3_vl_moe.Qwen3VLMoeModel)

    if model is not None:
        if isinstance(model, Qwen3VLMoeForConditionalGeneration):
            main_model: Qwen3VLMoeModel = model.model
            vision_model: Qwen3VLMoeVisionModel = main_model.visual
            text_model: Qwen3VLMoeTextModel = main_model.language_model
        elif isinstance(model, Qwen3VLMoeModel):
            main_model: Qwen3VLMoeModel = model
            vision_model: Qwen3VLMoeVisionModel = main_model.visual
            text_model: Qwen3VLMoeTextModel = main_model.language_model
        elif isinstance(model, Qwen3VLMoeTextModel):
            text_model: Qwen3VLMoeTextModel = model
            vision_model = None
        else:
            raise TypeError(
                f"Unsupported Qwen3-VL MoE model type. `model` must be "
                f"`Qwen3VLMoeForConditionalGeneration`, `Qwen3VLMoeModel`, or `Qwen3VLMoeTextModel`. "
                f"Got: {type(model)}."
            )

        _patch_qwen3_vl_moe_rms_norm = partial(_patch_rms_norm_module, offset=0.0, casting_mode="llama")

        if text_model is not None:
            if rms_norm:
                _patch_qwen3_vl_moe_rms_norm(text_model.norm)
            for decoder_layer in text_model.layers:
                if rms_norm:
                    _patch_qwen3_vl_moe_rms_norm(decoder_layer.input_layernorm)
                    _patch_qwen3_vl_moe_rms_norm(decoder_layer.post_attention_layernorm)
                    self_attn = getattr(decoder_layer, "self_attn", None)
                    if self_attn is not None:
                        if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                            _patch_qwen3_vl_moe_rms_norm(self_attn.q_norm)
                        if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                            _patch_qwen3_vl_moe_rms_norm(self_attn.k_norm)

        if vision_model is not None:
            for vision_block in vision_model.blocks:
                _patch_layer_norm_module(vision_block.norm1)
                _patch_layer_norm_module(vision_block.norm2)

    if pgm.process_group_manager.enable_parallel:
        modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts.forward = qwen3_vl_moe_experts_forward
        modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock.forward = qwen3_vl_moe_moe_sparse_layer_forward
