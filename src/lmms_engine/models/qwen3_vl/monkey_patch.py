import inspect
from functools import partial, wraps
from typing import Callable

from loguru import logger
from packaging import version
from transformers import PreTrainedModel, Qwen3VLTextModel

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.parallel.sequence_parallel.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    patch_vlm_for_ulysses_input_slicing,
)
from lmms_engine.parallel.vit_parallel.frame_parallel import wrap_vit_forward

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.monkey_patch import (
        _patch_layer_norm_module,
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import (
        liger_rotary_pos_emb,
        liger_rotary_pos_emb_vision,
    )
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
except:
    print("liger kernel not installed, please install it with `pip install liger-kernel`")

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("qwen3_vl", "liger")
def apply_liger_kernel_to_qwen3_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_rmpad: bool = False,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3-VL models.
    NOTE: liger-kernel not yet supported for Qwen3-VL. only fused linear ce, rope are supported
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

    from transformers.models.qwen3_vl import modeling_qwen3_vl
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLModel,
        Qwen3VLVisionModel,
    )

    from .qwen3_vl_liger import qwen3_vl_lce_forward as qwen3_vl_lce_forward

    if use_rmpad:

        def wrap_forward(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(use_rmpad=use_rmpad, *args, **kwargs)

            return wrapper

        qwen3_vl_lce_forward = wrap_forward(qwen3_vl_lce_forward)

    if rope:
        modeling_qwen3_vl.apply_rotary_pos_emb = liger_rotary_pos_emb
        modeling_qwen3_vl.apply_rotary_pos_emb_vision = liger_rotary_pos_emb_vision
    if rms_norm:
        modeling_qwen3_vl.Qwen3VLTextRMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_qwen3_vl.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward = qwen3_vl_lce_forward
    if swiglu:
        modeling_qwen3_vl.Qwen3VLTextMLP = LigerSwiGLUMLP

    if use_rmpad:
        from .qwen3_vl_ops import attn_forward as qwen3_ops_attn_forward
        from .qwen3_vl_ops import (
            decoder_layer_forward as qwen3_ops_decoder_layer_forward,
        )
        from .qwen3_vl_ops import model_forward as qwen3_ops_model_forward
        from .qwen3_vl_ops import text_model_forward as qwen3_ops_text_model_forward

        modeling_qwen3_vl.Qwen3VLModel.forward = qwen3_ops_model_forward
        modeling_qwen3_vl.Qwen3VLTextModel.forward = qwen3_ops_text_model_forward
        modeling_qwen3_vl.Qwen3VLTextDecoderLayer.forward = qwen3_ops_decoder_layer_forward
        modeling_qwen3_vl.Qwen3VLTextAttention.forward = qwen3_ops_attn_forward

    if get_ulysses_sequence_parallel_world_size() > 1:
        patch_vlm_for_ulysses_input_slicing(Qwen3VLTextModel)

    # Use linear instead of conv3d
    from .qwen3_vl_ops import patch_embed_forward as qwen3_ops_patch_embed_forward

    modeling_qwen3_vl.Qwen3VLVisionPatchEmbed.forward = qwen3_ops_patch_embed_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        if isinstance(model, Qwen3VLForConditionalGeneration):
            text_model: Qwen3VLTextModel = model.model.language_model
            vision_model: Qwen3VLVisionModel = model.model.visual
        elif isinstance(model, Qwen3VLModel):
            text_model: Qwen3VLTextModel = model.language_model
            vision_model: Qwen3VLVisionModel = model.visual
        elif isinstance(model, Qwen3VLTextModel):
            text_model: Qwen3VLTextModel = model
            vision_model = None

        _patch_qwen3_vl_rms_norm = partial(_patch_rms_norm_module, offset=0.0, casting_mode="llama")

        if text_model is not None:
            if rms_norm:
                _patch_qwen3_vl_rms_norm(text_model.norm)
            for decoder_layer in text_model.layers:
                if rms_norm:
                    _patch_qwen3_vl_rms_norm(decoder_layer.input_layernorm)
                    _patch_qwen3_vl_rms_norm(decoder_layer.post_attention_layernorm)
                    self_attn = getattr(decoder_layer, "self_attn", None)
                    if self_attn is not None:
                        if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                            _patch_qwen3_vl_rms_norm(self_attn.q_norm)
                        if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                            _patch_qwen3_vl_rms_norm(self_attn.k_norm)
                if swiglu:
                    _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)

        if vision_model is not None:
            for vision_block in vision_model.blocks:
                _patch_layer_norm_module(vision_block.norm1)
                _patch_layer_norm_module(vision_block.norm2)


@MONKEY_PATCHER.register("qwen3_vl", "vit_frame_parallel")
def apply_vit_frame_parallel_to_qwen3_vl(model: PreTrainedModel = None, **kwargs) -> None:
    """Wrap ``Qwen3VLVisionModel.forward`` with DPxCP frame-parallel dispatch."""
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    from .qwen3_vl_vit_ops import input_dispatch, output_dispatch

    if pgm.process_group_manager is None:
        logger.info("vit_frame_parallel: process_group_manager not initialized, skipping ViT wrap")
        return

    dp_cp_world_size = pgm.process_group_manager.dp_cp_world_size
    if dp_cp_world_size <= 1:
        logger.info("vit_frame_parallel: dp_cp_world_size <= 1, skipping ViT wrap")
        return

    dp_cp_group = pgm.process_group_manager.dp_cp_group
    cp_group = pgm.process_group_manager.cp_group if pgm.process_group_manager.cp_world_size > 1 else None
    orig_forward = modeling_qwen3_vl.Qwen3VLVisionModel.forward

    wrapped = wrap_vit_forward(
        input_dispatch=partial(input_dispatch, group=dp_cp_group, cp_group=cp_group),
        orig_forward=orig_forward,
        output_dispatch=output_dispatch,
    )
    modeling_qwen3_vl.Qwen3VLVisionModel.forward = wrapped
    logger.info(
        f"vit_frame_parallel: wrapped Qwen3VLVisionModel.forward "
        f"(dp_cp_size={dp_cp_world_size}, cp_size={pgm.process_group_manager.cp_world_size})"
    )
