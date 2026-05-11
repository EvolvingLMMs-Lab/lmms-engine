from functools import partial, wraps

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.monkey_patch import (
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.rms_norm import LigerRMSNormForQwen3Next
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP
except Exception:
    print("liger kernel not installed, please install it with `pip install liger-kernel`")

from loguru import logger
from transformers import PreTrainedModel

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.models.monkey_patch import MONKEY_PATCHER
from lmms_engine.parallel.vit_parallel.frame_parallel import wrap_vit_forward


@MONKEY_PATCHER.register("qwen3_5", "liger")
def apply_liger_kernel_to_qwen3_5(
    rope: bool = False,
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
        raise NotImplementedError(
            "liger_rotary_pos_emb is not available for Qwen3.5 (hybrid attention: "
            "Gated DeltaNet + Gated Attention). Keep rope=False."
        )
    if rms_norm:
        modeling_qwen3_5.Qwen3_5RMSNorm = LigerRMSNormForQwen3Next

    if fused_linear_cross_entropy:
        from .qwen3_5_liger import qwen3_5_lce_forward

        if use_rmpad:

            def wrap_forward(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(use_rmpad=use_rmpad, *args, **kwargs)

                return wrapper

            qwen3_5_lce_forward = wrap_forward(qwen3_5_lce_forward)
        # Both heads share the same LCE forward shape; patch whichever is
        # actually used at runtime.
        modeling_qwen3_5.Qwen3_5ForCausalLM.forward = qwen3_5_lce_forward
        modeling_qwen3_5.Qwen3_5ForConditionalGeneration.forward = qwen3_5_lce_forward

    if swiglu:
        modeling_qwen3_5.Qwen3_5MLP = LigerQwen3MoeSwiGLUMLP

    if use_rmpad:
        from .qwen3_5_ops import attn_forward as qwen3_5_ops_attn_forward
        from .qwen3_5_ops import (
            decoder_layer_forward as qwen3_5_ops_decoder_layer_forward,
        )
        from .qwen3_5_ops import linear_attn_forward as qwen3_5_ops_linear_attn_forward
        from .qwen3_5_ops import model_forward as qwen3_5_ops_model_forward
        from .qwen3_5_ops import text_model_forward as qwen3_5_ops_text_model_forward

        modeling_qwen3_5.Qwen3_5Model.forward = qwen3_5_ops_model_forward
        modeling_qwen3_5.Qwen3_5TextModel.forward = qwen3_5_ops_text_model_forward
        modeling_qwen3_5.Qwen3_5DecoderLayer.forward = qwen3_5_ops_decoder_layer_forward
        modeling_qwen3_5.Qwen3_5Attention.forward = qwen3_5_ops_attn_forward
        modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward = qwen3_5_ops_linear_attn_forward

    # Replace VisionPatchEmbed.forward with a Linear path. Mathematically
    # equivalent to the upstream Conv3d (kernel == stride), but avoids cudnn
    # falling back to a slow Conv3d kernel on packed varlen ViT inputs.
    from .qwen3_5_ops import patch_embed_forward as qwen3_5_ops_patch_embed_forward

    modeling_qwen3_5.Qwen3_5VisionPatchEmbed.forward = qwen3_5_ops_patch_embed_forward

    if model is not None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
            Qwen3_5Model,
            Qwen3_5TextModel,
        )

        # Navigate to the Qwen3_5TextModel (which carries .norm / .layers).
        if isinstance(model, Qwen3_5ForCausalLM):
            base_model: Qwen3_5TextModel = model.model
        elif isinstance(model, Qwen3_5ForConditionalGeneration):
            base_model: Qwen3_5TextModel = model.model.language_model
        elif isinstance(model, Qwen3_5Model):
            base_model: Qwen3_5TextModel = model.language_model
        elif isinstance(model, Qwen3_5TextModel):
            base_model = model
        else:
            base_model = getattr(model, "model", model)

        _patch_qwen3_5_rms_norm = partial(_patch_rms_norm_module, offset=1.0, casting_mode="gemma", in_place=False)

        if rms_norm:
            _patch_qwen3_5_rms_norm(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerQwen3MoeSwiGLUMLP)
            if rms_norm:
                _patch_qwen3_5_rms_norm(decoder_layer.input_layernorm)
                _patch_qwen3_5_rms_norm(decoder_layer.post_attention_layernorm)


@MONKEY_PATCHER.register("qwen3_5", "vit_frame_parallel")
def apply_vit_frame_parallel_to_qwen3_5(model: PreTrainedModel = None, **kwargs) -> None:
    """Wrap ``Qwen3_5VisionModel.forward`` with frame-parallel dispatch.

    Frames are redistributed across the DP group via LPT so each rank handles
    a balanced number of ViT patches. ``pgm.process_group_manager.dp_group``
    must be initialized before this runs.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5

    from .qwen3_5_vit_ops import input_dispatch, output_dispatch

    if pgm.process_group_manager is None or pgm.process_group_manager.dp_world_size <= 1:
        logger.info("vit_frame_parallel: dp_world_size <= 1, skipping ViT wrap")
        return

    dp_group = pgm.process_group_manager.dp_group
    orig_forward = modeling_qwen3_5.Qwen3_5VisionModel.forward

    wrapped = wrap_vit_forward(
        input_dispatch=partial(input_dispatch, group=dp_group),
        orig_forward=orig_forward,
        output_dispatch=output_dispatch,
    )
    modeling_qwen3_5.Qwen3_5VisionModel.forward = wrapped
    logger.info(
        f"vit_frame_parallel: wrapped Qwen3_5VisionModel.forward (dp_size={pgm.process_group_manager.dp_world_size})"
    )
