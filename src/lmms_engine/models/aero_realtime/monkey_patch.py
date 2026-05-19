"""Monkey patches for AeroRealtime training (OV2-style split).

The patcher is split into two independent entries, mirroring
``llava_onevision2`` (PR #170) and ``qwen3_5_moe`` (PR #171):

- ``liger`` — RoPE / RMSNorm / SwiGLU + binds aero's lce_forward
- ``rmpad`` — text/vision/audio rmpad ops + binds aero's rmpad forward

When both run (runner ordering ``["liger", "rmpad"]``), rmpad wins for
``forward`` rebinding, yielding rmpad + fused-LCE training.

Family dispatch lives in ``backbone_registry.family_liger_fn`` /
``family_rmpad_fn``.
"""

from transformers import PreTrainedModel

from lmms_engine.models.monkey_patch import MONKEY_PATCHER

from .backbone_registry import (
    family_liger_fn,
    family_rmpad_fn,
    family_vit_frame_parallel_fn,
)


@MONKEY_PATCHER.register("aero_realtime", "liger")
def apply_liger_kernel_to_aero_realtime(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = False,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """Apply Liger kernels to AeroRealtime's language sub-model.

    VoxtralRealtimeEncoder runs its own native forward — no liger patches
    applied to the audio tower (future optimization PR)."""
    from .modeling_aero_realtime import AeroRealtimeForConditionalGeneration

    if model is None:
        raise ValueError("apply_liger_kernel_to_aero_realtime requires model=...")
    if not isinstance(model, AeroRealtimeForConditionalGeneration):
        raise TypeError(f"Expected AeroRealtimeForConditionalGeneration, got {type(model)}")

    family = model.config.backbone_family
    liger_fn = family_liger_fn(family)

    # 1. Patch language sub-model
    liger_fn(
        model=model.language_model,
        rope=rope,
        rms_norm=rms_norm,
        swiglu=swiglu,
        cross_entropy=cross_entropy,
        fused_linear_cross_entropy=fused_linear_cross_entropy,
    )

    # 2. Bind aero's lce-flavoured forward
    from .aero_realtime_liger import aero_realtime_lce_forward

    AeroRealtimeForConditionalGeneration.forward = aero_realtime_lce_forward


@MONKEY_PATCHER.register("aero_realtime", "rmpad")
def apply_rmpad_to_aero_realtime(model: PreTrainedModel = None) -> None:
    """Apply rmpad ops to AeroRealtime's language sub-model + bind aero's
    rmpad-flavoured lce forward.

    VoxtralRealtimeEncoder runs its own native forward — no rmpad patches
    applied to the audio tower (future optimization PR)."""
    from .modeling_aero_realtime import AeroRealtimeForConditionalGeneration

    if model is None:
        raise ValueError("apply_rmpad_to_aero_realtime requires model=...")
    if not isinstance(model, AeroRealtimeForConditionalGeneration):
        raise TypeError(f"Expected AeroRealtimeForConditionalGeneration, got {type(model)}")

    family = model.config.backbone_family
    rmpad_fn = family_rmpad_fn(family)

    # 1. Patch language sub-model — rmpad-only
    rmpad_fn(model=model.language_model)

    # 2. Bind aero's rmpad forward (same lce forward — already rmpad-aware)
    from .aero_realtime_liger import aero_realtime_lce_forward

    AeroRealtimeForConditionalGeneration.forward = aero_realtime_lce_forward


@MONKEY_PATCHER.register("aero_realtime", "vit_frame_parallel")
def apply_vit_frame_parallel_to_aero_realtime(model: PreTrainedModel = None, **kwargs) -> None:
    """Wrap the family's VisionModel.forward with frame-parallel dispatch.

    Class-level patch — delegates to the inner family's frame-parallel fn
    (currently only ``qwen3_5`` supports this). For families without a
    frame-parallel implementation this is a no-op.

    Aero's ``model.vision_tower`` is an instance of the family's VisionModel
    class, so patching the class also patches this instance.
    """
    from loguru import logger

    from .modeling_aero_realtime import AeroRealtimeForConditionalGeneration

    if model is None:
        raise ValueError("apply_vit_frame_parallel_to_aero_realtime requires model=...")
    if not isinstance(model, AeroRealtimeForConditionalGeneration):
        raise TypeError(f"Expected AeroRealtimeForConditionalGeneration, got {type(model)}")

    family = model.config.backbone_family
    fp_fn = family_vit_frame_parallel_fn(family)
    if fp_fn is None:
        logger.info(f"vit_frame_parallel: backbone_family={family} has no frame-parallel ViT impl, skipping")
        return
    # Inner family fns ignore the ``model`` arg (class-level patch).
    fp_fn(model=model.vision_tower, **kwargs)
