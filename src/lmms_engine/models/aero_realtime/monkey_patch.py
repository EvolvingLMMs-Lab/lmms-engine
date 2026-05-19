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

from .backbone_registry import family_liger_fn, family_rmpad_fn


@MONKEY_PATCHER.register("aero_realtime", "liger")
def apply_liger_kernel_to_aero_realtime(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = False,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """Apply Liger kernels to AeroRealtime's language + audio sub-models."""
    from .modeling_aero_realtime import AeroRealtimeForConditionalGeneration

    if model is None:
        raise ValueError("apply_liger_kernel_to_aero_realtime requires model=...")
    if not isinstance(model, AeroRealtimeForConditionalGeneration):
        raise TypeError(f"Expected AeroRealtimeForConditionalGeneration, got {type(model)}")

    from lmms_engine.models.qwen2_audio.monkey_patch import (
        apply_liger_kernel_to_qwen2_audio,
    )

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

    # 2. Patch audio sub-model
    apply_liger_kernel_to_qwen2_audio(model=model.audio_tower)

    # 3. Bind aero's lce-flavoured forward
    from .aero_realtime_liger import aero_realtime_lce_forward

    AeroRealtimeForConditionalGeneration.forward = aero_realtime_lce_forward


@MONKEY_PATCHER.register("aero_realtime", "rmpad")
def apply_rmpad_to_aero_realtime(model: PreTrainedModel = None) -> None:
    """Apply rmpad ops to AeroRealtime's language + audio sub-models +
    bind aero's rmpad-flavoured lce forward."""
    from .modeling_aero_realtime import AeroRealtimeForConditionalGeneration

    if model is None:
        raise ValueError("apply_rmpad_to_aero_realtime requires model=...")
    if not isinstance(model, AeroRealtimeForConditionalGeneration):
        raise TypeError(f"Expected AeroRealtimeForConditionalGeneration, got {type(model)}")

    from lmms_engine.models.qwen2_audio.monkey_patch import (
        apply_liger_kernel_to_qwen2_audio,
    )

    family = model.config.backbone_family
    rmpad_fn = family_rmpad_fn(family)

    # 1. Patch language sub-model — rmpad-only
    rmpad_fn(model=model.language_model)

    # 2. Patch audio sub-model with rmpad
    apply_liger_kernel_to_qwen2_audio(model=model.audio_tower, use_rmpad=True)

    # 3. Bind aero's rmpad forward (same lce forward — already rmpad-aware)
    from .aero_realtime_liger import aero_realtime_lce_forward

    AeroRealtimeForConditionalGeneration.forward = aero_realtime_lce_forward
