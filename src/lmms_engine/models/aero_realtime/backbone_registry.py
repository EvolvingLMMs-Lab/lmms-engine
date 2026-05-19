# coding=utf-8
"""Backbone family registry for AeroRealtime.

Single source of truth mapping ``backbone_family`` ∈
``{qwen3_vl, qwen3_vl_moe, qwen3_5, qwen3_5_moe}`` to:

- Sub-config classes (text + vision)
- Sub-model classes (text + vision)

Sub-configs/sub-models are NOT in transformers.AutoConfig / AutoModel
mappings, so we import the concrete classes directly. The audio tower
(``qwen2_audio_encoder``) IS registered in AutoConfig/AutoModel and stays
on the auto path — it lives outside this registry.

Monkey-patch dispatch tables (LIGER_FN / RMPAD_FN) are built lazily by
``family_liger_fn`` / ``family_rmpad_fn`` to avoid eager-importing
inner backbone patch modules at module load.
"""

from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextModel,
    Qwen3_5VisionModel,
)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextModel,
    Qwen3_5MoeVisionModel,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextModel,
    Qwen3VLMoeVisionModel,
)

# Family → (text_model_type, vision_model_type, TextConfig, VisionConfig,
#          TextModel, VisionModel, is_moe)
BACKBONE_FAMILY = {
    "qwen3_vl": (
        "qwen3_vl_text",
        "qwen3_vl_vision",
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
        Qwen3VLTextModel,
        Qwen3VLVisionModel,
        False,
    ),
    "qwen3_vl_moe": (
        "qwen3_vl_moe_text",
        "qwen3_vl_moe_vision",
        Qwen3VLMoeTextConfig,
        Qwen3VLMoeVisionConfig,
        Qwen3VLMoeTextModel,
        Qwen3VLMoeVisionModel,
        True,
    ),
    "qwen3_5": (
        "qwen3_5_text",
        "qwen3_5_vision",
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
        Qwen3_5TextModel,
        Qwen3_5VisionModel,
        False,
    ),
    "qwen3_5_moe": (
        "qwen3_5_moe_text",
        "qwen3_5_moe_vision",
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeVisionConfig,
        Qwen3_5MoeTextModel,
        Qwen3_5MoeVisionModel,
        True,
    ),
}


def get_family_entry(family: str):
    if family not in BACKBONE_FAMILY:
        raise ValueError(f"unknown backbone_family={family}; " f"expected one of {sorted(BACKBONE_FAMILY)}")
    return BACKBONE_FAMILY[family]


def family_text_model_type(family: str) -> str:
    return get_family_entry(family)[0]


def family_vision_model_type(family: str) -> str:
    return get_family_entry(family)[1]


def family_text_config_cls(family: str):
    return get_family_entry(family)[2]


def family_vision_config_cls(family: str):
    return get_family_entry(family)[3]


def family_text_model_cls(family: str):
    return get_family_entry(family)[4]


def family_vision_model_cls(family: str):
    return get_family_entry(family)[5]


def family_is_moe(family: str) -> bool:
    return get_family_entry(family)[6]


# ---------------------------------------------------------------------------
# Monkey-patch dispatch tables.
# Each callable takes ``model=<aero.language_model>`` (+ **liger_flags for
# the liger variant) and applies the family's patches.
#
# For qwen3_5_moe we use the OV2-split entries directly. For the other
# three families (still combined-style), we adapt via lambda with
# use_rmpad=True/False and per-flag toggles.
#
# Tables are built lazily inside ``_build_family_patch_dispatch`` because
# importing the inner backbone monkey_patch modules at module load would
# eagerly import their model classes and slow down config-only consumers.
# ---------------------------------------------------------------------------


def _build_family_patch_dispatch():
    from lmms_engine.models.qwen3_5.monkey_patch import apply_liger_kernel_to_qwen3_5
    from lmms_engine.models.qwen3_5_moe.monkey_patch import (
        apply_liger_kernel_to_qwen3_5_moe,
        apply_rmpad_to_qwen3_5_moe,
    )
    from lmms_engine.models.qwen3_vl.monkey_patch import apply_liger_kernel_to_qwen3_vl
    from lmms_engine.models.qwen3_vl_moe.monkey_patch import (
        apply_liger_kernel_to_qwen3_vl_moe,
    )

    LIGER_FN = {
        "qwen3_vl": lambda model=None, **kw: apply_liger_kernel_to_qwen3_vl(model=model, use_rmpad=False, **kw),
        "qwen3_vl_moe": lambda model=None, **kw: apply_liger_kernel_to_qwen3_vl_moe(model=model, use_rmpad=False, **kw),
        "qwen3_5": lambda model=None, **kw: apply_liger_kernel_to_qwen3_5(model=model, use_rmpad=False, **kw),
        "qwen3_5_moe": apply_liger_kernel_to_qwen3_5_moe,
    }
    RMPAD_FN = {
        "qwen3_vl": lambda model=None: apply_liger_kernel_to_qwen3_vl(
            model=model,
            use_rmpad=True,
            rope=False,
            rms_norm=False,
            swiglu=False,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        ),
        "qwen3_vl_moe": lambda model=None: apply_liger_kernel_to_qwen3_vl_moe(
            model=model,
            use_rmpad=True,
            rope=False,
            rms_norm=False,
            swiglu=False,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        ),
        "qwen3_5": lambda model=None: apply_liger_kernel_to_qwen3_5(
            model=model,
            use_rmpad=True,
            rope=False,
            rms_norm=False,
            swiglu=False,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        ),
        "qwen3_5_moe": apply_rmpad_to_qwen3_5_moe,
    }
    return LIGER_FN, RMPAD_FN


def family_liger_fn(family: str):
    LIGER_FN, _ = _build_family_patch_dispatch()
    if family not in LIGER_FN:
        raise ValueError(f"no liger entry for backbone_family={family}")
    return LIGER_FN[family]


def family_rmpad_fn(family: str):
    _, RMPAD_FN = _build_family_patch_dispatch()
    if family not in RMPAD_FN:
        raise ValueError(f"no rmpad entry for backbone_family={family}")
    return RMPAD_FN[family]


# ---------------------------------------------------------------------------
# ViT frame-parallel dispatch (optional per family).
#
# Only ``qwen3_5`` exposes a frame-parallel ViT wrap today; the other three
# families fall back to no-op. The wrap is a *class-level* monkey-patch
# on the family's VisionModel class, so it works for aero out of the box
# once the right family fn is called.
# ---------------------------------------------------------------------------


def _build_family_vit_frame_parallel_dispatch():
    from lmms_engine.models.qwen3_5.monkey_patch import (
        apply_vit_frame_parallel_to_qwen3_5,
    )

    return {
        "qwen3_5": apply_vit_frame_parallel_to_qwen3_5,
    }


def family_vit_frame_parallel_fn(family: str):
    """Return the family's frame-parallel ViT wrap fn, or ``None`` if the
    family doesn't support frame-parallel ViT."""
    table = _build_family_vit_frame_parallel_dispatch()
    return table.get(family)
