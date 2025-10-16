"""Helper functions for importing optional dependencies for SiT models."""

from typing import Any


def get_timm_attention() -> Any:
    """Get timm Attention class with helpful error message."""
    try:
        from timm.models.vision_transformer import Attention

        return Attention
    except ImportError as e:
        raise ImportError(
            "timm is required for SiT models.\n"
            "Install with: pip install lmms_engine[sit]"
        ) from e


def get_timm_mlp() -> Any:
    """Get timm Mlp class with helpful error message."""
    try:
        from timm.models.vision_transformer import Mlp

        return Mlp
    except ImportError as e:
        raise ImportError(
            "timm is required for SiT models.\n"
            "Install with: pip install lmms_engine[sit]"
        ) from e


def get_timm_patch_embed() -> Any:
    """Get timm PatchEmbed class with helpful error message."""
    try:
        from timm.models.vision_transformer import PatchEmbed

        return PatchEmbed
    except ImportError as e:
        raise ImportError(
            "timm is required for SiT models.\n"
            "Install with: pip install lmms_engine[sit]"
        ) from e


def get_torchdiffeq_odeint() -> Any:
    """Get torchdiffeq odeint function with helpful error message."""
    try:
        from torchdiffeq import odeint

        return odeint
    except ImportError as e:
        raise ImportError(
            "torchdiffeq is required for SiT models.\n"
            "Install with: pip install lmms_engine[sit]"
        ) from e
