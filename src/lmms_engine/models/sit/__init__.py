from lmms_engine.mapping_func import register_model
from lmms_engine.utils.import_utils import _is_package_available

from .configuration_sit import SiTConfig

# Check if optional dependencies are available
_has_deps = _is_package_available("timm") and _is_package_available("torchdiffeq")

if _has_deps:
    from .modeling_sit import SiTModel
    from .models import SiT

    register_model(
        "sit",
        SiTConfig,
        SiTModel,
    )
else:
    # Create stub classes with helpful error messages
    class SiTModel:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SiT model requires optional dependencies.\n" "Install with: pip install lmms_engine[sit]"
            )

    class SiT:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SiT model requires optional dependencies.\n" "Install with: pip install lmms_engine[sit]"
            )


__all__ = [
    "SiTModel",
    "SiTConfig",
    "SiT",
]
