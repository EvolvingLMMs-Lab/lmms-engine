import importlib
import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, version


def patch_bundled_flash_attention_cute():
    try:
        version("flash-attn-4")
    except PackageNotFoundError:
        pass
    else:
        return

    try:
        importlib.import_module("flash_attn.cute")
    except ImportError:
        return
    except AttributeError as error:
        if error.name != "ThrMma" or getattr(error.obj, "__name__", None) != "cutlass.cute.core":
            raise
    else:
        return

    for name in list(sys.modules):
        if name == "flash_attn.cute" or name.startswith("flash_attn.cute."):
            del sys.modules[name]
    # Megatron catches ImportError for optional FA4, but not the bundled CuTe API error.
    sys.modules["flash_attn.cute"] = None
    if int(os.environ.get("RANK", "0")) == 0:
        logging.getLogger(__name__).warning(
            "Skipping bundled flash_attn.cute: CUTLASS lacks ThrMma and flash-attn-4 is not installed"
        )
