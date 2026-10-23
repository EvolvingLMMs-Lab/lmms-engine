from __future__ import annotations

import sys
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path


@lru_cache(maxsize=1)
def ensure_lmms_eval_importable() -> Path:
    """Make the vendored lmms-eval package importable when it is not installed."""
    if find_spec("lmms_eval") is not None:
        import lmms_eval

        return Path(lmms_eval.__file__).resolve().parent.parent

    vendored_root = Path(__file__).resolve().parents[3] / "lmms-eval"
    if vendored_root.exists():
        sys.path.insert(0, str(vendored_root))
        import lmms_eval

        return Path(lmms_eval.__file__).resolve().parent.parent

    raise ModuleNotFoundError(
        "Could not import lmms_eval and vendored src/lmms-eval was not found. "
        "Install lmms-eval or run from a full lmms-engine checkout."
    )
