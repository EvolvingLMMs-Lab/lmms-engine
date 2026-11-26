#!/usr/bin/env python3
"""Registration smoke tests for the MiniCPM-o integration."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_file_structure() -> bool:
    required = [
        project_root / "src" / "lmms_engine" / "models" / "minicpmo" / "__init__.py",
        project_root / "src" / "lmms_engine" / "models" / "minicpmo" / "modeling_minicpmo.py",
        project_root / "src" / "lmms_engine" / "models" / "minicpmo" / "processing_minicpmo.py",
        project_root / "src" / "lmms_engine" / "datasets" / "processor" / "minicpmo_processor.py",
        project_root / "src" / "lmms_engine" / "datasets" / "iterable" / "minicpmo_iterable_dataset.py",
        project_root / "src" / "lmms_engine" / "datasets" / "iterable" / "minicpmo_realtime_iterable_dataset.py",
        project_root / "src" / "lmms_engine" / "datasets" / "collator" / "minicpmo_collator.py",
        project_root / "examples" / "minicpmo" / "example_config.yaml",
        project_root / "examples" / "minicpmo" / "realtime_config.yaml",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("❌ Missing files:", missing)
        return False
    print("✅ file structure")
    return True


def test_lmms_engine_imports() -> bool:
    try:
        from lmms_engine.datasets.collator.minicpmo_collator import MiniCPMOCollator  # noqa: F401
        from lmms_engine.datasets.iterable.minicpmo_iterable_dataset import (  # noqa: F401
            MiniCPMOIterableDataset,
        )
        from lmms_engine.datasets.iterable.minicpmo_realtime_iterable_dataset import (  # noqa: F401
            MiniCPMORealtimeIterableDataset,
        )
        from lmms_engine.datasets.processor.minicpmo_processor import MiniCPMODataProcessor  # noqa: F401
        from lmms_engine.models.minicpmo import MiniCPMOConfig, MiniCPMOForTraining, MiniCPMOProcessor  # noqa: F401
    except ImportError as e:
        print(f"❌ lmms_engine import failed: {e}")
        return False
    print("✅ lmms_engine imports")
    return True


def test_model_registration() -> bool:
    from lmms_engine.models.minicpmo import MiniCPMOConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "minicpmo" not in CONFIG_MAPPING:
        print("❌ minicpmo not registered in AutoConfig")
        return False
    from transformers import AutoModelForCausalLM

    if MiniCPMOConfig not in AutoModelForCausalLM._model_mapping.keys():
        print("❌ MiniCPMOConfig not in AutoModelForCausalLM mapping")
        return False
    print("✅ model registration")
    return True


def test_dataset_registration() -> bool:
    from lmms_engine.mapping_func import DATASET_MAPPING, DATAPROCESSOR_MAPPING

    ok = True
    for name in ("minicpmo_iterable", "minicpmo_realtime_iterable"):
        if name not in DATASET_MAPPING:
            print(f"❌ dataset {name} not registered")
            ok = False
    if "minicpmo" not in DATAPROCESSOR_MAPPING:
        print("❌ processor minicpmo not registered")
        ok = False
    if ok:
        print("✅ dataset/processor registration")
    return ok


def main() -> bool:
    results = [test_file_structure(), test_lmms_engine_imports(), test_model_registration(), test_dataset_registration()]
    print(f"\n{'✅ ALL PASSED' if all(results) else '❌ SOME FAILED'} ({sum(results)}/{len(results)})")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
