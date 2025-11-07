import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_model_imports():
    """Test that model classes can be imported."""
    print("=" * 70)
    print("Testing Qwen3-Omni MoE Model Imports")
    print("=" * 70)

    try:
        # Note: These imports will fail if transformers doesn't have Qwen3-Omni support
        # This is expected - user needs transformers with Qwen3-Omni
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeThinkerConfig,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )

        print("✅ Model classes imported successfully from transformers")
        return True
    except ImportError as e:
        print(f"❌ Failed to import model classes from transformers: {e}")
        print("   This is expected if transformers doesn't have Qwen3-Omni support yet")
        print("   Skipping model registration test...")
        return False


def test_lmms_engine_imports():
    """Test that LMMs Engine components can be imported."""
    print("\n" + "=" * 70)
    print("Testing LMMs Engine Imports")
    print("=" * 70)

    try:
        # Test model module imports
        from lmms_engine.models.qwen3_omni_moe import (
            apply_liger_kernel_to_qwen3_omni_moe,
        )

        print("✅ Monkey patch function imported successfully")

        # Test processor imports
        from lmms_engine.datasets.processor.qwen3_omni_moe_processor import (
            Qwen3OmniMoeDataProcessor,
        )

        print("✅ Processor class imported successfully")

        # Test dataset imports (should reuse existing qwen_omni)
        from lmms_engine.datasets.naive.qwen_omni_dataset import QwenOmniDataset

        print("✅ Dataset class imported successfully")

        return True
    except ImportError as e:
        print(f"❌ Failed to import LMMs Engine components: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_registration():
    """Test that model is registered with the framework."""
    print("\n" + "=" * 70)
    print("Testing Model Registration")
    print("=" * 70)

    try:
        from lmms_engine.mapping_func import MODEL_MAPPING

        if "qwen3_omni_moe_thinker" in MODEL_MAPPING:
            print("✅ Model type 'qwen3_omni_moe_thinker' is registered")
            model_info = MODEL_MAPPING["qwen3_omni_moe_thinker"]
            print(f"   Config class: {model_info['config'].__name__}")
            print(f"   Model class: {model_info['model'].__name__}")
            return True
        else:
            print("❌ Model type 'qwen3_omni_moe_thinker' is NOT registered")
            print(f"   Available models: {list(MODEL_MAPPING.keys())}")
            return False
    except Exception as e:
        print(f"❌ Error checking model registration: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_processor_registration():
    """Test that processor is registered with the framework."""
    print("\n" + "=" * 70)
    print("Testing Processor Registration")
    print("=" * 70)

    try:
        from lmms_engine.mapping_func import DATAPROCESSOR_MAPPING

        if "Qwen3OmniMoeProcessor" in DATAPROCESSOR_MAPPING:
            print("✅ Processor 'Qwen3OmniMoeProcessor' is registered")
            processor_class = DATAPROCESSOR_MAPPING["Qwen3OmniMoeProcessor"]
            print(f"   Processor class: {processor_class.__name__}")
            return True
        else:
            print("❌ Processor 'Qwen3OmniMoeProcessor' is NOT registered")
            print(f"   Available processors: {list(DATAPROCESSOR_MAPPING.keys())}")
            return False
    except Exception as e:
        print(f"❌ Error checking processor registration: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\n" + "=" * 70)
    print("Testing File Structure")
    print("=" * 70)

    required_files = [
        "scripts/extract_qwen3_omni_thinker.py",
        "src/lmms_engine/models/qwen3_omni_moe/__init__.py",
        "src/lmms_engine/models/qwen3_omni_moe/monkey_patch.py",
        "src/lmms_engine/models/qwen3_omni_moe/qwen3_omni_moe_liger.py",
        "src/lmms_engine/models/qwen3_omni_moe/qwen3_omni_moe_ops.py",
        "src/lmms_engine/datasets/processor/qwen3_omni_moe_processor.py",
        "examples/qwen3_omni_moe.yaml",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False

    return all_exist


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Qwen3-Omni MoE Integration Test Suite")
    print("=" * 70 + "\n")

    results = {
        "File Structure": test_file_structure(),
        "LMMs Engine Imports": test_lmms_engine_imports(),
        "Model Imports": test_model_imports(),
        "Model Registration": test_model_registration(),
        "Processor Registration": test_processor_registration(),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed - see details above")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
