import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_model_imports():
    """Test that model classes can be imported."""
    print("=" * 70)
    print("Testing Qwen3 VL MoE Model Imports")
    print("=" * 70)

    try:
        from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
            Qwen3VLMoeConfig,
        )
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeForConditionalGeneration,
        )

        print("Model classes imported successfully from transformers")
        return True
    except ImportError as e:
        print(f"Failed to import model classes from transformers: {e}")
        print("   This is expected if transformers doesn't have Qwen3 VL MoE support yet")
        print("   Skipping model registration test...")
        return False


def test_lmms_engine_imports():
    """Test that LMMs Engine components can be imported."""
    print("\n" + "=" * 70)
    print("Testing LMMs Engine Imports")
    print("=" * 70)

    try:
        # Test model module imports
        from lmms_engine.models.qwen3_vl_moe import apply_liger_kernel_to_qwen3_vl_moe

        print("Monkey patch function imported successfully")

        # Test processor imports
        from lmms_engine.datasets.processor.qwen3_vl_processor import (
            Qwen3_VLDataProcessor,
        )

        print("Processor class imported successfully")

        # Test parallel imports
        from lmms_engine.parallel.qwen3_vl_moe import apply_qwen3_vl_moe_parallelize_fn

        print("Parallel function imported successfully")

        return True
    except ImportError as e:
        print(f"Failed to import LMMs Engine components: {e}")
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

        if "qwen3_vl_moe" in MODEL_MAPPING:
            print("Model type 'qwen3_vl_moe' is registered")
            model_info = MODEL_MAPPING["qwen3_vl_moe"]
            print(f"   Config class: {model_info['config'].__name__}")
            print(f"   Model class: {model_info['model'].__name__}")
            return True
        else:
            print("Model type 'qwen3_vl_moe' is NOT registered")
            print(f"   Available models: {list(MODEL_MAPPING.keys())}")
            return False
    except Exception as e:
        print(f"Error checking model registration: {e}")
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

        if "qwen3_vl_moe" in DATAPROCESSOR_MAPPING:
            print("Processor 'qwen3_vl_moe' is registered")
            processor_class = DATAPROCESSOR_MAPPING["qwen3_vl"]
            print(f"   Processor class: {processor_class.__name__}")
            return True
        else:
            print("Processor 'qwen3_vl_moe' is NOT registered")
            print(f"   Available processors: {list(DATAPROCESSOR_MAPPING.keys())}")
            return False
    except Exception as e:
        print(f"Error checking processor registration: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\n" + "=" * 70)
    print("Testing File Structure")
    print("=" * 70)

    required_files = [
        "src/lmms_engine/models/qwen3_vl_moe/__init__.py",
        "src/lmms_engine/models/qwen3_vl_moe/monkey_patch.py",
        "src/lmms_engine/models/qwen3_vl_moe/qwen3_vl_moe_liger.py",
        "src/lmms_engine/parallel/qwen3_vl_moe/__init__.py",
        "test/train/qwen3_vl_moe/train_qwen3_vl_moe_ep.py",
        "test/train/qwen3_vl_moe/train_qwen3_vl_moe_sp.py",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            all_exist = False

    return all_exist


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Qwen3 VL MoE Integration Test Suite")
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
        status = "PASS" if passed else "FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed - see details above")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
