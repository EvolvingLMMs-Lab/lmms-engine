#!/usr/bin/env python3
"""
Verification script for SiT training setup.
Checks if all dependencies and components are properly installed.
"""

import sys


def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        package = package_name or module_name
        print(f"✗ {module_name} - Install with: pip install {package}")
        return False


def check_sit_components():
    """Check if SiT-specific components are available."""
    try:
        from lmms_engine.datasets.naive.sit_dataset import SitDataset
        from lmms_engine.datasets.processor.sit_processor import SitDataProcessor
        from lmms_engine.models.sit import SiTConfig, SiTModel
        from lmms_engine.train.fsdp2.sit_trainer import SitTrainer

        print("✓ All SiT components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ SiT components - {str(e)}")
        return False


def main():
    print("=" * 60)
    print("SiT Training Setup Verification")
    print("=" * 60)

    all_ok = True

    # Check core dependencies
    print("\n[Core Dependencies]")
    all_ok &= check_import("torch")
    all_ok &= check_import("torchvision")
    all_ok &= check_import("transformers")
    all_ok &= check_import("datasets")
    all_ok &= check_import("yaml", "pyyaml")

    # Check SiT-specific dependencies
    print("\n[SiT Dependencies]")
    all_ok &= check_import("timm")
    all_ok &= check_import("diffusers")
    all_ok &= check_import("torchdiffeq")

    # Check optional performance dependencies
    print("\n[Optional Performance Dependencies]")
    flash_attn_ok = check_import("flash_attn")
    liger_kernel_ok = check_import("liger_kernel")

    # Check SiT components
    print("\n[SiT Components]")
    all_ok &= check_sit_components()

    # Check CUDA availability
    print("\n[Hardware]")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA available - {torch.cuda.device_count()} GPU(s)")
            print(f"  - GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ CUDA not available")
            all_ok = False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        all_ok = False

    # Final verdict
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All required dependencies are installed!")
        print("✓ Ready to train SiT models")
        if not (flash_attn_ok and liger_kernel_ok):
            print("\nNote: For optimal performance, consider installing:")
            if not flash_attn_ok:
                print("  - flash-attn: pip install flash-attn --no-build-isolation")
            if not liger_kernel_ok:
                print("  - liger-kernel: pip install liger-kernel")
        return 0
    else:
        print("✗ Some dependencies are missing")
        print("\nTo install all SiT dependencies:")
        print("  pip install lmms_engine[sit]")
        print("or:")
        print('  uv pip install -e ".[sit]"')
        return 1


if __name__ == "__main__":
    sys.exit(main())
