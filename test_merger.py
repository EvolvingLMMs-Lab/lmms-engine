#!/usr/bin/env python3
"""Test merger on two checkpoint paths."""

from pathlib import Path

from lmms_engine.merger import FSDP2Merger

# Two test paths - they should both point to the same checkpoint
path1 = Path("/pfs/training-data/kaichenzhang/lmms-engine-mini/output/sp_ablate/qwen3_vl_4B_llava_web_fixed_loss_x5_lr")
path2 = Path(
    "/pfs/training-data/kaichenzhang/lmms-engine-mini/output/sp_ablate/qwen3_vl_4B_llava_web_fixed_loss_x5_lr/checkpoint-3500"
)

print("Testing path 1 (parent directory):", path1)
merger = FSDP2Merger(checkpoint_type="regular")
try:
    result1 = merger.merge(path1)
    print(f"✓ Path 1 merged successfully: {result1}")
except Exception as e:
    print(f"✗ Path 1 failed: {e}")

print("\nTesting path 2 (specific checkpoint):", path2)
try:
    result2 = merger.merge(path2)
    print(f"✓ Path 2 merged successfully: {result2}")
except Exception as e:
    print(f"✗ Path 2 failed: {e}")
