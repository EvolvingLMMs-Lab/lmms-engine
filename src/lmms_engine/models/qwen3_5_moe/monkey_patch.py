"""Monkey patches for qwen3_5_moe (OV2-style split: liger + rmpad independent)."""

from lmms_engine.models.monkey_patch import MONKEY_PATCHER


@MONKEY_PATCHER.register("qwen3_5_moe", "liger")
def apply_liger_kernel_to_qwen3_5_moe(model=None, **kwargs):
    raise NotImplementedError("filled in Task 4")


@MONKEY_PATCHER.register("qwen3_5_moe", "rmpad")
def apply_rmpad_to_qwen3_5_moe(model=None):
    raise NotImplementedError("filled in Task 4")
