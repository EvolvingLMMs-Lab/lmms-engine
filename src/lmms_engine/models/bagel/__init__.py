# Adapted from https://github.com/ByteDance-Seed/Bagel/blob/main/modeling/bagel/__init__.py
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


from lmms_engine.mapping_func import register_model

from .bagel import Bagel, BagelConfig
from .qwen2_navit import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from .siglip_navit import SiglipVisionConfig, SiglipVisionModel

register_model(
    "bagel",
    BagelConfig,
    Bagel,
)

__all__ = [
    "BagelConfig",
    "Bagel",
    "Qwen2Config",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    "SiglipVisionConfig",
    "SiglipVisionModel",
]
