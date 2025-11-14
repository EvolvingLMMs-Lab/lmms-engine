from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
)

from lmms_engine.mapping_func import register_model

from .monkey_patch import apply_liger_kernel_to_qwen3_vl_moe

register_model(
    "qwen3_vl_moe",
    Qwen3VLMoeConfig,
    Qwen3VLMoeForConditionalGeneration,
    model_general_type="causal_lm",
)

__all__ = [
    "apply_liger_kernel_to_qwen3_vl_moe",
    "Qwen3VLMoeConfig",
    "Qwen3VLMoeForConditionalGeneration",
]
