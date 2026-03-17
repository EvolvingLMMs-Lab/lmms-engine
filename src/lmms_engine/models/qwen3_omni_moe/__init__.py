from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from lmms_engine.mapping_func import register_model

from .monkey_patch import apply_liger_kernel_to_qwen3_omni_moe

register_model(
    "qwen3_omni_moe_thinker",
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    model_general_type="causal_lm",
)

__all__ = [
    "apply_liger_kernel_to_qwen3_omni_moe",
    "Qwen3OmniMoeThinkerConfig",
    "Qwen3OmniMoeThinkerForConditionalGeneration",
]
