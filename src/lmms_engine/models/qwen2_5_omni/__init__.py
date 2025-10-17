from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from lmms_engine.mapping_func import register_model

from .monkey_patch import apply_liger_kernel_to_qwen2_5_omni

register_model(
    "qwen2_5_omni_thinker",
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
    model_general_type="causal_lm",
)

__all__ = [
    "apply_liger_kernel_to_qwen2_5_omni",
    "Qwen2_5OmniThinkerConfig",
    "Qwen2_5OmniThinkerForConditionalGeneration",
]
