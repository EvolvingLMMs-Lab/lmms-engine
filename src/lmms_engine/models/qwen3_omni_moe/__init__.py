from lmms_engine.utils.import_utils import is_transformers_version_greater_or_equal_to

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

if is_transformers_version_greater_or_equal_to("5.0"):
    #patch expert weight loading to handle old checkpoint shape convention
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextExperts

    _original_experts_load_state_dict = Qwen3OmniMoeThinkerTextExperts._load_from_state_dict

    def _patched_experts_load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key_suffix in ("gate_up_proj", "down_proj"):
            key = prefix + key_suffix
            if key in state_dict:
                ckpt_shape = state_dict[key].shape
                model_shape = getattr(self, key_suffix).shape
                if len(ckpt_shape) == 3 and ckpt_shape[0] == model_shape[0] and ckpt_shape[1] == model_shape[2] and ckpt_shape[2] == model_shape[1]:
                    state_dict[key] = state_dict[key].transpose(1, 2).contiguous()
        _original_experts_load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    Qwen3OmniMoeThinkerTextExperts._load_from_state_dict = _patched_experts_load_state_dict

from lmms_engine.mapping_func import register_model  # noqa: E402

from .monkey_patch import apply_liger_kernel_to_qwen3_omni_moe  # noqa: E402

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
