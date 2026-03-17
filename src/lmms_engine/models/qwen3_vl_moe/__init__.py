from lmms_engine.utils.import_utils import is_transformers_version_greater_or_equal_to

_IS_TRANSFORMERS_5 = is_transformers_version_greater_or_equal_to("5.0")

from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)

if _IS_TRANSFORMERS_5:
    #patch missing pad_token_id for transformers 5.0 compatibility
    _original_qwen3_vl_moe_text_config_init = Qwen3VLMoeTextConfig.__init__

    def _patched_qwen3_vl_moe_text_config_init(self, *args, **kwargs):
        kwargs.setdefault("pad_token_id", None)
        _original_qwen3_vl_moe_text_config_init(self, *args, **kwargs)

    Qwen3VLMoeTextConfig.__init__ = _patched_qwen3_vl_moe_text_config_init

    #patch expert weight loading to handle old checkpoint shape convention
    #old checkpoints store [E, H, 2I] / [E, I, H], new transformers 5.0 expects [E, 2I, H] / [E, H, I]
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

    _original_experts_load_state_dict = Qwen3VLMoeTextExperts._load_from_state_dict

    def _patched_experts_load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key_suffix in ("gate_up_proj", "down_proj"):
            key = prefix + key_suffix
            if key in state_dict:
                ckpt_shape = state_dict[key].shape
                model_shape = getattr(self, key_suffix).shape
                if len(ckpt_shape) == 3 and ckpt_shape[0] == model_shape[0] and ckpt_shape[1] == model_shape[2] and ckpt_shape[2] == model_shape[1]:
                    state_dict[key] = state_dict[key].transpose(1, 2).contiguous()
        _original_experts_load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    Qwen3VLMoeTextExperts._load_from_state_dict = _patched_experts_load_state_dict

from .monkey_patch import apply_liger_kernel_to_qwen3_vl_moe  # noqa: E402

__all__ = ["apply_liger_kernel_to_qwen3_vl_moe"]
