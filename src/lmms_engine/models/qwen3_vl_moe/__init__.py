from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)

#patch missing pad_token_id for transformers 5.0 compatibility
_original_qwen3_vl_moe_text_config_init = Qwen3VLMoeTextConfig.__init__


def _patched_qwen3_vl_moe_text_config_init(self, *args, **kwargs):
    kwargs.setdefault("pad_token_id", None)
    _original_qwen3_vl_moe_text_config_init(self, *args, **kwargs)


Qwen3VLMoeTextConfig.__init__ = _patched_qwen3_vl_moe_text_config_init

from .monkey_patch import apply_liger_kernel_to_qwen3_vl_moe  # noqa: E402

__all__ = ["apply_liger_kernel_to_qwen3_vl_moe"]
