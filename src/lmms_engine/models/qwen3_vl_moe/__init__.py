from lmms_engine.utils.import_utils import is_transformers_version_greater_or_equal_to

_IS_TRANSFORMERS_5 = is_transformers_version_greater_or_equal_to("5.0")

from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)

if _IS_TRANSFORMERS_5:
    # patch missing pad_token_id for transformers 5.0 compatibility
    _original_qwen3_vl_moe_text_config_init = Qwen3VLMoeTextConfig.__init__

    def _patched_qwen3_vl_moe_text_config_init(self, *args, **kwargs):
        kwargs.setdefault("pad_token_id", None)
        _original_qwen3_vl_moe_text_config_init(self, *args, **kwargs)

    Qwen3VLMoeTextConfig.__init__ = _patched_qwen3_vl_moe_text_config_init

    # patch Experts.__init__ to match checkpoint shape convention
    # checkpoint stores [E, H, 2I] / [E, I, H], but transformers 5.0 creates [E, 2I, H] / [E, H, I]
    # we patch __init__ to use the checkpoint convention so loading works directly
    import torch
    import torch.nn as nn
    from transformers.activations import ACT2FN
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeTextExperts,
    )

    def _patched_experts_init(self, config):
        nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, 2 * self.intermediate_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    Qwen3VLMoeTextExperts.__init__ = _patched_experts_init

from .monkey_patch import apply_liger_kernel_to_qwen3_vl_moe  # noqa: E402

__all__ = ["apply_liger_kernel_to_qwen3_vl_moe"]
