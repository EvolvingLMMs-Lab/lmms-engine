from typing import Any, Dict, Literal, Optional

from lmms_engine.protocol import Args


class ModelConfig(Args):
    # model_name_or_path: str
    load_from_pretrained_path: Optional[str] = None
    load_from_config: Optional[Dict[str, Any]] = None
    attn_implementation: Optional[Literal["flash_attention_2", "sdpa", "eager"]] = "sdpa"
    # Force a specific Auto* class when the model's config is registered to
    # more than one mapping (e.g. Qwen3.5, where the same config picks up
    # both causal_lm -> text-only and image_text_to_text -> VL). Defaults to
    # None (auto-detect by config type). Keys match
    # ``mapping_func.AUTO_REGISTER_MODEL_MAPPING``.
    model_general_type: Optional[Literal["causal_lm", "masked_lm", "image_text_to_text", "general"]] = None
    overwrite_config: Optional[Dict[str, Any]] = None
    monkey_patch_kwargs: Optional[Dict[str, Any]] = None
