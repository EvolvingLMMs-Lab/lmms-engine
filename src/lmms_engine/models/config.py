from typing import Any, Dict, Literal, Optional

from lmms_engine.protocol import Args


class ModelConfig(Args):
    # model_name_or_path: str
    load_from_pretrained_path: Optional[str] = None
    load_from_config: Optional[Dict[str, Any]] = None
    attn_implementation: Optional[Literal["flash_attention_2", "sdpa", "eager"]] = "sdpa"
    overwrite_config: Optional[Dict[str, Any]] = None
    monkey_patch_kwargs: Optional[Dict[str, Any]] = None
