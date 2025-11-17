from typing import Any, Dict, Literal, Optional

from lmms_engine.protocol import Args


class ModelConfig(Args):
    # model_name_or_path: str
    load_from_pretrained_path: Optional[str] = None
    load_from_config: Optional[Dict[str, Any]] = None
    attn_implementation: Optional[Literal["flash_attention_2", "sdpa", "eager"]] = "sdpa"
    overwrite_config: Optional[Dict[str, str]] = None
    monkey_patch_kwargs: Optional[Dict[str, Any]] = None

    # LLaVA-Video slow-fast frame configuration
    add_faster_video: Optional[bool] = False
    faster_token_stride: Optional[int] = 10
    mm_spatial_pool_stride: Optional[int] = 2
    mm_spatial_pool_mode: Optional[Literal["average", "max", "bilinear"]] = "average"
