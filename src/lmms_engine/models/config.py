from dataclasses import dataclass
from typing import Dict, Literal, Optional


@dataclass
class ModelConfig:
    # model_name_or_path: str
    load_from_pretrained_path: Optional[str] = None
    load_from_config: Optional[Dict[str, str]] = None
    pretrain_mm_mlp_adapter: Optional[str] = None
    vision_tower: Optional[str] = None
    mm_vision_select_layer: Optional[int] = -1
    mm_use_im_start_end: Optional[bool] = False
    mm_patch_merge_type: Optional[str] = "flat"
    mm_vision_select_feature: Optional[str] = "patch"
    image_token_format: Optional[str] = "<I{}>"
    scale_token_format: Optional[str] = "<S{}>"
    num_image_tokens: Optional[int] = -1
    num_scale_tokens: Optional[int] = 3
    attn_implementation: Optional[Literal["flash_attention_2", "sdpa"]] = "sdpa"
    overwrite_config: Optional[Dict[str, str]] = None
