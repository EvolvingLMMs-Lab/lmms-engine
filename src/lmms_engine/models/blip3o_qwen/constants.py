from dataclasses import dataclass


@dataclass
class Blip3oConstants:
    IGNORE_INDEX: int = -100
    IMAGE_TOKEN_INDEX: int = -200
    DEFAULT_IMAGE_TOKEN: str = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN: str = "<im_patch>"
    DEFAULT_IM_START_TOKEN: str = "<im_start>"
    DEFAULT_IM_END_TOKEN: str = "<im_end>"
    DIFFUSION_CONNECTOR_HIDDEN_SIZE: int = 2304
