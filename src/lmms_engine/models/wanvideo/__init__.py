from lmms_engine.mapping_func import register_model

from .configuration_wanvideo import WanVideoConfig
from .modeling_wanvideo import (
    WanVideoForConditionalGeneration,
    WanVideoOutput,
    WanVideoPreTrainedModel,
)
from .processing_wanvideo import WanVideoImageProcessor, WanVideoProcessor

register_model(
    "wanvideo",
    WanVideoConfig,
    WanVideoForConditionalGeneration,
    model_general_type="general",
)

__all__ = [
    "WanVideoConfig",
    "WanVideoForConditionalGeneration",
    "WanVideoPreTrainedModel",
    "WanVideoOutput",
    "WanVideoProcessor",
    "WanVideoImageProcessor",
]
