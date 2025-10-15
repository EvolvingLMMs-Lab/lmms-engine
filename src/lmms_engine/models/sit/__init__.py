from lmms_engine.mapping_func import register_model

from .configuration_sit import SiTConfig
from .modeling_sit import SiTModel
from .models import SiT

register_model(
    "sit",
    SiTConfig,
    SiTModel,
)

__all__ = [
    "SiTModel",
    "SiTConfig",
    "SiT",
]
