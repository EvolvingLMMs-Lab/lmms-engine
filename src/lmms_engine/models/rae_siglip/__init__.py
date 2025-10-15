from lmms_engine.mapping_func import register_model

from .configuration_rae_siglip import RaeSiglipConfig
from .discriminator import ProjectedDiscriminator
from .modeling_rae_siglip import RaeSiglipModel

register_model("rae_siglip", RaeSiglipConfig, RaeSiglipModel, "general")

__all__ = [
    "RaeSiglipModel",
    "RaeSiglipConfig",
    "ProjectedDiscriminator",
]
