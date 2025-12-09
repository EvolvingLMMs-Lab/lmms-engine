from .configuration_nit import NitConfig
from .modeling_nit import NitModel

from lmms_engine.mapping_func import register_model

register_model(
    "nit",
    NitConfig,
    NitModel,
)

__all__ = [
    "NitConfig",
    "NitModel",
]
