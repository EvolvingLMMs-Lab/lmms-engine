# from typing import TYPE_CHECKING

# from ...utils import _LazyModule
# from ...utils.import_utils import define_import_structure


# if TYPE_CHECKING:
#     from .configuration_qwen3_dllm import *
#     from .modeling_qwen3_dllm import *
# else:
#     import sys

#     _file = globals()["__file__"]
#     sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)


from lmms_engine.mapping_func import register_model

from .configuration_llada_dllm import LLaDADLLMConfig
from .modeling_llada_dllm import LLaDADLLMForMaskedLM

register_model(
    "dream_dllm",
    DreamDLLMConfig,
    DreamDLLMForMaskedLM,
    model_general_type="masked_lm",
)

__all__ = ["DreamDLLMConfig", "DreamDLLMForMaskedLM"]
