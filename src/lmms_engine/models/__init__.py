from lmms_engine.utils.import_utils import is_transformers_version_greater_or_equal_to

is_transformers_5 = is_transformers_version_greater_or_equal_to("5.0.0")

from .aero import AeroConfig, AeroForConditionalGeneration, AeroProcessor
from .bagel import Bagel, BagelConfig
from .config import ModelConfig
from .llava_onevision import apply_liger_kernel_to_llava_onevision
from .llava_onevision2 import apply_liger_kernel_to_llava_onevision2
from .monkey_patch import MONKEY_PATCHER
from .nanovlm import NanovlmConfig, NanovlmForConditionalGeneration
from .qwen2 import apply_liger_kernel_to_qwen2
from .qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
    apply_liger_kernel_to_qwen2_5_omni,
)
from .qwen2_5_vl import apply_liger_kernel_to_qwen2_5_vl
from .qwen2_audio import apply_liger_kernel_to_qwen2_audio
from .qwen3 import apply_liger_kernel_to_qwen3
from .qwen3_5 import apply_liger_kernel_to_qwen3_5
from .qwen3_moe import apply_liger_kernel_to_qwen3_moe
from .qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    apply_liger_kernel_to_qwen3_omni_moe,
)
from .qwen3_vl import apply_liger_kernel_to_qwen3_vl
from .qwen3_vl_moe import apply_liger_kernel_to_qwen3_vl_moe
from .rae_siglip import RaeSiglipConfig, RaeSiglipModel
from .sit import SiT, SiTConfig, SiTModel
from .wanvideo import (
    WanVideoConfig,
    WanVideoForConditionalGeneration,
    WanVideoProcessor,
)

__all__ = [
    "AeroForConditionalGeneration",
    "AeroConfig",
    "Bagel",
    "BagelConfig",
    "ModelConfig",
    "AeroProcessor",
    "apply_liger_kernel_to_llava_onevision",
    "apply_liger_kernel_to_llava_onevision2",
    "apply_liger_kernel_to_qwen2",
    "apply_liger_kernel_to_qwen3",
    "Qwen2_5OmniThinkerConfig",
    "Qwen2_5OmniThinkerForConditionalGeneration",
    "apply_liger_kernel_to_qwen2_5_omni",
    "apply_liger_kernel_to_qwen2_5_vl",
    "apply_liger_kernel_to_qwen2_audio",
    "apply_liger_kernel_to_qwen3_5",
    "apply_liger_kernel_to_qwen3_vl",
    "apply_liger_kernel_to_qwen3_vl_moe",
    "apply_liger_kernel_to_qwen3_moe",
    "Qwen3OmniMoeThinkerConfig",
    "Qwen3OmniMoeThinkerForConditionalGeneration",
    "apply_liger_kernel_to_qwen3_omni_moe",
    "WanVideoConfig",
    "WanVideoForConditionalGeneration",
    "WanVideoProcessor",
    "MONKEY_PATCHER",
    "NanovlmConfig",
    "NanovlmForConditionalGeneration",
    "RaeSiglipConfig",
    "RaeSiglipModel",
    "SiTModel",
    "SiTConfig",
    "SiT",
]

if not is_transformers_5:
    from .dream_dllm import DreamDLLMConfig, DreamDLLMForMaskedLM
    from .llada_dllm import LLaDADLLMConfig, LLaDADLLMForMaskedLM
    from .llava_onevision1_5 import (
        LLaVAOneVision1_5_ForConditionalGeneration,
        Llavaonevision1_5Config,
        apply_liger_kernel_to_llava_onevision1_5,
    )
    from .qwen3_dllm import Qwen3DLLMConfig, Qwen3DLLMForMaskedLM

    __all__.extend(
        [
            "Qwen3DLLMConfig",
            "Qwen3DLLMForMaskedLM",
            "DreamDLLMConfig",
            "DreamDLLMForMaskedLM",
            "LLaDADLLMConfig",
            "LLaDADLLMForMaskedLM",
            "Llavaonevision1_5Config",
            "LLaVAOneVision1_5_ForConditionalGeneration",
            "apply_liger_kernel_to_llava_onevision1_5",
        ]
    )
