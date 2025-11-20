from .language_model.blip3o_qwen import blip3oQwenConfig, blip3oQwenForCausalLM
from .language_model.blip3o_qwen_inference import blip3oQwenForInferenceLM
from .language_model.blip3o_qwen_grpo import blip3oQwenForGRPOLM
from lmms_engine.mapping_func import register_model
# from ..utils import sde_step_with_logprob, sde_step_with_logprob_simple

register_model(
    "blip3o_qwen",
    blip3oQwenConfig,
    blip3oQwenForCausalLM,
)

__all__ = [
    "blip3oQwenConfig",
    "blip3oQwenForCausalLM",
    "blip3oQwenForInferenceLM",
    "blip3oQwenForGRPOLM",
    # "sde_step_with_logprob",
    # "sde_step_with_logprob_simple",
]