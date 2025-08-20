from lmms_engine.mapping_func import register_model

from .blip3o_qwen import Blip3oQwenConfig, Blip3oQwenForCausalLM

register_model(
    "blip3o_qwen",
    Blip3oQwenConfig,
    Blip3oQwenForCausalLM,
)

__all__ = ["Blip3oQwenConfig", "Blip3oQwenForCausalLM"]
