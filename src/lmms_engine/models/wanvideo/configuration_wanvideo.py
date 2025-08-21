# coding=utf-8
# Copyright 2024 WanVideo team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from transformers.configuration_utils import PretrainedConfig


class WanVideoConfig(PretrainedConfig):
    model_type = "wanvideo"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        # DiT model parameters
        dit_hidden_size: int = 3072,
        dit_num_layers: int = 42,
        dit_num_heads: int = 24,
        dit_patch_size: int = 2,
        dit_patch_size_t: int = 1,
        dit_in_channels: int = 16,
        dit_mlp_ratio: float = 4.0,
        dit_qk_norm: bool = True,
        dit_enable_flash_attn: bool = True,
        dit_rope_scaling_factor: float = 2.0,
        dit_temporal_rope_scaling_factor: float = 2.0,
        # VAE parameters
        vae_in_channels: int = 3,
        vae_out_channels: int = 3,
        vae_latent_channels: int = 16,
        vae_block_out_channels: List[int] = None,
        vae_layers_per_block: int = 2,
        vae_scaling_factor: float = 0.33208,
        # Text encoder parameters
        text_encoder_model: str = "umt5-xxl-enc",
        text_encoder_hidden_size: int = 4096,
        text_encoder_intermediate_size: int = 10240,
        text_encoder_num_layers: int = 24,
        text_encoder_num_heads: int = 64,
        text_encoder_head_dim: int = 64,
        max_text_length: int = 256,
        # Image encoder parameters (for I2V)
        image_encoder_model: Optional[str] = "clip-vit-large-patch14",
        image_encoder_hidden_size: Optional[int] = 768,
        use_image_encoder: bool = False,
        # Training parameters
        num_train_timesteps: int = 1000,
        scheduler_type: str = "flow_match",
        scheduler_shift: int = 5,
        scheduler_sigma_min: float = 0.0,
        # gradient_checkpointing: bool = False,
        use_lora: bool = False,
        lora_rank: int = 32,
        lora_target_modules: List[str] = None,
        # Generation parameters
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        fps: int = 15,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 20,
        # Model variants
        # model_variant: str = "Wan2.1-T2V-1.3B",  # T2V, I2V, VACE, Fun, etc.
        # model_size: str = "1.3B",  # 1.3B, 5B, 14B
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        # DiT configuration
        self.dit_hidden_size = dit_hidden_size
        self.dit_num_layers = dit_num_layers
        self.dit_num_heads = dit_num_heads
        self.dit_patch_size = dit_patch_size
        self.dit_patch_size_t = dit_patch_size_t
        self.dit_in_channels = dit_in_channels
        self.dit_mlp_ratio = dit_mlp_ratio
        self.dit_qk_norm = dit_qk_norm
        self.dit_enable_flash_attn = dit_enable_flash_attn
        self.dit_rope_scaling_factor = dit_rope_scaling_factor
        self.dit_temporal_rope_scaling_factor = dit_temporal_rope_scaling_factor

        # VAE configuration
        self.vae_in_channels = vae_in_channels
        self.vae_out_channels = vae_out_channels
        self.vae_latent_channels = vae_latent_channels
        self.vae_block_out_channels = vae_block_out_channels or [128, 256, 512, 512]
        self.vae_layers_per_block = vae_layers_per_block
        self.vae_scaling_factor = vae_scaling_factor

        # Text encoder configuration
        self.text_encoder_model = text_encoder_model
        self.text_encoder_hidden_size = text_encoder_hidden_size
        self.text_encoder_intermediate_size = text_encoder_intermediate_size
        self.text_encoder_num_layers = text_encoder_num_layers
        self.text_encoder_num_heads = text_encoder_num_heads
        self.text_encoder_head_dim = text_encoder_head_dim
        self.max_text_length = max_text_length

        # Image encoder configuration
        self.image_encoder_model = image_encoder_model
        self.image_encoder_hidden_size = image_encoder_hidden_size
        self.use_image_encoder = use_image_encoder

        # Training configuration
        self.num_train_timesteps = num_train_timesteps
        self.scheduler_type = scheduler_type
        self.scheduler_shift = scheduler_shift
        self.scheduler_sigma_min = scheduler_sigma_min
        # self.gradient_checkpointing = gradient_checkpointing
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_target_modules = lora_target_modules or [
            "q",
            "k",
            "v",
            "o",
            "ffn.0",
            "ffn.2",
        ]

        # Generation configuration
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        # Model variants
        # self.model_variant = model_variant
        # self.model_size = model_size
        # self.tie_word_embeddings = tie_word_embeddings

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    # def get_model_size_config(self):
    #     """Get model size specific configurations"""
    #     size_configs = {
    #         "1.3B": {
    #             "dit_hidden_size": 2432,
    #             "dit_num_layers": 28,
    #             "dit_num_heads": 19,
    #         },
    #         "5B": {
    #             "dit_hidden_size": 3840,
    #             "dit_num_layers": 42,
    #             "dit_num_heads": 30,
    #         },
    #         "14B": {
    #             "dit_hidden_size": 5120,
    #             "dit_num_layers": 48,
    #             "dit_num_heads": 40,
    #         },
    #     }
    #     return size_configs.get(self.model_size, {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        output = super().to_dict()
        return output
