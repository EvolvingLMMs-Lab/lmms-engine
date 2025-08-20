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

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    TransformersKwargs,
    can_return_tuple,
    logging,
)
from transformers.utils.generic import check_model_inputs

from .configuration_wanvideo import WanVideoConfig

logger = logging.get_logger(__name__)

# Try to import flash attention
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.info("Flash Attention not available, using standard attention")


@dataclass
class WanVideoOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    noise_pred: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    text_embeddings: Optional[torch.FloatTensor] = None
    image_embeddings: Optional[torch.FloatTensor] = None
    latents: Optional[torch.FloatTensor] = None


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(dtype)
        return x * self.weight


# def eager_attention_forward(
#     module: nn.Module,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     attention_mask: Optional[torch.Tensor],
#     scaling: float,
#     dropout: float = 0.0,
#     **kwargs: Unpack[TransformersKwargs],
# ):
#     key_states = repeat_kv(key, module.num_key_value_groups)
#     value_states = repeat_kv(value, module.num_key_value_groups)

#     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
#     if attention_mask is not None:
#         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#         attn_weights = attn_weights + causal_mask

#     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
#         query.dtype
#     )
#     attn_weights = nn.functional.dropout(
#         attn_weights, p=dropout, training=module.training
#     )
#     attn_output = torch.matmul(attn_weights, value_states)
#     attn_output = attn_output.transpose(1, 2).contiguous()

#     return attn_output, attn_weights


# class Qwen3DLLMAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config: Qwen3DLLMConfig, layer_idx: int):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         self.head_dim = getattr(
#             config, "head_dim", config.hidden_size // config.num_attention_heads
#         )
#         self.num_key_value_groups = (
#             config.num_attention_heads // config.num_key_value_heads
#         )
#         self.scaling = self.head_dim**-0.5
#         self.attention_dropout = config.attention_dropout
#         self.is_causal = False

#         self.q_proj = nn.Linear(
#             config.hidden_size,
#             config.num_attention_heads * self.head_dim,
#             bias=config.attention_bias,
#         )
#         self.k_proj = nn.Linear(
#             config.hidden_size,
#             config.num_key_value_heads * self.head_dim,
#             bias=config.attention_bias,
#         )
#         self.v_proj = nn.Linear(
#             config.hidden_size,
#             config.num_key_value_heads * self.head_dim,
#             bias=config.attention_bias,
#         )
#         self.o_proj = nn.Linear(
#             config.num_attention_heads * self.head_dim,
#             config.hidden_size,
#             bias=config.attention_bias,
#         )
#         self.q_norm = Qwen3DLLMRMSNorm(
#             self.head_dim, eps=config.rms_norm_eps
#         )  # unlike olmo, only on the head dim!
#         self.k_norm = Qwen3DLLMRMSNorm(
#             self.head_dim, eps=config.rms_norm_eps
#         )  # thus post q_norm does not need reshape

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_embeddings: tuple[torch.Tensor, torch.Tensor],
#         attention_mask: Optional[torch.Tensor],
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
#         input_shape = hidden_states.shape[:-1]
#         hidden_shape = (*input_shape, -1, self.head_dim)

#         query_states = self.q_norm(
#             self.q_proj(hidden_states).view(hidden_shape)
#         ).transpose(1, 2)
#         key_states = self.k_norm(
#             self.k_proj(hidden_states).view(hidden_shape)
#         ).transpose(1, 2)
#         value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(
#             query_states, key_states, cos, sin
#         )

#         attention_interface: Callable = eager_attention_forward
#         if self.config._attn_implementation != "eager":
#             attention_interface = ALL_ATTENTION_FUNCTIONS[
#                 self.config._attn_implementation
#             ]

#         attn_output, attn_weights = attention_interface(
#             self,
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             dropout=0.0 if not self.training else self.attention_dropout,
#             scaling=self.scaling,
#             sliding_window=None,
#             **kwargs,
#         )

#         attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#         attn_output = self.o_proj(attn_output)
#         return attn_output, attn_weights


class WanAttention(nn.Module):
    def __init__(self, config: WanVideoConfig, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        if config.dit_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.use_flash_attn = config.dit_enable_flash_attn and FLASH_ATTN_AVAILABLE

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)

        # Apply RoPE if frequency embeddings provided
        if freqs_cis is not None:
            # Apply RoPE here if needed
            pass

        # Apply normalization if configured
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply attention
        if self.use_flash_attn:
            attn_output = flash_attn_func(q, k, v)
            attn_output = rearrange(attn_output, "b s n d -> b s (n d)")
        else:
            q = rearrange(q, "b s n d -> b n s d")
            k = rearrange(k, "b s n d -> b n s d")
            v = rearrange(v, "b s n d -> b n s d")
            attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask)
            attn_output = rearrange(attn_output, "b n s d -> b s (n d)")

        attn_output = self.o_proj(attn_output)
        return attn_output


class WanMLP(nn.Module):
    def __init__(self, config: WanVideoConfig, hidden_size: int):
        super().__init__()
        mlp_hidden_size = int(hidden_size * config.dit_mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class WanDiTBlock(GradientCheckpointingLayer):
    def __init__(self, config: WanVideoConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # self.gradient_checkpointing = False
        hidden_size = config.dit_hidden_size

        self.norm1 = RMSNorm(hidden_size)
        self.attn = WanAttention(config, hidden_size, config.dit_num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = WanMLP(config, hidden_size)

        # Adaptive layer norm for conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=False),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get modulation parameters from timestep and text embeddings
        if text_emb is not None:
            conditioning = timestep_emb + text_emb
        else:
            conditioning = timestep_emb

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(conditioning).chunk(6, dim=-1)

        # Self-attention with adaptive layer norm
        normed = self.norm1(hidden_states)
        normed = modulate(normed, shift_msa, scale_msa)
        attn_output = self.attn(normed, freqs_cis=freqs_cis)
        hidden_states = hidden_states + gate_msa * attn_output

        # MLP with adaptive layer norm
        normed = self.norm2(hidden_states)
        normed = modulate(normed, shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed)
        hidden_states = hidden_states + gate_mlp * mlp_output

        return hidden_states


class WanDiT(nn.Module):
    """Diffusion Transformer for WanVideo"""

    def __init__(self, config: WanVideoConfig):
        super().__init__()
        self.config = config
        # self.gradient_checkpointing = False

        # Patch embedding
        self.patch_embed = nn.Conv3d(
            config.dit_in_channels,
            config.dit_hidden_size,
            kernel_size=(
                config.dit_patch_size_t,
                config.dit_patch_size,
                config.dit_patch_size,
            ),
            stride=(
                config.dit_patch_size_t,
                config.dit_patch_size,
                config.dit_patch_size,
            ),
        )

        # Position embedding will be computed dynamically
        self.pos_embed_type = "rope"  # Can be "rope" or "sinusoidal"

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size * 4, config.dit_hidden_size),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [WanDiTBlock(config, idx) for idx in range(config.dit_num_layers)]
        )

        # Text projection layer
        if config.text_encoder_hidden_size != config.dit_hidden_size:
            self.text_proj = nn.Linear(
                config.text_encoder_hidden_size,
                config.dit_hidden_size,
                bias=False,
            )
        else:
            self.text_proj = None

        # Final layer
        self.norm_out = RMSNorm(config.dit_hidden_size)
        self.proj_out = nn.Linear(
            config.dit_hidden_size,
            config.dit_patch_size_t
            * config.dit_patch_size
            * config.dit_patch_size
            * config.dit_in_channels,
            bias=False,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv3d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def unpatchify(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """
        x: (B, N, patch_size_t * patch_size * patch_size * C)
        return: (B, C, T, H, W)
        """
        c = self.config.dit_in_channels
        pt = self.config.dit_patch_size_t
        p = self.config.dit_patch_size

        h = H // p
        w = W // p
        t = T // pt

        x = x.reshape(x.shape[0], t, h, w, pt, p, p, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # (B, C, t, pt, h, p, w, p)
        x = x.reshape(x.shape[0], c, T, H, W)
        return x

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        # use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        B, C, T, H, W = latents.shape

        # Patch embedding
        x = self.patch_embed(latents)
        x = rearrange(x, "b c t h w -> b (t h w) c")

        # Get timestep embedding
        t_emb = sinusoidal_embedding_1d(self.config.dit_hidden_size, timestep)
        t_emb = t_emb.to(dtype=latents.dtype, device=latents.device)
        t_emb = self.time_embed(t_emb)

        # Add text/image conditioning if available
        cond_emb = t_emb
        if text_embeddings is not None:
            # Pool or project text embeddings to match hidden size
            if text_embeddings.dim() == 3:
                # (B, seq_len, hidden_dim) -> (B, hidden_dim)
                text_embeddings = text_embeddings.mean(dim=1)
            if self.text_proj is not None:
                text_embeddings = self.text_proj(text_embeddings)
            cond_emb = cond_emb + text_embeddings

        if image_embeddings is not None:
            # Add image embeddings for I2V
            cond_emb = cond_emb + image_embeddings
        for block in self.blocks:
            x = block(x, cond_emb)

        # Final projection
        x = self.norm_out(x)
        x = self.proj_out(x)

        # Unpatchify
        x = self.unpatchify(x, T, H, W)

        return x


class WanVideoPreTrainedModel(PreTrainedModel):
    config_class = WanVideoConfig
    base_model_prefix = "wanvideo"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WanDiTBlock"]
    _supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(
        self, enable: bool = True, gradient_checkpointing_func=None
    ):
        if enable:
            self.gradient_checkpointing = True
            # Set gradient checkpointing for DiT module
            if hasattr(self, "dit"):
                self.dit.gradient_checkpointing = True
        else:
            self.gradient_checkpointing = False
            if hasattr(self, "dit"):
                self.dit.gradient_checkpointing = False


# @register_model("wanvideo", WanVideoConfig, WanVideoPreTrainedModel)
class WanVideoForConditionalGeneration(WanVideoPreTrainedModel):
    def __init__(self, config: WanVideoConfig):
        super().__init__(config)
        self.config = config

        # Main DiT model
        self.dit = WanDiT(config)

        # Placeholder for VAE - typically loaded separately
        self.vae = None
        self.text_encoder = None
        self.image_encoder = None

        # Scheduler placeholder
        self.scheduler = None
        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return None  # No traditional input embeddings

    def set_input_embeddings(self, value):
        pass  # No traditional input embeddings

    # def encode_prompt(
    #     self,
    #     prompt: Union[str, List[str]],
    #     device: Optional[torch.device] = None,
    # ) -> torch.Tensor:
    #     """Encode text prompt to embeddings"""
    #     # This would use the text encoder when available
    #     # For now, return dummy embeddings
    #     if isinstance(prompt, str):
    #         prompt = [prompt]
    #     batch_size = len(prompt)

    #     # Dummy text embeddings
    #     text_embeddings = torch.randn(
    #         batch_size,
    #         self.config.max_text_length,
    #         self.config.text_encoder_hidden_size,
    #         device=device or self.device,
    #         dtype=self.dtype,
    #     )
    #     return text_embeddings

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to latents using VAE"""
        if self.vae is None:
            # Return dummy latents if VAE not loaded
            B, T, C, H, W = video.shape
            latents = torch.randn(
                B,
                self.config.dit_in_channels,
                T // 4,  # Temporal compression
                H // 8,  # Spatial compression
                W // 8,
                device=video.device,
                dtype=video.dtype,
            )
            return latents
        # Use actual VAE encoding when available
        return self.vae.encode(video)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video using VAE"""
        if self.vae is None:
            # Return dummy video if VAE not loaded
            B, C, T, H, W = latents.shape
            video = torch.randn(
                B,
                T * 4,  # Temporal upsampling
                3,
                H * 8,  # Spatial upsampling
                W * 8,
                device=latents.device,
                dtype=latents.dtype,
            )
            return video
        # Use actual VAE decoding when available
        return self.vae.decode(latents)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        timesteps: Optional[torch.LongTensor] = None,
        latents: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        # use_gradient_checkpointing: Optional[bool] = None,
    ) -> Union[Tuple, WanVideoOutput]:
        """
        Forward pass for training or inference.

        Args:
            pixel_values: Input video frames (B, T, C, H, W) or (B, C, T, H, W)
            input_ids: Text input IDs for prompt encoding
            attention_mask: Attention mask for text inputs
            prompt: Text prompt(s) as string(s)
            timesteps: Diffusion timesteps
            latents: Pre-computed latents (optional)
            noise: Noise for training (optional)
            # labels: Target for training (optional)
            return_dict: Whether to return ModelOutput
            # use_gradient_checkpointing: Whether to use gradient checkpointing
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # use_gradient_checkpointing = (
        #     use_gradient_checkpointing
        #     if use_gradient_checkpointing is not None
        #     else self.config.gradient_checkpointing
        # )

        # Encode video to latents if needed
        if latents is None and pixel_values is not None:
            # Ensure correct shape (B, T, C, H, W)
            if pixel_values.dim() == 5 and pixel_values.shape[2] != 3:
                # (B, T, H, W, C) -> (B, T, C, H, W)
                pixel_values = pixel_values.permute(0, 1, 4, 2, 3)
            latents = self.encode_video(pixel_values)  # B, C, T, H, W

        # Get text embeddings
        text_embeddings = None
        if prompt is not None:
            # text_embeddings = self.encode_prompt(
            #     prompt, device=latents.device if latents is not None else None
            # )
            pass
        elif input_ids is not None and self.text_encoder is not None:
            # Use text encoder if available

            text_embeddings = self.text_encoder(
                input_ids, attention_mask=attention_mask
            )[0]

        # Training mode
        # (B, T, C, H, W)
        if self.training:
            if timesteps is None:
                # Sample random timesteps for training
                batch_size = latents.shape[0] if latents is not None else 1
                timesteps = torch.randint(
                    0,
                    self.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device if latents is not None else self.device,
                )  # B,

            # Add noise to latents for training
            if noise is None:
                noise = torch.randn_like(latents)

            # Simple noise scheduling (can be improved with proper scheduler)
            noise_level = timesteps.float() / self.config.num_train_timesteps
            noise_level = noise_level.view(-1, 1, 1, 1, 1)
            noisy_latents = latents * (1 - noise_level) + noise * noise_level

            # Predict noise
            # print(noisy_latents.shape, text_embeddings.shape)
            noise_pred = self.dit(
                noisy_latents,
                timesteps,
                text_embeddings=text_embeddings,
                # use_gradient_checkpointing=use_gradient_checkpointing,
            )

            # Compute loss
            # if labels is not None:
            #     target = labels
            # else:
            #     target = noise

            loss = F.mse_loss(noise_pred, noise)

            if not return_dict:
                return (loss, noise_pred)

            return WanVideoOutput(
                loss=loss,
                noise_pred=noise_pred,
                latents=latents,
                text_embeddings=text_embeddings,
            )

        # Inference mode
        else:
            if timesteps is None:
                timesteps = torch.zeros(
                    (1,),
                    device=latents.device if latents is not None else self.device,
                )

            # Generate noise prediction
            noise_pred = self.dit(
                latents,
                timesteps,
                text_embeddings=text_embeddings,
                # use_gradient_checkpointing=use_gradient_checkpointing,
            )

            if not return_dict:
                return (noise_pred,)

            return WanVideoOutput(
                noise_pred=noise_pred,
                latents=latents,
                text_embeddings=text_embeddings,
            )

    def generate(
        self,
        prompt: Union[str, List[str]],
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate video from text prompt.

        Args:
            prompt: Text prompt(s)
            num_frames: Number of frames to generate
            height: Video height
            width: Video width
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for CFG
            generator: Random generator for reproducibility

        Returns:
            Generated video tensor
        """
        # Use defaults from config if not provided
        num_frames = num_frames or self.config.num_frames
        height = height or self.config.height
        width = width or self.config.width
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale

        # Encode prompt
        text_embeddings = self.encode_prompt(prompt, device=self.device)
        batch_size = text_embeddings.shape[0]

        # Initialize latents
        latent_height = height // 8
        latent_width = width // 8
        latent_frames = num_frames // 4

        latents = torch.randn(
            batch_size,
            self.config.dit_in_channels,
            latent_frames,
            latent_height,
            latent_width,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Simple denoising loop (placeholder - needs proper scheduler)
        for i in range(num_inference_steps):
            t = torch.tensor([i], device=self.device)

            # Predict noise
            noise_pred = self.dit(latents, t, text_embeddings=text_embeddings)

            # Simple denoising step (needs proper scheduler)
            latents = latents - noise_pred * (1.0 / num_inference_steps)

        # Decode latents to video
        video = self.decode_latents(latents)

        return video
