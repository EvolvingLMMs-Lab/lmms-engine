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
        dit_hidden_size: int = 3072,
        dit_num_layers: int = 30,
        dit_num_heads: int = 24,
        dit_intermediate_size: int = 14336,
        dit_patch_size: tuple = (1, 2, 2),
        dit_in_channels: int = 48,
        dit_out_channels: int = 48,
        dit_freq_dim: int = 256,
        dit_text_dim: int = 4096,
        dit_eps: float = 1e-6,
        dit_has_image_input: bool = False,
        dit_has_image_pos_emb: bool = False,
        dit_has_ref_conv: bool = False,
        dit_add_control_adapter: bool = False,
        dit_in_channels_control_adapter: int = 24,
        trainable_modules=None,
        seperated_timestep: bool = True,
        require_clip_embedding: bool = False,
        require_vae_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        # DiT configuration
        self.dit_hidden_size = dit_hidden_size
        self.dit_num_layers = dit_num_layers
        self.dit_num_heads = dit_num_heads
        self.dit_intermediate_size = dit_intermediate_size
        self.dit_patch_size = dit_patch_size
        self.dit_in_channels = dit_in_channels
        self.dit_out_channels = dit_out_channels
        self.dit_freq_dim = dit_freq_dim
        self.dit_text_dim = dit_text_dim
        self.dit_eps = dit_eps
        self.dit_has_image_input = dit_has_image_input

        self.dit_has_image_pos_emb = dit_has_image_pos_emb
        self.dit_has_ref_conv = dit_has_ref_conv
        self.dit_add_control_adapter = dit_add_control_adapter
        self.dit_in_channels_control_adapter = dit_in_channels_control_adapter

        self.seperated_timestep = seperated_timestep
        self.require_clip_embedding = require_clip_embedding
        self.require_vae_embedding = require_vae_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.trainable_modules = trainable_modules
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        output = super().to_dict()
        return output
