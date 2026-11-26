# coding=utf-8
# Vendored from the official MiniCPM-o 4.5 release (openbmb/MiniCPM-o-4_5,
# Apache-2.0) and trimmed for training use inside lmms-engine:
#   - the ``MiniCPMODuplex`` streaming/inference wrapper and the session /
#     chat / generation helpers were removed (training only needs the
#     embedding-injection forward path),
#   - ``init_tts`` / ``_ensure_asset_dir`` (which download Token2wav assets
#     at runtime) were removed; the frozen ``MiniCPMTTS`` module is kept so
#     checkpoints load and round-trip byte-identically,
#   - ``MiniCPMOForTraining`` adds a Hugging Face Trainer-compatible
#     ``forward`` signature on top of the upstream ``MiniCPMO``.
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

import inspect
import math
import os
import types
from dataclasses import dataclass
from functools import partial
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm

from transformers import LlamaConfig
from transformers import LlamaModel
from transformers import PreTrainedModel
from transformers import Qwen3ForCausalLM
from transformers import Qwen3PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import EncoderDecoderCache
from transformers.cache_utils import StaticCache
from transformers.generation.logits_process import TopKLogitsWarper
from transformers.generation.logits_process import TopPLogitsWarper
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import ModelOutput
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from .configuration_minicpmo import MiniCPMOConfig
from .configuration_minicpmo import MiniCPMTTSConfig
from .modeling_navit_siglip import SiglipVisionTransformer

import logging

logger = logging.getLogger(__name__)


@dataclass
class TTSSamplingParams:
    top_p: float = 0.85
    min_p: float = 0.01
    top_k: int = 25
    repetition_penalty: float = 1.05
    temperature: float = 0.8
    win_size: int = 16
    tau_r: float = 0.1



class MiniCPMOPreTrainedModel(Qwen3PreTrainedModel):
    config_class = MiniCPMOConfig


class MiniCPMO(MiniCPMOPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.llm = Qwen3ForCausalLM(config)
        self.embed_dim = self.llm.config.hidden_size
        self.llm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, self.llm)  # patch llm

        # init vision module
        if self.config.init_vision:
            self.vpm = self.init_vision_module()
            self.vision_dim = self.vpm.embed_dim
            self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)

        # init audio module
        if self.config.init_audio:
            self.apm = self.init_audio_module()
            audio_output_dim = int(self.apm.config.encoder_ffn_dim // 4)
            self.audio_avg_pooler = nn.AvgPool1d(self.config.audio_pool_step, stride=self.config.audio_pool_step)
            self.audio_projection_layer = MultiModalProjector(in_dim=audio_output_dim, out_dim=self.embed_dim)
            self.audio_encoder_layer = -1

        # init tts module
        if self.config.init_tts:
            self.tts = self.init_tts_module()

        # streaming audio processing constants (shared with the official
        # inference code; harmless for training)
        self.SAMPLE_RATE = 16000
        self.CHUNK_MS = 1000  # regular chunk length (ms)
        self.FIRST_CHUNK_MS = 1035  # first chunk length (ms)
        self.CNN_REDUNDANCY_MS = 0  # CNN redundancy (ms)

        # lmms-engine addition: upstream never calls post_init(); newer
        # transformers (>=5) require the tied-weights registry it sets up.
        self.post_init()

    def init_vision_module(self):
        if self.config._attn_implementation == "flash_attention_2":
            self.config.vision_config._attn_implementation = "flash_attention_2"
        else:
            self.config.vision_config._attn_implementation = "eager"
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True,
        )

    def init_audio_module(self):
        if self.config._attn_implementation == "eager":
            self.config.audio_config._attn_implementation = "eager"
        else:
            # using flash_attention_2 will cause: RuntimeError: cu_seqlens_q must have shape (batch_size + 1)
            self.config.audio_config._attn_implementation = "sdpa"

        return MiniCPMWhisperEncoder(self.config.audio_config)

    def init_tts_module(self):
        if self.config._attn_implementation == "flash_attention_2":
            self.config.tts_config.attn_implementation = "flash_attention_2"
        else:
            self.config.tts_config.attn_implementation = "eager"

        return MiniCPMTTS(config=self.config.tts_config, audio_tokenizer=None)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    @staticmethod
    def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        """Create mask for subsequent steps (size, size) with chunk size,
        this is for streaming encoder

        Args:
            size (int): size of mask
            chunk_size (int): size of chunk
            num_left_chunks (int): number of left chunks
                <0: use full chunk
                >=0: use num_left_chunks
            device (torch.device): "cpu" or "cuda" or torch.Tensor.device
            num_lookhead:

        Returns:
            torch.Tensor: mask
        """
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        for i in range(size):
            if num_left_chunks < 0:
                start = 0
            else:
                start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
            ending = min((i // chunk_size + 1) * chunk_size + num_lookhead, size)
            ret[i, start:ending] = True
        return ret

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Computes the output length of the convolutional layers and the output length of the audio encoder"""
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_vision_embedding(self, data):
        if "vision_hidden_states" not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0
                )
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                            tgt_sizes=tgt_sizes[start_idx:end_idx],
                        ).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values,
                        patch_attention_mask=patch_attn_mask,
                        tgt_sizes=tgt_sizes,
                    ).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                    tgt_sizes = torch.Tensor(
                        [
                            [
                                (224 // self.config.patch_size),
                                math.ceil(224 / self.config.patch_size),
                            ]
                        ]
                    ).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)
        else:
            vision_hidden_states = data["vision_hidden_states"]

        return vision_hidden_states

    def get_vllm_embedding(self, data):
        vision_hidden_states = self.get_vision_embedding(data)

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]

        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(
                    0,
                    max_seq_len,
                    dtype=audio_feature_lens.dtype,
                    device=audio_feature_lens.device,
                )
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand  # 1 for padded values

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.apm.conv1.weight.dtype, device=self.apm.conv1.weight.device
            )

            if chunk_length > 0:
                chunk_num_frame = int(chunk_length * 50)
                chunk_mask = self.subsequent_chunk_mask(
                    size=max_seq_len,
                    chunk_size=chunk_num_frame,
                    num_left_chunks=-1,
                    device=audio_attention_mask_.device,
                )
                audio_attention_mask_ = torch.logical_or(audio_attention_mask_, torch.logical_not(chunk_mask))

            audio_attention_mask[audio_attention_mask_] = float("-inf")
            audio_states = self.apm(
                wavforms, output_hidden_states=True, attention_mask=audio_attention_mask
            ).hidden_states[self.audio_encoder_layer]
            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        elif self.training and dummy:
            dtype = self.apm.embed_positions.weight.dtype
            device = self.apm.embed_positions.weight.device

            dummy_wavs = torch.zeros((1, 80, 100), device=device, dtype=dtype)
            audio_states = self.apm(dummy_wavs, output_hidden_states=True).hidden_states[self.audio_encoder_layer]

            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)
            return [audio_embeds]
        else:
            return []

    def get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
        """
        Args:
            data:
            input_embeddings:
            chunk_length: whisper use full attention or chunk attention
            stream_input: use streaming audio embedding or not

        Returns:
            final embeddings with audio feature
        """
        if stream_input:
            audio_embeddings = self.get_audio_embedding_streaming(data)
        else:
            audio_embeddings = self.get_audio_embedding(data, chunk_length)

        bs = len(input_embeddings)
        if len(data.get("audio_features", [])) > 0:
            assert len(audio_embeddings) == len(input_embeddings)

            if len(audio_embeddings) > 0:
                audio_bounds = data["audio_bounds"]

                if self.config.stream_input:
                    assert bs == 1, "audio stream_input mode only support batch size 1"
                    for i in range(bs):
                        audio_embs = torch.cat(audio_embeddings[i], dim=0).to(
                            device=input_embeddings.device, dtype=input_embeddings.dtype
                        )
                        audio_start_pos = 0
                        for bound in audio_bounds[i]:
                            audio_len = bound[1] - bound[0]
                            input_embeddings[i, bound[0] : bound[1]] = audio_embs[
                                audio_start_pos : audio_start_pos + audio_len, :
                            ]
                            audio_start_pos += audio_len
                else:
                    for i in range(bs):
                        audio_embs = audio_embeddings[i]
                        bounds = audio_bounds[i]

                    one_to_one_match = len(audio_embs) == len(bounds) and all(
                        embs.shape[0] == int(bound[1] - bound[0])
                        for embs, bound in zip(audio_embs, bounds)
                    )

                    if one_to_one_match:
                        for embs, bound in zip(audio_embs, bounds):
                            input_embeddings[i, bound[0] : bound[1]] = embs.to(
                                device=input_embeddings.device,
                                dtype=input_embeddings.dtype,
                            )
                    else:
                        flat_audio_embs = torch.cat(audio_embs, dim=0).to(
                            device=input_embeddings.device,
                            dtype=input_embeddings.dtype,
                        )
            
                        total_bound_len = sum(int(bound[1] - bound[0]) for bound in bounds)
            
                        if flat_audio_embs.shape[0] != total_bound_len:
                            raise ValueError(
                                f"Audio total length mismatch: {flat_audio_embs.shape[0]} != {total_bound_len}"
                            )
            
                        offset = 0
                        for bound in bounds:
                            audio_len = int(bound[1] - bound[0])
                            input_embeddings[i, bound[0] : bound[1]] = flat_audio_embs[
                                offset : offset + audio_len
                            ]
                            offset += audio_len
                            
        elif self.training:
            for i in range(bs):
                # dummy audio_embedings
                input_embeddings += audio_embeddings[0].mean() * 0

        return input_embeddings

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        vllm_embedding = self.get_omni_embedding(
            data,
            input_embeddings=vllm_embedding,
            chunk_length=self.config.audio_chunk_length,
        )

        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs,
        )
def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
       given learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (batch_size, num_queries, embed_dim)
    """

    def __init__(
        self,
        num_queries,
        embed_dim,
        num_heads,
        kv_dim=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        adaptive=False,
        max_size=(70, 70),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))

        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size, device="cpu"):
        if is_deepspeed_zero3_enabled():
            device = "cuda"
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes, device):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, tgt_sizes=None):
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(
            1, 0, 2
        )  # BLD => L * B * D

        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class MiniCPMWhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model
        try:
            # compatible old transformers
            from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES

            self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
                layer_idx=layer_idx,
            )
        except:
            from transformers.models.whisper.modeling_whisper import WhisperAttention

            self.self_attn = WhisperAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
                layer_idx=layer_idx,
            )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # lmms-engine fix: the upstream 3-tuple unpacking breaks on newer
        # transformers, whose attention returns 2 values and dropped
        # layer_head_mask.  Adapt to the installed signature.
        attn_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "output_attentions": output_attentions,
        }
        attn_params = inspect.signature(self.self_attn.forward).parameters
        if "layer_head_mask" in attn_params:
            attn_kwargs["layer_head_mask"] = layer_head_mask
        if "past_key_value" in attn_params:
            attn_kwargs["past_key_value"] = past_key_values
        attn_outputs = self.self_attn(**attn_kwargs)
        hidden_states = attn_outputs[0]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


# Copied from from transformers.models.whisper.modeling_whisper.WhisperEncoder and add use_cache for streaming inference
class MiniCPMWhisperEncoder(WhisperEncoder):

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MiniCPMWhisperEncoderLayer(config, layer_idx=i) for i in range(config.encoder_layers)]
        )

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        use_extra_context: Optional[bool] = False,
        prefix_extra_frames: Optional[int] = 1,
        suffix_extra_frames: Optional[int] = 1,
        cnn_min_length: Optional[int] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        # Optional: pad short input to minimum length for CNN computation consistency
        original_length = input_features.shape[2]
        padded_for_cnn = False
        if cnn_min_length is not None and original_length < cnn_min_length:
            padded_features = torch.zeros(
                input_features.shape[0],
                input_features.shape[1],
                cnn_min_length,
                dtype=input_features.dtype,
                device=input_features.device,
            )
            padded_features[:, :, :original_length] = input_features
            input_features = padded_features
            padded_for_cnn = True

        conv1_output = self.conv1(input_features)
        inputs_embeds = nn.functional.gelu(conv1_output)
        conv2_output = self.conv2(inputs_embeds)
        inputs_embeds = nn.functional.gelu(conv2_output)
        # If padding was done before, now need to remove the effect of padding
        if padded_for_cnn:
            # Conv1: stride=1, output length=input length
            # Conv2: stride=2, output length=(input length+1)//2
            actual_cnn_output_length = (original_length + 1) // 2
            inputs_embeds = inputs_embeds[:, :, :actual_cnn_output_length]

        # If extra context is used, CNN operations need to remove redundant frames
        # conv2 stride=2, so the redundant frames in the input will be halved (upward rounding)
        if use_extra_context:
            # Input has prefix_extra_frames prefix frames and suffix_extra_frames suffix frames
            # conv2 stride=2, output length = ceil(input length / 2)
            # For 2 redundant frames, the output is 1 frame (ceil(2/2) = 1)
            prefix_to_remove = (prefix_extra_frames + 1) // 2 if prefix_extra_frames > 0 else 0
            suffix_to_remove = (suffix_extra_frames + 1) // 2 if suffix_extra_frames > 0 else 0

            # Remove redundant frames before and after (batch, channels, time)
            if prefix_to_remove > 0:
                inputs_embeds = inputs_embeds[:, :, prefix_to_remove:]
            if 0 < suffix_to_remove < inputs_embeds.shape[2]:
                inputs_embeds = inputs_embeds[:, :, :-suffix_to_remove]

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, list):
                past_key_values = EncoderDecoderCache(DynamicCache.from_legacy_cache(past_key_values), DynamicCache())
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            else:
                pass
            past_key_values_length = past_key_values.self_attention_cache.get_usable_length(inputs_embeds.shape[1])
            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
                logger.warning("seems the audio is longer than 30s. repeating the last part of the audio")
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat(
                    (
                        embed_pos_front,
                        torch.repeat_interleave(
                            embed_pos[-1, :].unsqueeze(0),
                            inputs_embeds.shape[1] - embed_pos.shape[0] + past_key_values_length,
                            dim=0,
                        ),
                    )
                )
            else:
                embed_pos = embed_pos[past_key_values_length : inputs_embeds.shape[1] + past_key_values_length, :]
        else:
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                        past_key_values,
                        use_cache,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

            if use_cache:
                next_encoder_cache = layer_outputs[2 if output_attentions else 1]
            else:
                next_encoder_cache = None

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            result = tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
            return result
        result = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            past_key_values=next_encoder_cache,
        )

        return result


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, audio_features):
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class MiniCPMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_dim = config.llm_hidden_size
        self.out_dim = config.hidden_size
        self.intermediate_size = config.llm_intermediate_size
        self.gate_proj = nn.Linear(self.in_dim, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.in_dim, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.out_dim, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


@dataclass
class MiniCPMTTSGenerationOutput(ModelOutput):
    """
    Output class for MiniCPMTTS generation.

    Args:
        new_ids (torch.LongTensor): Newly generated audio code sequence, shape (batch_size, sequence_length, num_vq).
        audio_input_ids (torch.LongTensor): Updated input IDs including condition and generated audio codes, shape (batch_size, full_sequence_length, num_vq).
        past_key_values (Tuple[Tuple[torch.FloatTensor]]): Tuple containing pre-computed keys and values used for attention mechanism. Each element has shape (batch_size, num_heads, sequence_length, embed_size_per_head).
        finished (bool): Boolean indicating whether generation is complete.
    """

    new_ids: torch.LongTensor = None
    audio_input_ids: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_input_ids: Optional[torch.LongTensor] = None
    finished: bool = None


def make_streaming_chunk_mask_inference(
    tts_text_scope: List[int],
    tts_text_mask: torch.Tensor,
    streaming_audio_chunk_size: int = 50,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    max_sequence_length: int = 4096,
):
    """
    Example:
    Input sequence:
    [t1, t2, t3, t4, t5, [Ptts], a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, ...]
    Output 4D causal mask:
    ------- text positions -------
    [0] <- here is [Stts]
    [0,   0] <- here is [spk_emb] * N
    [0,   0,   0]
    [0,   0,   0,    0]
    [0,   0,   0,    0,    0]
    ------- audio positions --------
    [0,    0, -inf, -inf, -inf, 0] <- here is [Ptts], [Ptts]'s last hidden state should predict the first audio token
                                v- here is [Ptts]
    [0,    0, -inf, -inf, -inf, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0, 0, 0] # end of first 1s audio chunk
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """

    # Create a complete attention mask for input embeds [batch_size, seq_len], without considering audio mask as audio is always at the end

    assert tts_text_mask.dtype == torch.int8

    padding_mask = torch.ones(max_sequence_length, dtype=torch.int8, device=device)
    padding_mask[tts_text_scope[0] : tts_text_scope[1]] = tts_text_mask

    # Initialize a standard upper triangular causal mask
    min_dtype = torch.finfo(dtype).min

    causal_mask = torch.full(
        (max_sequence_length, max_sequence_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    if max_sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    else:
        raise ValueError("max_sequence_length of tts could not be 1.")

    # For each data sample
    audio_token_start = tts_text_scope[1]
    audio_duration = max_sequence_length - tts_text_scope[1]

    # Record which text chunk the current audio chunk can see up to
    text_pivot = 0
    num_valid_text_tokens = torch.sum(tts_text_mask).item() - 1  # [Ptts] excluded
    # How many audio chunks are in total, the num of buckets should be smaller as possible

    num_text_tokens_per_audio_chunk = 10

    # For each chunk of audio
    for chunk_idx in range(math.ceil(audio_duration / streaming_audio_chunk_size)):
        audio_chunk_start = audio_token_start + chunk_idx * streaming_audio_chunk_size
        audio_chunk_end = audio_token_start + (chunk_idx + 1) * streaming_audio_chunk_size
        # New text seen by this new audio chunk
        new_text_this_chunk = num_text_tokens_per_audio_chunk
        # The right bound of visible text tokens
        text_pivot = min(new_text_this_chunk + text_pivot, num_valid_text_tokens)
        # Mask all text chunks after the visible ones
        # -> [text_pivot, len(tts_text_scope)-1] excluding [Ptts]
        causal_mask[
            audio_chunk_start - 1 : audio_chunk_end - 1,
            # tts_text_scope[0] + text_pivot: tts_text_scope[1],
            tts_text_scope[0] + text_pivot : tts_text_scope[1] - 1,
        ] = min_dtype

    # Mask the padding parts in tts_text_masks (no position will attend to it)
    causal_mask[:, padding_mask == 0] = min_dtype

    # Add extra dimensions, [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return causal_mask


class MiniCPMTTS(PreTrainedModel):
    config_class = MiniCPMTTSConfig

    def __init__(self, config: MiniCPMTTSConfig, audio_tokenizer: None):
        super().__init__(config)

        self.use_llm_hidden_state = config.use_llm_hidden_state

        self.use_text = config.use_text
        self.streaming = config.streaming
        self.streaming_text_chunk_min = config.streaming_text_chunk_min
        self.streaming_text_chunk_max = config.streaming_text_chunk_max
        self.streaming_audio_chunk_size = config.streaming_audio_chunk_size
        self.streaming_text_reserved_len = config.streaming_text_reserved_len
        # streaming tts
        self.streaming_text_chunk_size = config.streaming_text_chunk_max
        self.audio_bos_token_id = config.audio_bos_token_id
        self.num_mel_bins = config.num_mel_bins
        self.num_vq = config.num_vq
        self.num_audio_tokens = config.num_audio_tokens

        self.top_p = config.top_p
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty

        self.interleaved = config.interleaved
        self.attention_type = config.attention_type
        self.recomputed_chunks = config.recomputed_chunks

        # Two different window size concepts:
        # 1. chunk_window_size: number of chunks for sliding_recompute mode (default 2)
        # 2. token_window_size: number of tokens for sliding_window mode (default 300)
        self.chunk_window_size = config.window_size  # chunk-level window for sliding_recompute
        self.token_window_size = (
            config.streaming_sliding_window_audio_window_size
        )  # token-level window for sliding_window

        # Legacy aliases (for backward compatibility with existing code)
        self.window_size = self.chunk_window_size  # used in generate_streaming for sliding_recompute
        self.sliding_window_size = self.token_window_size  # used in TTSStreamingGenerator for sliding_window

        if self.attention_type == "sliding_recompute" and self.chunk_window_size <= self.recomputed_chunks:
            raise ValueError(
                f"sliding_recompute requires chunk_window_size > recomputed_chunks, "
                f"but got chunk_window_size={self.chunk_window_size} and recomputed_chunks={self.recomputed_chunks}"
            )

        if config.backbone_model == "llama":
            model_config = LlamaConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_hidden_layers=config.num_hidden_layers,
                num_key_value_heads=config.num_key_value_heads,
                max_position_embeddings=config.max_position_embeddings,
                attn_implementation=config.attn_implementation,
            )

            self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)

            model = LlamaModel(model_config)
            self.model = model
        else:
            raise ValueError(f"Unsupported backbone model: {config.backbone_model}")

        self.projector_spk = self.create_projector(config)
        self.projector_semantic = self.create_projector(config)

        self.audio_tokenizer = audio_tokenizer

        self.emb_code = nn.ModuleList(
            [nn.Embedding(config.num_audio_tokens, config.hidden_size) for _ in range(config.num_vq)]
        )

        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False),
                    name="weight",
                )
                for _ in range(config.num_vq)
            ]
        )

        self.condition_type = config.condition_type

        return

    @staticmethod
    def create_projector(config):
        if config.projector_type == "mlp":
            return MultiModalProjector(config.llm_dim, config.hidden_size)
        elif config.projector_type == "minicpm":
            return MiniCPMMLP(config)
        elif config.projector_type == "default":
            return nn.Linear(config.llm_dim, config.hidden_size, bias=False)
        else:
            raise ValueError(f"Unsupported projector type: {config.projector_type}")

    # non-streaming
    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        min_new_token=50,
        max_new_token=2048,
        show_tqdm=True,
        streaming=False,
        text_lengths=None,
        sampling_params: TTSSamplingParams = TTSSamplingParams(),
    ):
        temperature = torch.tensor(
            [sampling_params.temperature] * self.config.num_vq,
            dtype=torch.float,
            device=self.device,
        )
        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        condition_length = inputs_embeds.shape[1]
        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        if streaming:
            raise NotImplementedError("this kind of streaming is not supported yet")

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        past_key_values = None

        for t in range(max_new_token):
            audio_bos = False
            # If this is the first audio token, the case is special
            if t == 0:
                audio_bos = True
                inputs_embeds = inputs_embeds
                position_ids = torch.tensor(
                    list(range(0, condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                if streaming:
                    raise NotImplementedError("this kind of streaming is not supported yet")
                else:
                    causal_mask_4d = None

            else:
                code_emb = []
                for q in range(self.num_vq):
                    x = self.emb_code[q](new_tokens[:, t - 1 : t, q])
                    code_emb.append(x)

                inputs_embeds = torch.stack(code_emb, 3).sum(3)

                position_ids = torch.tensor([condition_length + t - 1], dtype=torch.long, device=self.device).unsqueeze(
                    0
                )

                if streaming:
                    raise NotImplementedError("this kind of streaming is not supported yet")
                else:
                    causal_mask_4d = None

            if self.config.backbone_model == "llama":
                outputs: BaseModelOutputWithPast = self.model(
                    position_ids=position_ids,
                    cache_position=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask_4d,
                    use_cache=True,
                    output_attentions=False,
                    # return_dict=True,  # Add this to ensure returns dict with past_key_values
                )
            else:
                raise ValueError(f"Unsupported backbone model: {self.config.backbone_model}")

            del position_ids
            del inputs_embeds

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            logits = logits[:, -1].float()

            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))

            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens
                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

                del logits_token

            if t < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits

            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)

            del scores

            idx_next = idx_next.view(-1, self.num_vq)

            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            logger.warning(f"incomplete result. hit max_new_token: {max_new_token}")

        genrated_input_ids = new_tokens[:, 0:t, :]

        return MiniCPMTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=None,  # for update purpose
            past_key_values=None,  # for update purpose
            past_input_ids=None,  # for update purpose
            finished=finish.all(),
        )

    # fake streaming
    @torch.inference_mode()
    def generate_mock_legacy_streaming(
        self,
        inputs_embeds: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        min_new_token=50,
        max_new_token=2048,
        show_tqdm=True,
        streaming=False,
        text_lengths=None,
        sampling_params: TTSSamplingParams = TTSSamplingParams(),
        valid_text_length=None,
    ):
        assert valid_text_length is not None, "valid_text_length should be not None"

        tts_text_scope = [0, inputs_embeds.shape[1]]
        tts_text_mask = torch.zeros(inputs_embeds.shape[1], dtype=torch.int8, device=inputs_embeds.device)
        tts_text_mask[0:valid_text_length] = 1
        tts_text_mask[-1] = 1  # [Ptts]

        streaming_mask_4d_full = make_streaming_chunk_mask_inference(
            tts_text_scope=tts_text_scope,
            tts_text_mask=tts_text_mask,
            dtype=torch.bfloat16,
            device=self.device,
            streaming_audio_chunk_size=50,
            max_sequence_length=4096,
        )

        temperature = torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.device)
        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        condition_length = inputs_embeds.shape[1]
        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        past_key_values = None

        for t in range(max_new_token):
            audio_bos = False
            if t == 0:
                audio_bos = True
                inputs_embeds = inputs_embeds
                position_ids = torch.tensor(
                    list(range(0, condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                causal_mask_4d = streaming_mask_4d_full[:, :, :condition_length, :condition_length]
            else:
                code_emb = []
                for q in range(self.num_vq):
                    x = self.emb_code[q](new_tokens[:, t - 1 : t, q])
                    code_emb.append(x)

                inputs_embeds = torch.stack(code_emb, 3).sum(3)

                position_ids = torch.tensor([condition_length + t - 1], dtype=torch.long, device=self.device).unsqueeze(
                    0
                )

                causal_mask_4d = streaming_mask_4d_full[
                    :,
                    :,
                    condition_length + t : condition_length + t + 1,
                    : condition_length + t,
                ]

                # get length of past_key_values
                past_key_values_length = past_key_values[0][0].shape[2]

                assert causal_mask_4d.shape[-1] == (past_key_values_length + 1)

            if self.config.backbone_model == "llama":
                outputs: BaseModelOutputWithPast = self.model(
                    position_ids=position_ids,
                    cache_position=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask_4d,
                    use_cache=True,
                    output_attentions=False,
                    # return_dict=True,  # Add this to ensure returns dict with past_key_values
                )
            else:
                raise ValueError(f"Unsupported backbone model: {self.config.backbone_model}")

            del position_ids
            del inputs_embeds

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            logits = logits[:, -1].float()
            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))
            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens

                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

                del logits_token

            if t < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits
            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)

            del scores

            idx_next = idx_next.view(-1, self.num_vq)
            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            logger.warning(f"incomplete result. hit max_new_token: {max_new_token}")

        genrated_input_ids = new_tokens[:, 0:t, :]

        return MiniCPMTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=None,  # for update purpose
            past_key_values=None,  # for update purpose
            past_input_ids=None,  # for update purpose
            finished=finish.all(),
        )

    # non-streaming, interleave
    @torch.inference_mode()
    def generate_chunk(
        self,
        inputs_embeds: torch.Tensor,
        temperature: torch.Tensor,
        repetition_penalty: float,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        max_new_token=500,
        min_new_tokens=0,
        past_key_values=None,
        logits_processors=None,
        text_start_pos=None,
    ):
        """For inputs_embeds, it should be like [bs=1, seq_len, hidden_dim], its content is like:
        |Text BOS|Spk embeds|Text-Hidden states Interleave (if applicable)|Audio BOS|
        where the last position is the audio BOS token.
        So, the first iteration in generation directly forward the model with inputs_embeds, and
        the last hidden states of the last position (Audio BOS) will be decoded to get the first audio token.
        """
        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens, repetition_penalty=repetition_penalty
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        condition_length = inputs_embeds.shape[1]

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        for t in range(max_new_token):
            audio_bos = False

            # If this is the first audio token, the case is special
            if t == 0:
                audio_bos = True
                inputs_embeds_ = inputs_embeds
                position_ids = torch.tensor(
                    list(range(text_start_pos, text_start_pos + condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
            else:
                # Generate the following audio tokens, it is applicable to all other cases, including second and the following calling of `generate`
                inputs_embeds_ = self.emb_code[0](new_tokens[:, t - 1 : t, 0])

                position_ids = torch.tensor(
                    [text_start_pos + condition_length + t - 1],  # prefill the previous token
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

            outputs: BaseModelOutputWithPast = self.model(
                position_ids=position_ids,
                # cache_position=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds_,
                use_cache=True,
                output_attentions=False,
                # return_dict=True,  # Add this to ensure returns dict with past_key_values
            )

            del position_ids
            del inputs_embeds_

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            logits = logits[:, -1].float()
            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))

            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens

                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                del logits_token

            if force_no_stop or t < min_new_tokens:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)
            del logits

            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)
            del scores

            idx_next = idx_next.view(-1, self.num_vq)

            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

        # The latest generated token is not in the range returned this time. If it is an eos token, it is not returned. If it is a normal token, it is not returned.
        genrated_input_ids = new_tokens[:, 0:t, :]

        return genrated_input_ids, past_key_values

    @torch.inference_mode()
    def interleaved_generate(
        self,
        spk_embeds: torch.Tensor,
        conditions: List[torch.Tensor],
        temperature: torch.Tensor,
        repetition_penalty: float,
        eos_token: Union[int, torch.Tensor],
        **kwargs,
    ):
        """
        For inputs_embeds, it should be like [bs=1, seq_len, hidden_dim], its content is like:
        |Text BOS|Spk embeds|Text-Hidden states Interleave (if applicable)|Audio BOS|
        where the last position is the audio BOS token.
        So, the first iteration in generation directly forward the model with inputs_embeds, and the last hidden states of the last position (Audio BOS) will be decoded to get the first audio token.
        """
        temperature = torch.tensor([temperature], dtype=torch.float, device=self.device)

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=repetition_penalty,
        )

        eos_token = eos_token.to(conditions[0].device)

        num_chunks = len(conditions)
        text_start_pos = 0
        last_window_size = 0
        past_key_values = None

        for idx in range(num_chunks):
            condition = conditions[idx].to(conditions[0].device)
            if self.attention_type == "sliding_recompute":
                recomputed_conditions = []

                if (
                    idx >= self.window_size
                    and (idx - self.recomputed_chunks) % (self.window_size - self.recomputed_chunks) == 0
                ):
                    for i in range(self.recomputed_chunks):
                        recomputed_conditions.append(conditions[idx - self.recomputed_chunks + i])
                        recomputed_conditions.append(
                            self.emb_code[0](generated_tokens[-self.recomputed_chunks + i][:, :, 0])
                        )
                    recomputed_conditions.append(condition)
                    condition = torch.cat(recomputed_conditions, dim=1)

                    text_start_pos = 0
                    new_tokens, old_kv = self.generate_chunk(
                        inputs_embeds=condition,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        eos_token=eos_token,
                        force_no_stop=False,
                        max_new_token=500,
                        past_key_values=None,
                        logits_processors=logits_processors,
                        text_start_pos=text_start_pos,
                    )

                else:
                    new_tokens, old_kv = self.generate_chunk(
                        inputs_embeds=condition,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        eos_token=eos_token,
                        force_no_stop=False,
                        max_new_token=500,
                        past_key_values=past_key_values,
                        logits_processors=logits_processors,
                        text_start_pos=text_start_pos,
                    )
            else:
                new_tokens, old_kv = self.generate_chunk(
                    inputs_embeds=condition,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    eos_token=eos_token,
                    force_no_stop=False,
                    max_new_token=500,
                    past_key_values=past_key_values,
                    logits_processors=logits_processors,
                    text_start_pos=text_start_pos,
                )

            past_key_values = []
            if self.attention_type == "sliding_window" and idx >= 1:
                for layer_idx in range(len(old_kv)):
                    past_key_values.append(
                        (
                            old_kv[layer_idx][0][:, :, last_window_size:, :],
                            old_kv[layer_idx][1][:, :, last_window_size:, :],
                        )
                    )
            else:
                past_key_values = old_kv

            last_window_size = condition.shape[1] + new_tokens.shape[1]
            text_start_pos += last_window_size

            if idx == 0:
                generated_tokens = [new_tokens]
            else:
                generated_tokens.append(new_tokens)

        return MiniCPMTTSGenerationOutput(new_ids=torch.cat(generated_tokens, dim=1), finished=True)


class CustomRepetitionPenaltyLogitsProcessorRepeat:
    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        if freq.size(0) > self.max_input_ids:
            freq.narrow(0, self.max_input_ids, freq.size(0) - self.max_input_ids).zero_()
        alpha = torch.pow(self.penalty, freq)
        scores = scores.contiguous()
        inp = scores.multiply(alpha)
        oth = scores.divide(alpha)
        con = scores < 0
        out = torch.where(con, inp, oth)
        del inp, oth, scores, con, alpha
        return out


def gen_logits(num_code: int, top_p=0.7, top_k=20, repetition_penalty=1.0):
    logits_warpers = []

    if top_p is not None:
        logits_warpers.append(TopPLogitsWarper(top_p, min_tokens_to_keep=3))

    if top_k is not None:
        logits_warpers.append(TopKLogitsWarper(top_k, min_tokens_to_keep=3))

    logits_processors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        logits_processors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, num_code, 16))

    return logits_warpers, logits_processors


# Copy and modified from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

            # This clo≠clo≠clone call is needed to avoid recapturing cuda graphs with →rch.comπ≤→rch.comπ≤torch.compile's  mode=reduce−overheadmode=reduce-overheadmode="reduce-overhead, as otherwise the input positionidspositionidsposition_ids would have various stride during the decoding. Here, simply using .contiguous().contiguous().contiguous() is not sufficient as in the batch size = 1 case, positionidspositionidsposition_ids is already contiguous but with varying stride which retriggers a capture.
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    # if ∈putsembeds∈putsembedsinputs_embeds are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    else:
        # The clone here is for the same reason as for positionidspositionidsposition_ids.
        model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs["input_ids"].shape
            device = model_inputs["input_ids"].device

        dtype = self.lm_head.weight.dtype
        min_dtype = torch.finfo(dtype).min

        from transformers.models.paligemma.modeling_paligemma import (
            _prepare_4d_causal_attention_mask_with_cache_position,
        )

        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=past_key_values.get_max_length(),
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=batch_size,
        )

    model_inputs.update(
        {
            "position_ids": position_ids,
            # "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )

    return model_inputs


class MiniCPMOForTraining(MiniCPMO):
    """``MiniCPMO`` with a Hugging Face ``Trainer``-compatible forward.

    The upstream ``MiniCPMO.forward`` consumes a single ``data`` dict, which
    does not fit the ``model(**batch)`` calling convention used by the
    transformers Trainer / FSDP2 trainers.  This wrapper unpacks a standard
    keyword batch into the dict layout expected by ``get_vllm_embedding`` /
    ``get_omni_embedding`` and forwards ``labels`` (plus any extra kwargs
    such as ``num_items_in_batch``) to the inner ``Qwen3ForCausalLM`` so the
    causal-LM loss is computed exactly like a plain HF causal model.

    The TTS head is loaded from the checkpoint but is expected to be frozen
    with ``trainer_args.freeze_modules=["tts"]`` (speech generation is not
    part of this training path).
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[List] = None,
        image_sizes: Optional[List] = None,
        tgt_sizes: Optional[List] = None,
        image_bound: Optional[List] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_lens: Optional[List] = None,
        audio_bounds: Optional[List] = None,
        spk_bounds: Optional[List] = None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            raise ValueError("MiniCPMOForTraining does not support inputs_embeds; pass input_ids instead.")

        data = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_bound": image_bound,
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "audio_bounds": audio_bounds,
            "spk_bounds": spk_bounds,
        }

        vllm_embedding, _ = self.get_vllm_embedding(data)
        vllm_embedding = self.get_omni_embedding(
            data,
            input_embeddings=vllm_embedding,
            chunk_length=self.config.audio_chunk_length,
        )

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(
                input_ids.shape[0], -1
            )
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=vllm_embedding,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

