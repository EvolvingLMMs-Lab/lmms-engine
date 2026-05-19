# coding=utf-8
# Copyright 2025 LMMs-Lab team. All rights reserved.
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

"""AeroRealtime model implementation.

A multimodal model combining audio, vision, and language capabilities.
Inspired by VoxtralRealtime but extended with vision support for
image and video inputs.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.initialization import normal_, zeros_
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.utils import logging

from ..common_ops.rope import qwen3_vl_get_rope_index
from .backbone_registry import family_text_model_cls, family_vision_model_cls
from .configuration_aero_realtime import AeroRealtimeConfig

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class AeroRealtimeCausalLMOutputWithPast(ModelOutput):
    """
    Output class for AeroRealtime causal language model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states for sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states at each layer output.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights after softmax.
        audio_hidden_states (`torch.FloatTensor`, *optional*):
            Audio hidden states produced by the audio encoder after projection.
        vision_hidden_states (`torch.FloatTensor`, *optional*):
            Vision hidden states produced by the vision encoder.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    audio_hidden_states: Optional[torch.FloatTensor] = None
    vision_hidden_states: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Audio multi-modal projector (same architecture as VoxtralRealtime)
# ---------------------------------------------------------------------------


class AeroRealtimeMultiModalProjector(nn.Module):
    """Two-layer MLP projector that maps (downsampled) audio features to the
    language model's hidden dimension.

    Input dimension:  ``audio_config.hidden_size * downsample_factor``
    Output dimension: ``text_config.hidden_size``

    Architecture mirrors ``VoxtralRealtimeMultiModalProjector``: two linear
    layers (no bias) with a configurable activation in between.
    """

    def __init__(self, config: AeroRealtimeConfig):
        super().__init__()
        audio_hidden_size = config.audio_hidden_size
        text_hidden_size = config.text_config.hidden_size
        downsample_factor = config.downsample_factor

        self.linear_1 = nn.Linear(
            audio_hidden_size * downsample_factor,
            text_hidden_size,
            bias=False,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            text_hidden_size,
            text_hidden_size,
            bias=False,
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------


class AeroRealtimePreTrainedModel(PreTrainedModel):
    """Base class for AeroRealtime models, providing weight initialization
    and a simple pre/post-init interface."""

    config_class = AeroRealtimeConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "AeroRealtimeMultiModalProjector",
    ]

    def _init_weights(self, module):
        std = (
            self.config.text_config.initializer_range if hasattr(self.config.text_config, "initializer_range") else 0.02
        )
        if isinstance(module, nn.Linear):
            normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight.data[module.padding_idx].zero_()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class AeroRealtimeForConditionalGeneration(AeroRealtimePreTrainedModel, GenerationMixin):
    """AeroRealtime multimodal model for conditional generation.

    Combines:
    - An audio encoder tower (``audio_tower``)
    - A vision encoder tower (``vision_tower``)
    - A language model backbone (``language_model``, Qwen3VLTextModel)
    - An LM head (``lm_head``)
    - A 2-layer MLP audio projector (``multi_modal_projector``)

    The vision tower (e.g. Qwen3 VL / Qwen3.5 VL) includes a built-in
    merger/projector that already maps vision features to the text hidden
    dimension, so no separate vision MLP is needed.
    """

    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _tied_weights_keys = {"lm_head.weight": "language_model.embed_tokens.weight"}

    def __init__(self, config: AeroRealtimeConfig):
        super().__init__(config)

        self.vocab_size = config.text_config.vocab_size

        # --- Sub-modules ---
        text_model_cls = family_text_model_cls(config.backbone_family)
        vision_model_cls = family_vision_model_cls(config.backbone_family)
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.vision_tower = vision_model_cls._from_config(config.vision_config)
        self.language_model = text_model_cls._from_config(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.multi_modal_projector = AeroRealtimeMultiModalProjector(config)

        # Cache audio tower type for dispatch
        self.audio_tower_type = config.audio_config.model_type

        self.post_init()

        # Cached rope deltas for incremental position_ids during generation
        self.rope_deltas = None

    # --- Embedding accessors ---

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        is_first_iteration=False,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        # aero_realtime-specific kwargs (forwarded via **kwargs by super)
        text_stream_ids=None,
        input_features=None,
        audio_attention_mask=None,
        **kwargs,
    ):
        # Let the default GenerationMixin handle input_ids slicing,
        # position_ids slicing, attention mask, and passthrough of extra
        # kwargs -- mirrors official Qwen3VL.
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            # pass aero-specific kwargs so super forwards them
            text_stream_ids=text_stream_ids,
            input_features=input_features,
            audio_attention_mask=audio_attention_mask,
            **kwargs,
        )

        # After the first iteration the multimodal features have been
        # consumed and cached in KV-cache — clear them to avoid
        # recomputation (same pattern as official Qwen3VL).
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["input_features"] = None
            model_inputs["audio_attention_mask"] = None
            # text_stream_ids only meaningful during prefill
            model_inputs["text_stream_ids"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """Compute 3D MROPE position ids, aligned with official Qwen3VL."""

        # Standard 2D text positions from parent
        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        # Continuing generation from past KV — use cached rope_deltas
        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.rope_deltas is not None:
            position_ids = text_positions[None, ...] + self.rope_deltas
            return position_ids

        # First call: compute 3D vision positions via the shared rope helper
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        has_vision = model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None

        if is_input_ids and has_vision:
            vision_positions, rope_deltas = qwen3_vl_get_rope_index(
                self,
                inputs_tensor,
                image_grid_thw=model_kwargs.get("image_grid_thw"),
                video_grid_thw=model_kwargs.get("video_grid_thw"),
                attention_mask=model_kwargs.get("attention_mask"),
            )
            self.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.rope_deltas = torch.zeros(
                inputs_tensor.shape[0],
                1,
                dtype=torch.long,
                device=inputs_tensor.device,
            )

        # Concatenate text + vision → [4, B, S]
        text_positions = text_positions[None, ...]
        position_ids = torch.cat([text_positions, vision_positions], dim=0)

        return position_ids

    # Weight tying is handled by the parent class via ``_tied_weights_keys``
    # which maps ``lm_head.weight`` -> ``language_model.embed_tokens.weight``.

    # --- Feature extraction helpers ---

    def get_vision_features(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Extract vision features via the vision tower.

        The vision tower (e.g. Qwen3 VL) includes a built-in merger that
        projects features to ``text_config.hidden_size``.  The merged output
        is returned from ``pooler_output`` (post-merger), not
        ``last_hidden_state`` (pre-merger).

        Args:
            pixel_values: Pixel values, shape depends on the vision tower.
                Typically ``[total_patches, C, H, W]`` (flat across batch).
            grid_thw: Grid of ``(temporal, height, width)`` per image/video.
                Shape ``[num_images_or_videos, 3]``.

        Returns:
            Vision features of shape ``[total_merged_tokens, hidden_dim]``
            where ``hidden_dim = text_config.hidden_size`` and
            ``total_merged_tokens = total_patches / merge_size^2``.
        """
        if grid_thw is not None:
            vision_outputs = self.vision_tower(pixel_values, grid_thw=grid_thw)
        else:
            vision_outputs = self.vision_tower(pixel_values)

        if isinstance(vision_outputs, torch.Tensor):
            return vision_outputs

        # Prefer pooler_output (post-merger) over last_hidden_state (pre-merger)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            return vision_outputs.pooler_output

        return vision_outputs.last_hidden_state

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract audio features via the audio tower, downsample, and project.

        Following VoxtralRealtime's approach:
        1. Run audio through the audio encoder tower.
        2. Reshape by concatenating ``downsample_factor`` consecutive frames.
        3. Project through the 2-layer MLP (``multi_modal_projector``).

        Args:
            input_features: Mel spectrogram features.
                Shape ``[batch_size, num_mel_bins, mel_seq_len]`` or as
                expected by the audio tower.
            audio_attention_mask: Attention mask at mel-frame level.
                Shape ``[batch_size, mel_seq_len]``.

        Returns:
            Tuple of:
            - Projected audio features of shape
              ``[batch_size, num_audio_tokens, hidden_dim]``
              where ``num_audio_tokens = encoder_seq_len / downsample_factor``
              and ``hidden_dim = text_config.hidden_size``.
            - ``audio_output_lengths`` of shape ``[batch_size]``, the number
              of valid (unpadded) encoder output tokens per sample.
        """
        # Compute audio feature/output lengths via encoder's own method
        audio_output_lengths = None
        encoder_kwargs = {}

        if audio_attention_mask is not None and hasattr(self.audio_tower, "_get_feat_extract_output_lengths"):
            mel_lengths = audio_attention_mask.sum(-1)
            audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(mel_lengths)

            # Qwen2Audio encoder needs a custom 4D attention mask
            if self.audio_tower_type == "qwen2_audio_encoder":
                encoder_kwargs["attention_mask"] = self._build_qwen2_audio_attention_mask(
                    input_features, audio_feat_lengths
                )

        audio_outputs = self.audio_tower(input_features=input_features, **encoder_kwargs)

        if isinstance(audio_outputs, torch.Tensor):
            audio_hidden_states = audio_outputs
        else:
            audio_hidden_states = audio_outputs.last_hidden_state

        # Downsample: concatenate `downsample_factor` consecutive frames
        # [B, seq_len, audio_hidden] -> [B, seq_len/df, audio_hidden * df]
        # Truncate seq_len to nearest multiple of downsample_factor
        df = self.config.downsample_factor
        seq_len = audio_hidden_states.shape[1]
        usable_len = (seq_len // df) * df
        audio_hidden_states = audio_hidden_states[:, :usable_len, :]

        audio_hidden_states = audio_hidden_states.reshape(
            audio_hidden_states.shape[0],
            -1,
            self.config.audio_hidden_size * df,
        )

        # Project to text hidden dim
        audio_features = self.multi_modal_projector(audio_hidden_states)

        # Adjust output lengths for downsample factor
        if audio_output_lengths is not None:
            audio_output_lengths = audio_output_lengths // self.config.downsample_factor

        return audio_features, audio_output_lengths

    def _build_qwen2_audio_attention_mask(
        self,
        input_features: torch.Tensor,
        audio_feat_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Build the 4D attention mask expected by the Qwen2Audio encoder.

        The encoder applies two downsampling stages (conv2 stride=2, avg_pool
        stride=2), so the mask is built at the post-conv2 sequence length.

        Args:
            input_features: ``[batch_size, num_mel_bins, mel_seq_len]``
            audio_feat_lengths: ``[batch_size]`` -- number of valid tokens
                after the first conv2 downsampling.

        Returns:
            4D float attention mask of shape
            ``[batch_size, 1, max_seq_len, max_seq_len]`` with ``-inf``
            at padding positions.
        """
        batch_size = input_features.shape[0]
        max_mel_seq_len = input_features.shape[-1]
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1

        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        padding_mask = seq_range >= lengths_expand

        attention_mask = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        attention_mask = attention_mask.to(
            dtype=self.audio_tower.conv1.weight.dtype,
            device=self.audio_tower.conv1.weight.device,
        )
        attention_mask = attention_mask.clone()
        attention_mask[padding_mask.view(batch_size, 1, 1, max_seq_len).expand_as(attention_mask)] = float("-inf")
        return attention_mask

    @staticmethod
    def _unpad_audio_features(
        audio_features: torch.Tensor,
        audio_output_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Remove padding from batched audio features and flatten.

        Args:
            audio_features: ``[batch_size, max_seq_len, hidden_dim]``
            audio_output_lengths: ``[batch_size]`` -- valid token count per sample.

        Returns:
            Flat tensor ``[total_valid_tokens, hidden_dim]``.
        """
        unpadded = [feat[:length] for feat, length in zip(audio_features, audio_output_lengths)]
        return torch.cat(unpadded, dim=0)

    # --- Forward ---

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        text_stream_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # Audio inputs
        input_features: Optional[torch.FloatTensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        # Vision inputs — images
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        # Vision inputs — videos
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # Standard args
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, AeroRealtimeCausalLMOutputWithPast]:
        """Forward pass for AeroRealtime.

        Audio and video are kept as **separate** token streams in the input
        sequence (per-chunk envelope ``[VS][video_pad×S][VE][AS]
        [audio_pad×N][AE]``) so time alignment is expressed entirely through token
        order and RoPE.  Vision features replace vision placeholders; audio
        features are added to the realtime text stream on audio placeholders.

        Modality combinations:

        **Image mode** (``pixel_values`` + ``image_grid_thw``):
            Vision features are scattered (replace) at ``image_token_index``
            positions.  ``text_stream_ids`` is not used.

        **Video mode** (``pixel_values_videos`` + ``video_grid_thw``):
            Video features are scattered (replace) at
            ``video_token_index`` positions.

        **Audio mode** (``input_features``):
            Audio features are **added** to embeddings at
            ``audio_token_index`` positions.

        **Video + Audio**: video placeholders receive pure vision features.
        ``text_stream_ids`` carries realtime markers (``<|rt_speak|>``,
        ``<|rt_start|>``, ``<|rt_end|>``, and speech text) only at audio
        positions, where audio features are added to the realtime text embeddings.

        Pipeline:
            1. Embed ``text_stream_ids`` (if provided) or ``input_ids``.
            2. Image features → scatter at ``image_token_index``.
            3. Video features → scatter at ``video_token_index``.
            4. Audio features → add at ``audio_token_index``.
            5. Forward through the language model.

        Args:
            input_ids: Token ids with placeholder tokens for vision/audio.
                Shape ``[batch_size, seq_len]``.  Used to determine the
                position masks for image/video/audio features.
            text_stream_ids: Parallel text-stream token ids.
                Shape ``[batch_size, seq_len]``.  At audio positions contains
                ``<|rt_pad|>``, ``<|rt_speak|>``, speech boundary tokens, or
                actual text tokens; mirrors ``input_ids`` elsewhere.
                If not provided, falls back to ``input_ids``.
            pixel_values: Image pixel values (flat across batch).
            image_grid_thw: Grid info per image. ``[num_images, 3]``.
            pixel_values_videos: Video pixel values (flat across batch).
            video_grid_thw: Grid info per video. ``[num_videos, 3]``.
            input_features: Audio mel spectrogram features.
                Shape ``[batch_size, num_mel_bins, mel_seq_len]``.
            audio_attention_mask: Mel-level attention mask for audio.
                Shape ``[batch_size, mel_seq_len]``.
            labels: Target token ids for loss computation. ``-100`` ignored.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Determine which token ids to use for embedding
        # text_stream_ids provides the realtime text tokens at audio positions.
        # input_ids is used for determining modality placeholder masks.
        embed_ids = text_stream_ids if text_stream_ids is not None else input_ids

        # ----------------------------------------------------------------
        # 1. Text embeddings (from text_stream_ids or input_ids)
        # ----------------------------------------------------------------
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(embed_ids)

        # ----------------------------------------------------------------
        # 2. Image features — extract and scatter at image_token_index
        #    (image mode is unchanged — uses scatter/replace)
        # ----------------------------------------------------------------
        if pixel_values is not None:
            image_features = self.get_vision_features(pixel_values, grid_thw=image_grid_thw)

            image_mask = input_ids == self.config.image_token_index
            n_image_tokens = image_mask.sum().item()
            n_image_features = image_features.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image token count ({n_image_tokens}) does not match " f"image feature count ({n_image_features})."
                )

            image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask_expanded,
                image_features.to(inputs_embeds.dtype),
            )

        # ----------------------------------------------------------------
        # 3. Video features — extract (scatter happens below)
        # ----------------------------------------------------------------
        video_features = None
        if pixel_values_videos is not None:
            video_features = self.get_vision_features(pixel_values_videos, grid_thw=video_grid_thw)

        # ----------------------------------------------------------------
        # 4. Audio features — extract, downsample, project
        # ----------------------------------------------------------------
        audio_features = None
        audio_output_lengths = None
        audio_features_flat = None
        if input_features is not None:
            audio_features, audio_output_lengths = self.get_audio_features(
                input_features, audio_attention_mask=audio_attention_mask
            )
            # Flatten, removing padding if output lengths are available
            if audio_output_lengths is not None:
                audio_features_flat = self._unpad_audio_features(audio_features, audio_output_lengths)
            else:
                audio_features_flat = audio_features.reshape(-1, audio_features.shape[-1])

        # ----------------------------------------------------------------
        # 5. Scatter video features and add audio features to text-stream embeddings.
        #    Realtime text conditioning lives only on the audio timeline.
        # ----------------------------------------------------------------

        # 5a. Video features -> scatter at video_token_index positions
        if video_features is not None:
            video_mask = input_ids == self.config.video_token_index
            n_video_tokens = video_mask.sum().item()
            n_video_features = video_features.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video token count ({n_video_tokens}) does not match " f"video feature count ({n_video_features})."
                )

            video_mask_expanded = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask_expanded,
                video_features.to(inputs_embeds.dtype),
            )

        # 5b. Audio features -> add at audio_token_index positions
        if audio_features_flat is not None:
            audio_mask = input_ids == self.config.audio_token_index
            n_audio_tokens = audio_mask.sum().item()
            n_audio_features = audio_features_flat.shape[0]
            if n_audio_tokens != n_audio_features:
                raise ValueError(
                    f"Audio token count ({n_audio_tokens}) does not match " f"audio feature count ({n_audio_features})."
                )

            audio_mask_flat = audio_mask.reshape(-1)
            inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            inputs_embeds_flat[audio_mask_flat] = inputs_embeds_flat[audio_mask_flat] + audio_features_flat.to(
                inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds_flat.reshape(inputs_embeds.shape)

        # ----------------------------------------------------------------
        # 6. Language model forward + LM head
        # ----------------------------------------------------------------
        # Qwen3VLTextModel returns BaseModelOutputWithPast (hidden states,
        # no logits/loss).  We apply lm_head and compute loss here.
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AeroRealtimeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_features_flat,
            vision_hidden_states=video_features,
        )
