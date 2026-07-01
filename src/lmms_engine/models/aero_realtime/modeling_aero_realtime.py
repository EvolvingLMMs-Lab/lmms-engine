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

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.initialization import normal_, zeros_
from transformers.integrations import (
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernelized_func,
)
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling, logging
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ..common_ops.rope import qwen3_vl_get_rope_index
from .backbone_registry import family_text_model_cls, family_vision_model_cls
from .configuration_aero_realtime import (
    AeroRealtimeAudioEncoderConfig,
    AeroRealtimeConfig,
)

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
    last_hidden_state: Optional[torch.FloatTensor] = None
    cu_seq_lens: Optional[torch.LongTensor] = None
    indices: Optional[torch.LongTensor] = None


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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract audio features via the Voxtral encoder, then reshape and project.

        Voxtral's audio encoder is run on a padded batch with the post-conv2
        attention mask passed through. Transformers' FA2 integration auto-
        unpads on a 2D mask (see ``modeling_flash_attention_utils._upad_input``),
        so padding-region tokens cost nothing in attention/MLP under
        ``flash_attention_2``. The mask is ALSO used downstream to derive
        per-sample valid LM-token lengths via ``audio_output_lengths``.

        Args:
            input_features: Mel spectrogram features, shape ``(B, n_mels=128, T_mel)``.
            audio_attention_mask: **Post-conv2** attention mask, shape
                ``(B, T_enc)`` where ``T_enc = T_mel // 2``. The
                processor is responsible for emitting this at the correct
                length (Voxtral's conv2 with stride=2 reduces mel_len // 2;
                a mel-level mask will trigger a runtime size mismatch).
                ``1 = valid``, ``0 = padding``.

        Returns:
            Tuple of:
              - audio_features: shape ``(B, T_enc // downsample_factor, text_hidden)``
              - audio_output_lengths: shape ``(B,)``; number of valid LM
                audio tokens per sample, or ``None`` if no mask was given.
        """
        audio_outputs = self.audio_tower(
            input_features=input_features,
            attention_mask=audio_attention_mask,
        )
        if isinstance(audio_outputs, torch.Tensor):
            audio_hidden_states = audio_outputs
        else:
            audio_hidden_states = audio_outputs.last_hidden_state

        # Reshape: concat downsample_factor consecutive encoder frames into one
        # LM-bound audio frame. Truncate to a multiple of downsample_factor
        # along seq dim. (B, T_enc, hidden) -> (B, T_enc // df, hidden * df)
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

        # Derive LM-token-level valid lengths from the post-conv2 mask
        audio_output_lengths = None
        if audio_attention_mask is not None:
            audio_output_lengths = audio_attention_mask.sum(-1) // df

        return audio_features, audio_output_lengths

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
        output_last_hidden_state: bool = False,
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
            last_hidden_state=outputs.last_hidden_state if output_last_hidden_state else None,
            cu_seq_lens=None,
            indices=None,
        )


class AeroRealtimeAudioConv1dCacheLayer:
    def __init__(self):
        self.cache: torch.Tensor | None = None
        self.is_initialized: bool = False

    def lazy_initialization(self, hidden_states, conv_module):
        self.left_pad = conv_module.left_pad
        self.in_channels = conv_module.in_channels
        self.cache = torch.zeros(
            hidden_states.shape[0],
            self.in_channels,
            self.left_pad,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.cache)

        self.is_initialized = True

    def update(self, hidden_states, conv_module=None):
        if not self.is_initialized and conv_module is not None:
            self.lazy_initialization(hidden_states, conv_module)
        elif not self.is_initialized:
            raise ValueError(
                "AeroRealtimeAudioConv1dCacheLayer is not initialized. Make sure to provide conv_module to the update method."
            )

        # get the padding states
        if self.left_pad > 0:
            shortfall = max(0, self.left_pad - hidden_states.shape[-1])
            if shortfall > 0:
                padding_states = torch.cat([self.cache[:, :, -shortfall:], hidden_states], dim=-1)
            else:
                padding_states = hidden_states[:, :, -self.left_pad :]
        else:
            padding_states = torch.empty(
                hidden_states.shape[0], self.in_channels, 0, dtype=hidden_states.dtype, device=hidden_states.device
            )

        current_cache = self.cache.clone()
        self.cache.copy_(padding_states)

        return current_cache


class AeroRealtimeAudioConv1dPaddingCache:
    def __init__(self):
        self.layers = {}

    def update(self, hidden_states, cache_key, conv_module):
        if cache_key not in self.layers:
            self.layers[cache_key] = AeroRealtimeAudioConv1dCacheLayer()

        padding_states = self.layers[cache_key].update(hidden_states, conv_module)
        padded_hidden_states = torch.cat([padding_states, hidden_states], dim=-1)
        return padded_hidden_states


@dataclass
class AeroRealtimeAudioEncoderOutput(BaseModelOutputWithPast):
    padding_cache: AeroRealtimeAudioConv1dPaddingCache | None = None


class AeroRealtimeAudioRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: AeroRealtimeAudioEncoderConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: AeroRealtimeAudioEncoderConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class AeroRealtimeAudioConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        cache_key: str,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        padding_mode: str = "causal",
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.cache_key = cache_key
        if padding_mode not in ("causal", "symmetric"):
            raise ValueError(f"padding_mode must be 'causal' or 'symmetric', got {padding_mode}")
        self.padding_mode_ = padding_mode

    @cached_property
    def left_pad(self):
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(
        self,
        x: torch.Tensor,
        padding_cache: AeroRealtimeAudioConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        if self.padding_mode_ == "symmetric":
            # Symmetric padding ignores padding_cache (chunks are independent).
            pad = self.left_pad // 2
            x = nn.functional.pad(x, (pad, self.left_pad - pad))
        elif padding_cache is not None:
            x = padding_cache.update(x, self.cache_key, self)
        else:
            x = nn.functional.pad(x, (self.left_pad, 0))

        return super().forward(x)


@use_kernel_forward_from_hub("RMSNorm")
class AeroRealtimeAudioRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        AeroRealtimeAudioRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _aero_audio_build_local_additive_mask(
    q_len: int,
    k_len: int,
    window_left: int,
    window_right: int,
    key_padding_mask: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    q_idx = torch.arange(q_len, device=device)
    k_idx = torch.arange(k_len, device=device)
    delta = k_idx[None, :] - q_idx[:, None]
    left_ok = torch.ones_like(delta, dtype=torch.bool) if window_left < 0 else (delta >= -window_left)
    right_ok = torch.ones_like(delta, dtype=torch.bool) if window_right < 0 else (delta <= window_right)
    allow = left_ok & right_ok
    if key_padding_mask is not None:
        allow = allow[None] & key_padding_mask.to(torch.bool)[:, None, :]
    else:
        allow = allow[None]
    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros_like(allow, dtype=dtype).masked_fill(~allow, neg_inf)
    return mask.unsqueeze(1)


def _aero_audio_eager_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    window_left: int,
    window_right: int,
    **kwargs,
):
    _, _, q_len, _ = query.shape
    k_len = key.shape[-2]
    key_rep = repeat_kv(key, module.num_key_value_groups)
    value_rep = repeat_kv(value, module.num_key_value_groups)

    if bool(getattr(module.config, "is_causal", False)):
        window_right = 0 if window_right < 0 else min(window_right, 0)

    local_mask = _aero_audio_build_local_additive_mask(
        q_len=q_len,
        k_len=k_len,
        window_left=window_left,
        window_right=window_right,
        key_padding_mask=attention_mask if (attention_mask is not None and attention_mask.dim() == 2) else None,
        dtype=query.dtype,
        device=query.device,
    )
    if attention_mask is not None and attention_mask.dim() == 4:
        local_mask = local_mask + attention_mask

    attn_weights = torch.matmul(query, key_rep.transpose(2, 3)) * scaling
    attn_weights = attn_weights + local_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_rep).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _aero_audio_sdpa_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    window_left: int,
    window_right: int,
    **kwargs,
):
    _, _, q_len, _ = query.shape
    k_len = key.shape[-2]
    key_rep = repeat_kv(key, module.num_key_value_groups)
    value_rep = repeat_kv(value, module.num_key_value_groups)

    if bool(getattr(module.config, "is_causal", False)):
        window_right = 0 if window_right < 0 else min(window_right, 0)

    local_mask = _aero_audio_build_local_additive_mask(
        q_len=q_len,
        k_len=k_len,
        window_left=window_left,
        window_right=window_right,
        key_padding_mask=attention_mask if (attention_mask is not None and attention_mask.dim() == 2) else None,
        dtype=query.dtype,
        device=query.device,
    )
    if attention_mask is not None and attention_mask.dim() == 4:
        local_mask = local_mask + attention_mask

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key_rep,
        value_rep,
        attn_mask=local_mask,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
        is_causal=False,
    )
    return attn_output.transpose(1, 2).contiguous(), None


def _aero_audio_fa_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    window_left: int,
    window_right: int,
    **kwargs,
):
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    bsz, q_len, _, _ = q.shape

    causal = bool(getattr(module.config, "is_causal", False)) or (q_len == k.shape[1])
    window = (window_left, window_right)

    if attention_mask is None or attention_mask.dim() != 2:
        attn_output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout if module.training else 0.0,
            softmax_scale=scaling,
            causal=causal,
            window_size=window,
        )
    else:
        q_unpad, indices_q, cu_q, max_q, *_ = unpad_input(q, attention_mask)
        k_unpad, _, cu_k, max_k, *_ = unpad_input(k, attention_mask)
        v_unpad, _, _, _, *_ = unpad_input(v, attention_mask)
        out_unpad = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            dropout_p=dropout if module.training else 0.0,
            softmax_scale=scaling,
            causal=causal,
            window_size=window,
        )
        attn_output = pad_input(out_unpad, indices_q, bsz, q_len)
    return attn_output, None


AERO_AUDIO_ATTENTION_FORWARDS = {
    "eager": _aero_audio_eager_forward,
    "sdpa": _aero_audio_sdpa_forward,
    "flash_attention_2": _aero_audio_fa_forward,
}


@use_kernelized_func(apply_rotary_pos_emb)
class AeroRealtimeAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=bool(getattr(config, "k_proj_bias", False)),
        )
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = AERO_AUDIO_ATTENTION_FORWARDS.get(
            getattr(self.config, "_attn_implementation", "eager"), _aero_audio_eager_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            window_left=int(getattr(self.config, "attention_window_left", -1)),
            window_right=int(getattr(self.config, "attention_window_right", -1)),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AeroRealtimeAudioMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class AeroRealtimeAudioGeluMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[config.activation_function]

    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))


def _build_norm(config) -> nn.Module:
    norm_type = getattr(config, "norm_type", "rms_norm")
    if norm_type == "rms_norm":
        return AeroRealtimeAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    if norm_type == "layer_norm":
        return nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
    raise ValueError(f"Unknown norm_type: {norm_type!r}")


def _build_mlp(config) -> nn.Module:
    mlp_type = getattr(config, "mlp_type", "swiglu")
    if mlp_type == "swiglu":
        return AeroRealtimeAudioMLP(config)
    if mlp_type == "gelu":
        return AeroRealtimeAudioGeluMLP(config)
    raise ValueError(f"Unknown mlp_type: {mlp_type!r}")


class AeroRealtimeAudioEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = AeroRealtimeAudioConv1d(
            config.num_mel_bins,
            config.hidden_size,
            kernel_size=3,
            cache_key="conv1",
            padding_mode=config.conv_padding,
        )
        self.conv2 = AeroRealtimeAudioConv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=3,
            stride=2,
            cache_key="conv2",
            padding_mode=config.conv_padding,
        )

    def forward(self, input_features, padding_cache=None):
        inputs_embeds = nn.functional.gelu(self.conv1(input_features, padding_cache=padding_cache))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds, padding_cache=padding_cache))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        return inputs_embeds


class AeroRealtimeAudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = AeroRealtimeAudioAttention(config, layer_idx)
        self.self_attn_layer_norm = _build_norm(config)
        self.activation_fn = ACT2FN[config.activation_function]
        self.final_layer_norm = _build_norm(config)
        self.mlp = _build_mlp(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class AeroRealtimeAudioPreTrainedModel(PreTrainedModel):
    config: AeroRealtimeAudioEncoderConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = None
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_attention_backend = True
    # TODO: @eustlb, this should be enabled soon
    _can_compile_fullgraph = False

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)


class AeroRealtimeAudioEncoder(AeroRealtimeAudioPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`AeroRealtimeAudioEncoderLayer`].

    Args:
        config: AeroRealtimeAudioEncoderConfig
    """

    config: AeroRealtimeAudioEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["AeroRealtimeAudioEncoderLayer"]
    _can_record_outputs = {
        "attentions": AeroRealtimeAudioAttention,
        "hidden_states": AeroRealtimeAudioEncoderLayer,
    }

    def __init__(self, config):
        super().__init__(config)
        self.embedder = AeroRealtimeAudioEmbedder(config)
        self.layers = nn.ModuleList(
            [AeroRealtimeAudioEncoderLayer(config, layer_idx) for layer_idx in range(config.encoder_layers)]
        )
        self.norm = _build_norm(config)
        self.rotary_emb = AeroRealtimeAudioRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        padding_cache: AeroRealtimeAudioConv1dPaddingCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        use_padding_cache: bool | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        padding_cache (`AeroRealtimeAudioConv1dPaddingCache`, *optional*):
            Cache for padding in convolutional layers to maintain state across streaming chunks.
        use_padding_cache (`bool`, *optional*):
            Whether to use the padding cache.
        """
        if (input_features is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_features or inputs_embeds")

        if use_padding_cache and padding_cache is None:
            padding_cache = AeroRealtimeAudioConv1dPaddingCache()

        if inputs_embeds is None:
            inputs_embeds = self.embedder(input_features, padding_cache)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return AeroRealtimeAudioEncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
        )
