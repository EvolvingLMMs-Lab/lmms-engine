"""RMPad + LigerCE forward for AeroRealtime.

Mirrors what ``qwen3_vl_ops.model_forward`` does for ``Qwen3VLModel``:
  1. Compute position_ids via ``qwen3_vl_get_rope_index``
  2. Unpad inputs_embeds, position_ids, labels with ``_unpad_input``
  3. Pass ``indices`` and ``cu_seq_lens`` to the language model
  4. Use ``LigerFusedLinearCrossEntropyLoss`` for memory-efficient loss

This function replaces ``AeroRealtimeForConditionalGeneration.forward``
when rmpad is enabled via the monkey patch system.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.utils import is_flash_attn_2_available

from lmms_engine.parallel.sequence_parallel.ulysses import (
    calculate_seq_len_per_rank,
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    pad_to_max_across_ranks,
    slice_input_tensor,
    ulysses_pad,
)

from ..common_ops.rope import qwen3_vl_get_rope_index
from ..sequence_packing_utils import _unpad_input

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, rearrange

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )

    _HAS_LIGER = True
except Exception:
    _HAS_LIGER = False


def aero_realtime_lce_forward(
    self,  # AeroRealtimeForConditionalGeneration
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
):
    """RMPad-aware forward for AeroRealtime with LigerCE loss.

    Same pipeline as the original forward (embed → scatter vision → add audio
    on audio token positions — realtime conditioning lives on the audio
    timeline).
    Adds:
    - Proper mrope position_ids via ``qwen3_vl_get_rope_index``
    - Unpadding of inputs_embeds/position_ids/labels before the language model
    - ``LigerFusedLinearCrossEntropyLoss`` instead of materializing full logits
    """
    from .modeling_aero_realtime import AeroRealtimeCausalLMOutputWithPast

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    sp_size = get_ulysses_sequence_parallel_world_size()

    # ---- 1. Embedding (same as original) ----
    embed_ids = text_stream_ids if text_stream_ids is not None else input_ids

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(embed_ids)

    # Keep original input_ids for mask computation (before unpadding)
    original_input_ids = input_ids
    batch_size, seq_length = original_input_ids.shape

    # ---- 2. Image features — scatter (same as original) ----
    if pixel_values is not None:
        image_features = self.get_vision_features(pixel_values, grid_thw=image_grid_thw)
        image_mask = original_input_ids == self.config.image_token_index
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

    # ---- 3. Video features ----
    video_features = None
    if pixel_values_videos is not None:
        video_features = self.get_vision_features(pixel_values_videos, grid_thw=video_grid_thw)

    # ---- 4. Audio features ----
    audio_features = None
    audio_features_flat = None
    if input_features is not None:
        audio_features, audio_output_lengths = self.get_audio_features(
            input_features, audio_attention_mask=audio_attention_mask
        )
        if audio_output_lengths is not None:
            audio_features_flat = self._unpad_audio_features(audio_features, audio_output_lengths)
        else:
            audio_features_flat = audio_features.reshape(-1, audio_features.shape[-1])

    # ---- 5. Scatter video features and add audio features ----

    # 5a. Scatter video features at video_token_index positions
    if video_features is not None:
        video_mask = original_input_ids == self.config.video_token_index
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

    # 5b. Add audio features at audio_token_index positions
    if audio_features_flat is not None:
        audio_mask = original_input_ids == self.config.audio_token_index
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

    # ==================================================================
    # RMPad: unpad + position_ids (mirrors qwen3_vl_ops.model_forward)
    # ==================================================================

    # 7a. Compute position_ids using qwen3_vl_get_rope_index
    # The function expects self.config to have image_token_id, video_token_id,
    # vision_start_token_id, and vision_config.spatial_merge_size.
    if position_ids is None:
        position_ids, rope_deltas = qwen3_vl_get_rope_index(
            self,
            original_input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask=attention_mask,
        )

    # 7b. Unpad inputs_embeds
    inputs_embeds, indices, cu_seq_lens, _ = _unpad_input(inputs_embeds, attention_mask=attention_mask)

    # 7c. Unpad position_ids: [3, B, S] -> index by valid positions -> [3, 1, total_tokens]
    position_ids = (
        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
    )

    # Qwen3VLTextModel's Ulysses wrapper pads/slices inputs_embeds before the
    # text forward. Pad global position_ids to the same padded length so RoPE
    # cos/sin length matches q/k after all-to-all when total tokens is odd.
    if sp_size > 1:
        dummy_ids = torch.zeros(
            (1, inputs_embeds.shape[0]),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        _, position_ids, _ = ulysses_pad(dummy_ids, position_ids, sp_size=sp_size)

    # 7d. Unpad labels
    if labels is not None:
        labels_unpad = labels.view(-1)[indices]
        if sp_size > 1:
            pad_size = (sp_size - labels_unpad.shape[0] % sp_size) % sp_size
            if pad_size > 0:
                labels_unpad = torch.nn.functional.pad(labels_unpad, (0, pad_size), value=-100)
            labels_unpad = slice_input_tensor(labels_unpad, dim=0, padding=False)
    else:
        labels_unpad = None

    # ---- 8. Language model forward ----
    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        indices=indices,
        cu_seq_lens=cu_seq_lens,
    )

    hidden_states = outputs.last_hidden_state

    # ---- 9. Loss computation ----
    loss = None
    logits = None

    if labels_unpad is not None:
        # Shift per sequence (rmpad: sequences are packed, can't just shift globally)
        shift_cu_seq_lens = cu_seq_lens
        if sp_size > 1:
            shift_cu_seq_lens = calculate_seq_len_per_rank(cu_seq_lens.tolist())

        shift_hidden_states = []
        shift_labels = []
        for i in range(len(shift_cu_seq_lens) - 1):
            start = shift_cu_seq_lens[i]
            end = shift_cu_seq_lens[i + 1]
            cur_hidden = hidden_states[start:end, :]
            cur_labels = labels_unpad[start:end]
            shift_hidden_states.append(cur_hidden[:-1, :].contiguous())
            shift_labels.append(cur_labels[1:].contiguous())
        shift_hidden_states = torch.cat(shift_hidden_states, dim=0)
        shift_labels = torch.cat(shift_labels, dim=0)

        # Flatten
        shift_hidden_states = shift_hidden_states.view(-1, self.config.text_config.hidden_size)
        shift_labels = shift_labels.view(-1)

        if _HAS_LIGER:
            reduction = "none" if sp_size > 1 else "mean"
            lce = LigerFusedLinearCrossEntropyLoss(reduction=reduction)
            loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
        else:
            logits = self.lm_head(shift_hidden_states)
            reduction = "none" if sp_size > 1 else "mean"
            loss_fct = nn.CrossEntropyLoss(reduction=reduction)
            loss = loss_fct(logits, shift_labels)
            logits = None  # Don't return partial logits

        if sp_size > 1:
            loss, total_padding = pad_to_max_across_ranks(loss, dim=0)
            loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=total_padding)
            num_valid_tokens = (shift_labels != -100).sum().float()
            sp_group = get_ulysses_sequence_parallel_group()
            if sp_group is not None:
                dist.all_reduce(num_valid_tokens, op=dist.ReduceOp.SUM, group=sp_group)
            loss = torch.sum(loss) / (num_valid_tokens + 1e-8)
    else:
        # Inference: materialize logits
        logits = self.lm_head(hidden_states)

    if not return_dict:
        output = (logits,)
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
