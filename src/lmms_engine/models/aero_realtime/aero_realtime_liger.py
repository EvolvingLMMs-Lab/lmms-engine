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


def _get_audio_features_rmpad(self, input_features: torch.Tensor, audio_attention_mask: Optional[torch.Tensor]):
    audio_outputs = self.audio_tower(
        input_features=input_features,
        attention_mask=audio_attention_mask,
    )
    if isinstance(audio_outputs, torch.Tensor):
        audio_hidden_states = audio_outputs
    else:
        audio_hidden_states = audio_outputs.last_hidden_state

    df = self.config.downsample_factor
    H = self.config.audio_hidden_size
    audio_output_lengths = None
    if audio_attention_mask is not None:
        audio_output_lengths = audio_attention_mask.sum(-1) // df

    if audio_hidden_states.dim() == 2:
        if audio_attention_mask is None:
            raise ValueError("Packed audio hidden states require audio_attention_mask.")

        B, max_T = audio_attention_mask.shape
        total_valid = audio_hidden_states.shape[0]

        # Fast path: every row of audio_attention_mask is fully valid AND
        # max_T is divisible by df. This is the dominant case under chunked
        # streaming training (each chunk produces exactly df encoder frames),
        # and lets us skip the per-segment python loop + cat in the slow path.
        if max_T % df == 0 and total_valid == B * max_T:
            audio_hidden_states = audio_hidden_states.reshape(-1, H * df)
        else:
            # Slow path: ragged segments. Build a fully-GPU gather index that
            # selects only the `usable_len = (length // df) * df` rows from
            # each segment, then reshape in one shot. Avoids the python loop
            # + per-chunk slice + cat over potentially thousands of segments.
            lengths = audio_attention_mask.sum(-1)
            usable_lens = (lengths // df) * df
            total_usable = int(usable_lens.sum().item())
            if total_usable == 0:
                audio_hidden_states = audio_hidden_states.new_empty((0, H * df))
            else:
                in_starts = torch.cumsum(lengths, dim=0) - lengths
                out_starts = torch.cumsum(usable_lens, dim=0) - usable_lens
                flat = torch.arange(total_usable, device=lengths.device)
                seg = torch.searchsorted(
                    out_starts[1:].contiguous() if B > 1 else out_starts.new_zeros(0),
                    flat,
                    right=True,
                )
                gather_idx = flat + (in_starts - out_starts)[seg]
                audio_hidden_states = audio_hidden_states.index_select(0, gather_idx)
                audio_hidden_states = audio_hidden_states.reshape(-1, H * df)
    else:
        seq_len = audio_hidden_states.shape[1]
        usable_len = (seq_len // df) * df
        audio_hidden_states = audio_hidden_states[:, :usable_len, :]
        audio_hidden_states = audio_hidden_states.reshape(
            audio_hidden_states.shape[0],
            -1,
            H * df,
        )

    return self.multi_modal_projector(audio_hidden_states), audio_output_lengths


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
    output_last_hidden_state: bool = False,
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

    # Keep original input_ids for mrope + modality placeholder masks.
    original_input_ids = input_ids
    batch_size, seq_length = original_input_ids.shape

    # ==================================================================
    # RMPad first, then modality injection (mirrors qwen3_5_ops.model_forward)
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

    # 7b. Unpad input_ids and text_stream_ids. We scatter modalities on the
    # packed layout, matching qwen3_5_ops.model_forward. This keeps the
    # vit_frame_parallel autograd collective path identical to the base model.
    input_ids, indices, cu_seq_lens, _ = _unpad_input(input_ids, attention_mask=attention_mask)
    if text_stream_ids is not None:
        embed_ids = text_stream_ids.reshape(-1)[indices]
    else:
        embed_ids = input_ids

    # 7c. Unpad position_ids: [3, B, S] -> index by valid positions -> [3, 1, total_tokens]
    position_ids = (
        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
    )

    # Qwen3VLTextModel's Ulysses wrapper pads/slices inputs_embeds before the
    # text forward. We mirror qwen3_5_ops.model_forward: pad the packed seq
    # to a multiple of sp_size and mark the pad span as its own sample in
    # ``cu_seq_lens`` so linear-attn / causal-conv don't leak the pad region
    # back into the real tail sample (full-attn doesn't care because pad
    # tokens are loss-masked anyway).
    pad_size = 0
    if sp_size > 1:
        input_ids, position_ids, pad_size = ulysses_pad(input_ids.unsqueeze(0), position_ids, sp_size=sp_size)
        input_ids = input_ids.squeeze(0)
        if pad_size > 0:
            embed_ids = torch.nn.functional.pad(embed_ids, (0, pad_size), value=self.config.rt_pad_token_index)
            # Mark pad span as its own sample so linear-attn / causal-conv
            # don't see it as a continuation of the last real sample.
            cu_seq_lens = torch.cat([cu_seq_lens, cu_seq_lens.new_tensor([cu_seq_lens[-1].item() + pad_size])])

    # ---- 2. Embedding on packed ids ----
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(embed_ids)
    elif pixel_values is not None or pixel_values_videos is not None or input_features is not None:
        inputs_embeds = inputs_embeds.clone()

    # ---- 3. Image features — scatter into packed embeddings ----
    image_features = None
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
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, image_features.to(inputs_embeds.dtype))

    # ---- 4. Video features — scatter into packed embeddings ----
    video_features = None
    if pixel_values_videos is not None:
        video_features = self.get_vision_features(pixel_values_videos, grid_thw=video_grid_thw)
        video_mask = input_ids == self.config.video_token_index
        n_video_tokens = video_mask.sum().item()
        n_video_features = video_features.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video token count ({n_video_tokens}) does not match " f"video feature count ({n_video_features})."
            )
        video_mask_expanded = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask_expanded, video_features.to(inputs_embeds.dtype))

    # ---- 5. Audio features — add into packed embeddings ----
    audio_features_flat = None
    if input_features is not None:
        audio_features, audio_output_lengths = _get_audio_features_rmpad(self, input_features, audio_attention_mask)
        if audio_features.dim() == 2:
            audio_features_flat = audio_features
        elif audio_output_lengths is not None:
            audio_features_flat = self._unpad_audio_features(audio_features, audio_output_lengths)
        else:
            audio_features_flat = audio_features.reshape(-1, audio_features.shape[-1])
        audio_mask = input_ids == self.config.audio_token_index
        n_audio_tokens = audio_mask.sum().item()
        n_audio_features = audio_features_flat.shape[0]
        if n_audio_tokens != n_audio_features:
            raise ValueError(
                f"Audio token count ({n_audio_tokens}) does not match " f"audio feature count ({n_audio_features})."
            )
        audio_mask_flat = audio_mask.reshape(-1)
        inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        additive = torch.zeros_like(inputs_embeds_flat)
        additive[audio_mask_flat] = audio_features_flat.to(inputs_embeds.dtype)
        inputs_embeds = (inputs_embeds_flat + additive).reshape(inputs_embeds.shape)

    # 7d. Unpad labels
    if labels is not None:
        labels_unpad = labels.view(-1)[indices]
        if sp_size > 1:
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

    gathered_last_hidden_state = None
    gathered_cu_seq_lens = None
    if output_last_hidden_state:
        if sp_size > 1:
            gathered_last_hidden_state = gather_outputs_and_unpad(
                outputs.last_hidden_state, gather_dim=0, unpad_dim=0, padding_size=pad_size
            )
            gathered_cu_seq_lens = cu_seq_lens[:-1] if pad_size > 0 else cu_seq_lens
        else:
            gathered_last_hidden_state = outputs.last_hidden_state
            gathered_cu_seq_lens = cu_seq_lens

    return AeroRealtimeCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        audio_hidden_states=audio_features_flat,
        vision_hidden_states=video_features,
        last_hidden_state=gathered_last_hidden_state,
        cu_seq_lens=gathered_cu_seq_lens,
        indices=indices if output_last_hidden_state else None,
    )
