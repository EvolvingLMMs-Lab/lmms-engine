from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerCausalLMOutputWithPast,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    load_balancing_loss_func,
)
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
from lmms_engine.utils import Logging

from ..sequence_packing_utils import _unpad_input

if is_flash_attn_2_available():
    try:
        from einops import rearrange
        from flash_attn.bert_padding import index_first_axis
    except:
        raise ModuleNotFoundError("flash_attn is not available. Please install it via `pip install flash_attn`.")

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except:
    print("Liger Kernel is not installed, pip install liger-kernel to use this patch")


def lce_forward(
    self: Qwen3OmniMoeThinkerForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
    audio_feature_lengths: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    video_second_per_grid: Optional[torch.Tensor] = None,
    use_audio_in_video: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    use_rmpad: Optional[bool] = False,
    **kwargs,
) -> Union[Tuple, Qwen3OmniMoeThinkerCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else getattr(self.config.text_config, "output_router_logits", True)
    )

    tokens_count = attention_mask.sum().item()
    n_image_tokens = (
        (input_ids == self.config.image_token_id).sum().item() if hasattr(self.config, "image_token_id") else 0
    )
    n_video_tokens = (
        (input_ids == self.config.video_token_id).sum().item() if hasattr(self.config, "video_token_id") else 0
    )
    n_audio_tokens = (
        (input_ids == self.config.audio_token_id).sum().item() if hasattr(self.config, "audio_token_id") else 0
    )
    visual_tokens = n_image_tokens + n_video_tokens

    cu_seq_lens = None
    indices = None
    original_input_ids = None
    if use_rmpad and attention_mask is not None:
        # input_ids is 2D [batch, seq_len]
        original_input_ids = input_ids
        # unpad input_ids: 2D [batch, seq_len] -> 1D [total_non_pad_tokens]
        input_ids, indices, cu_seq_lens, _ = _unpad_input(input_ids, attention_mask=attention_mask)
        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                batch_size, seq_length = original_input_ids.shape
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                # get_rope_index expects RAW audio feature lengths before any downsampling.
                # Processor provides audio_feature_lengths after first downsampling.
                # Reconstruct raw length: raw = (audio_feature_lengths - 1) * 2 + 1
                if audio_feature_lengths is not None:
                    audio_raw_lengths = (audio_feature_lengths - 1) * 2 + 1
                else:
                    audio_raw_lengths = None

                position_ids, rope_deltas = self.get_rope_index(
                    original_input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_raw_lengths,
                    video_second_per_grid,
                )

                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = original_input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=original_input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        position_ids = (
            index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
        )

        if get_ulysses_sequence_parallel_world_size() > 1:
            sp_size = get_ulysses_sequence_parallel_world_size()
            input_ids, position_ids, pad_size = ulysses_pad(
                input_ids.unsqueeze(0),
                position_ids,
                sp_size=sp_size,
            )
            input_ids = input_ids.squeeze(0)
            actual_tokens = input_ids.shape[0]
            # update the actual seg_len if pad is used
            if cu_seq_lens is not None and len(cu_seq_lens) > 0:
                cu_seq_lens = torch.tensor(
                    [0] + [actual_tokens] * (len(cu_seq_lens) - 1),
                    dtype=cu_seq_lens.dtype,
                    device=cu_seq_lens.device,
                )
    if inputs_embeds is None and input_ids is not None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
    if input_features is not None:
        audio_features = self.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        n_audio_tokens_check = (input_ids == self.config.audio_token_id).sum().item()
        n_audio_features = audio_features.shape[0]
        if n_audio_tokens_check != n_audio_features:
            raise ValueError(
                f"Audio features and audio tokens do not match: "
                f"tokens: {n_audio_tokens_check}, features {n_audio_features}. "
                f"This indicates a mismatch between the audio encoder output and placeholder tokens."
            )
        audio_mask = (
            (input_ids == self.config.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        n_image_tokens_check = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens_check != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens_check}, features {n_image_features}"
            )

        image_mask = (
            (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        n_video_tokens_check = (input_ids == self.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens_check != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens_check}, features {n_video_features}"
            )

        video_mask = (
            (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    outputs = self.model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        rope_deltas=rope_deltas,
        use_audio_in_video=use_audio_in_video,
        video_second_per_grid=video_second_per_grid,
        cu_seq_lens=cu_seq_lens,
        indices=indices,
        output_router_logits=output_router_logits,
    )

    seq_lens = outputs.get("seq_lens", None)
    word_idx = outputs.get("word_idx", None)
    hidden_states = outputs[0]
    loss = None
    logits = None

    if labels is not None and word_idx is not None:
        labels_unpad = labels.view(-1)[word_idx.long()]
        if get_ulysses_sequence_parallel_world_size() > 1:
            seq_lens = calculate_seq_len_per_rank(seq_lens.tolist()) if seq_lens is not None else None
            labels_unpad = slice_input_tensor(labels_unpad, dim=0, padding=True)
        labels = labels_unpad

    if labels is not None:
        if use_rmpad and seq_lens is not None:
            shift_hidden_states = []
            shift_labels = []
            for i in range(len(seq_lens) - 1):
                cur_hidden_states = hidden_states[seq_lens[i] : seq_lens[i + 1], :]
                cur_labels = labels[seq_lens[i] : seq_lens[i + 1]]
                cur_shift_hidden_states = cur_hidden_states[:-1, :].contiguous()
                cur_shift_labels = cur_labels[1:].contiguous()
                shift_hidden_states.append(cur_shift_hidden_states)
                shift_labels.append(cur_shift_labels)
            shift_hidden_states = torch.cat(shift_hidden_states, dim=0)
            shift_labels = torch.cat(shift_labels, dim=0)
        else:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        hidden_size = (
            self.config.text_config.hidden_size if hasattr(self.config, "text_config") else self.config.hidden_size
        )
        shift_hidden_states = shift_hidden_states.view(-1, hidden_size)
        shift_labels = shift_labels.view(-1)

        reduction = "sum" if "num_items_in_batch" in kwargs else "mean"
        # If using sp, we follow the loss calculation in verl, get loss for each token, then gather and sum them up
        if get_ulysses_sequence_parallel_world_size() > 1:
            reduction = "none"
        lce = LigerFusedLinearCrossEntropyLoss(reduction=reduction)
        loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
        if get_ulysses_sequence_parallel_world_size() > 1:
            # Pad to max size across ranks, then gather and unpad
            loss, total_padding = pad_to_max_across_ranks(loss, dim=0)
            loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=total_padding)
            # Calculate the actual number of valid tokens (non-ignored labels) across all ranks
            # shift_labels shape is (num_tokens,) after flatten, -100 means ignore
            num_valid_tokens = (shift_labels != -100).sum().float()
            # Gather num_valid_tokens across all SP ranks to get the total count
            sp_group = get_ulysses_sequence_parallel_group()
            if sp_group is not None:
                dist.all_reduce(num_valid_tokens, op=dist.ReduceOp.SUM, group=sp_group)
            loss = torch.sum(loss) / (num_valid_tokens + 1e-8)

        if reduction == "sum":
            loss /= kwargs["num_items_in_batch"]

    else:
        logits = self.lm_head(hidden_states)

        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

    # MoE auxiliary loss handling
    aux_loss = None
    router_logits = getattr(outputs, "router_logits", None)
    if output_router_logits and router_logits is not None:
        aux_loss_mask = None if use_rmpad else attention_mask
        aux_loss = load_balancing_loss_func(
            router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            aux_loss_mask,
        )
        if labels is not None and loss is not None:
            # Add auxiliary loss weighted by router_aux_loss_coef
            loss = loss + self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen3OmniMoeThinkerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        aux_loss=aux_loss,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )
