from typing import List, Optional, Tuple, Union

import torch
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
)

from lmms_engine.parallel.sequence_parallel.ulysses import (
    calculate_seq_len_per_rank,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except:
    print("Liger Kernel is not installed, pip install liger-kernel to use this patch")


def lce_forward(
    self: Qwen2_5OmniThinkerForConditionalGeneration,
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
    audio_values: Optional[torch.FloatTensor] = None,
    audio_attention_mask: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    use_rmpad: Optional[bool] = False,
    freeze_talker: Optional[bool] = False,
    **kwargs,
) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # count tokens
    if attention_mask is not None:
        tokens_count = attention_mask.sum().item()
    n_image_tokens = (input_ids == self.config.image_token_id).sum().item() if hasattr(self.config, 'image_token_id') else 0
    n_video_tokens = (input_ids == self.config.video_token_id).sum().item() if hasattr(self.config, 'video_token_id') else 0
    n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item() if hasattr(self.config, 'audio_token_id') else 0
    visual_tokens = n_image_tokens + n_video_tokens

    # Get audio start/end token ids for talker freezing
    audio_start_token_id = getattr(self.config, 'audio_start_token_id', None)
    audio_end_token_id = getattr(self.config, 'audio_end_token_id', None)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        audio_values=audio_values,
        audio_attention_mask=audio_attention_mask,
    )
    seq_lens = outputs.get("seq_lens", None)
    word_idx = outputs.get("word_idx", None)
    hidden_states = outputs[0]
    loss = None
    logits = None
    if word_idx is not None:
        labels_unpad = labels.view(-1)[word_idx.long()]
        if get_ulysses_sequence_parallel_world_size() > 1:
            seq_lens = (
                calculate_seq_len_per_rank(seq_lens.tolist())
                if seq_lens is not None
                else None
            )
            labels_unpad = slice_input_tensor(labels_unpad, dim=0, padding=True)
        labels = labels_unpad

    # if in training mode, don't materialize logits
    if labels is not None:
        if use_rmpad and seq_lens is not None:
            # We need to shift the tokens according to seq lens
            # Otherwise, the first labels of the next seq will be the last labels of the current seq
            shift_hidden_states = []
            shift_labels = []
            for i in range(len(seq_lens) - 1):
                cur_hidden_states = hidden_states[seq_lens[i] : seq_lens[i + 1], :]
                cur_shift_hidden_states = cur_hidden_states[:-1, :].contiguous()
                cur_labels = labels[seq_lens[i] : seq_lens[i + 1]]
                cur_shift_labels = cur_labels[1:].contiguous()

                # Handle talker freezing - mask out audio generation tokens
                if freeze_talker and audio_start_token_id is not None and audio_end_token_id is not None:
                    # Find audio generation regions (between audio_start and audio_end tokens)
                    audio_start_mask = (cur_labels[:-1] == audio_start_token_id)
                    audio_end_mask = (cur_labels[:-1] == audio_end_token_id)

                    # Create mask for audio generation tokens
                    in_audio = False
                    audio_mask = torch.zeros_like(cur_shift_labels, dtype=torch.bool)
                    for idx in range(len(cur_shift_labels)):
                        if idx > 0 and audio_start_mask[idx-1]:
                            in_audio = True
                        if in_audio:
                            audio_mask[idx] = True
                        if idx > 0 and audio_end_mask[idx-1]:
                            in_audio = False

                    # Set audio generation tokens to -100 to ignore in loss
                    cur_shift_labels = torch.where(audio_mask, -100, cur_shift_labels)

                shift_hidden_states.append(cur_shift_hidden_states)
                shift_labels.append(cur_shift_labels)
            shift_hidden_states = torch.cat(shift_hidden_states, dim=0)
            shift_labels = torch.cat(shift_labels, dim=0)
        else:
            # We do the same thing as ForCausalLMLoss but using Liger FLCE
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Handle talker freezing for non-rmpad case
            if freeze_talker and audio_start_token_id is not None and audio_end_token_id is not None:
                # Flatten for easier processing
                flat_labels = labels.view(-1)
                audio_start_positions = (flat_labels == audio_start_token_id).nonzero(as_tuple=True)[0]
                audio_end_positions = (flat_labels == audio_end_token_id).nonzero(as_tuple=True)[0]

                # Create mask for audio generation tokens
                audio_mask = torch.zeros_like(shift_labels.view(-1), dtype=torch.bool)
                for start_pos, end_pos in zip(audio_start_positions, audio_end_positions):
                    if end_pos > start_pos:
                        # Mask the region between start and end (exclusive)
                        audio_mask[start_pos:end_pos] = True

                # Reshape and apply mask
                audio_mask = audio_mask.view(shift_labels.shape)
                shift_labels = torch.where(audio_mask, -100, shift_labels)

        # flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
        shift_labels = shift_labels.view(-1)

        reduction = "sum" if "num_items_in_batch" in kwargs else "mean"
        lce = LigerFusedLinearCrossEntropyLoss(reduction=reduction)

        loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
        if reduction == "sum":
            loss /= kwargs["num_items_in_batch"]

    else:  # if in inference mode materialize logits
        logits = self.lm_head(hidden_states)
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5OmniThinkerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )