from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)

from lmms_engine.parallel.sequence_parallel.ulysses import (
    calculate_seq_len_per_rank,
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    pad_to_max_across_ranks,
    slice_input_tensor,
)

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except:
    print("Liger Kernel is not installed, pip install liger-kernel to use this patch")


def lce_forward(
    self: Qwen2_5_VLForConditionalGeneration,
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
    **kwargs,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    tokens_count = attention_mask.sum().item()
    n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
    n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
    visual_tokens = n_image_tokens + n_video_tokens

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
    # if we are using sequence parallel, we need to slice the hidden states and labels
    labels_unpad = labels.view(-1)[word_idx.long()]
    if get_ulysses_sequence_parallel_world_size() > 1:
        seq_lens = calculate_seq_len_per_rank(seq_lens.tolist()) if seq_lens is not None else None
        labels_unpad = slice_input_tensor(labels_unpad, dim=0, padding=True)
    labels = labels_unpad

    # if in training mode, don't materialize logits
    if labels is not None:
        if use_rmpad:
            # We need to shift the tokens according to seq lens
            # Otherwise, the first labels of the next seq will be the last labels of the current seq
            shift_hidden_states = []
            shift_labels = []
            for i in range(len(seq_lens) - 1):
                cur_hidden_states = hidden_states[seq_lens[i] : seq_lens[i + 1], :]
                cur_shift_hidden_states = cur_hidden_states[:-1, :].contiguous()
                cur_labels = labels[seq_lens[i] : seq_lens[i + 1]]
                cur_shift_labels = cur_labels[1:].contiguous()
                shift_hidden_states.append(cur_shift_hidden_states)
                shift_labels.append(cur_shift_labels)
            shift_hidden_states = torch.cat(shift_hidden_states, dim=0)
            shift_labels = torch.cat(shift_labels, dim=0)
        else:
            # We do the same thing as ForCausalLMLoss but using Liger FLCE

            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        # flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
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

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )
