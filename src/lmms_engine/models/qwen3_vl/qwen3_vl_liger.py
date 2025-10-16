from typing import List, Optional, Tuple, Union

import torch
from transformers import Qwen3VLForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except:
    print("Liger Kernel is not installed, pip install liger-kernel to use this patch")


def qwen3_vl_lce_forward(
    self: Qwen3VLForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    use_rmpad: Optional[bool] = False,
    **kwargs,
) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.

    Example:
        TODO: Add example
    """
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]

    seq_lens = outputs.get("seq_lens", None)
    word_idx = outputs.get("word_idx", None)

    hidden_states = outputs[0]

    loss = None
    logits = None
    # if we are using sequence parallel, we need to slice the hidden states and labels
    labels_unpad = labels.view(-1)[word_idx.long()]
    labels = labels_unpad

    config = getattr(self.config, "text_config", self.config)

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
        shift_hidden_states = shift_hidden_states.view(-1, config.hidden_size)
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
                vocab_size=config.vocab_size,
                **kwargs,
            )

    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        rope_deltas=outputs.rope_deltas,
    )
