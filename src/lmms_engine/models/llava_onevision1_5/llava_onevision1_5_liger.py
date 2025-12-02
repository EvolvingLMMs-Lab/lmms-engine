from typing import Optional, Union

import torch
from transformers.cache_utils import Cache

from .modeling_llavaonevision1_5 import (
    LLaVAOneVision1_5_CausalLMOutputWithPast,
    LLaVAOneVision1_5_ForConditionalGeneration,
)

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except:
    print("Liger Kernel is not installed, pip install liger-kernel to use this patch")

from ..sequence_packing_utils import _unpad_input


def forward(
    self: LLaVAOneVision1_5_ForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    use_rmpad: bool = False,
    **kwargs,
) -> Union[tuple, LLaVAOneVision1_5_CausalLMOutputWithPast]:
    r"""
    pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size))`, *optional*):
        The tensors corresponding to the input videos. Pixel values can be obtained using [`AutoImageProcessor`].
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in the language model.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in the language model.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
        or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only
        computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoModelForCausalLM, AutoProcessor

    >>> model = AutoModelForCausalLM.from_pretrained("Deep-VLM/LLaVAOV1.5-4b", trust_remote_code=True)
    >>> processor = AutoProcessor.from_pretrained("Deep-VLM/LLaVAOV1.5-4b", trust_remote_code=True)

    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": [
    ...             {"type": "image"},
    ...             {"type": "text", "text": "What is shown in this image?"},
    ...         ],
    ...     },
    ... ]
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> vision_infos = processor.process_vision_info(images=[image])
    >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "The image shows a street scene with a red stop sign in the foreground ..."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        rope_deltas=rope_deltas,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )
    if use_rmpad:
        input_ids, indices, cu_seq_lens, max_seqlen_in_batch = _unpad_input(input_ids, attention_mask)
        word_idx = indices.long()
        seq_lens = cu_seq_lens.long()

    hidden_states = outputs[0]

    logits = None
    loss = None
    # if in training mode, don't materialize logits
    if self.training and (labels is not None):
        if use_rmpad:
            labels = labels.view(-1)[word_idx.long()]
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
        shift_hidden_states = shift_hidden_states.view(-1, self.language_model.config.hidden_size)
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

    return LLaVAOneVision1_5_CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )
