from typing import Optional, Union

import torch
from transformers.cache_utils import Cache
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionCausalLMOutputWithPast,
    LlavaOnevisionForConditionalGeneration,
)

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except:
    print("Liger Kernel is not installed, pip install liger-kernel to use this patch")

from ..sequence_packing_utils import _unpad_input


def forward(
    self: LlavaOnevisionForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    image_sizes: Optional[torch.LongTensor] = None,
    pixel_values_videos: torch.FloatTensor = None,
    image_sizes_videos: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[Union[int, list[int]]] = None,
    vision_feature_select_strategy: Optional[str] = None,
    vision_aspect_ratio: Optional[str] = None,
    batch_num_images: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    use_rmpad: bool = False,
    **kwargs,
) -> Union[tuple, LlavaOnevisionCausalLMOutputWithPast]:
    r"""
    image_sizes_videos (`torch.LongTensor` of shape `(batch_size, frames, 2)`, *optional*):
        The sizes of the videos in the batch, being (height, width) for each frame in the video.
    vision_aspect_ratio (`str`, *optional*, defaults to `"anyres_max_9"`):
        Aspect ratio used when processong image features. The default value is "anyres_max_9".
    batch_num_images (`torch.LongTensor`, *optional*):
        Number of images in each sample.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> import torch
    >>> from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

    >>> model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype="float16", device_map="cuda:0")
    >>> processor = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

    >>> conversation = [
    ...     {
    ...       "role": "user",
    ...       "content": [
    ...           {"type": "text", "text": "What is shown in this image?"},
    ...           {"type": "image"},
    ...         ],
    ...     },
    ... ]
    >>> prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    >>> image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> raw_image = Image.open(requests.get(image_file, stream=True).raw)
    >>> inputs = processor(text=prompt, images=raw_image, return_tensors='pt').to(0, torch.float16)

    >>> output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    >>> processor.batch_decode(output, skip_special_tokens=True)[0]
    "user\n\nWhat is shown in this image?\nassistant\ncat"
    ```"""
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
    vision_feature_layer = (
        vision_feature_layer
        if vision_feature_layer is not None
        else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )
    vision_aspect_ratio = (
        vision_aspect_ratio
        if vision_aspect_ratio is not None
        else self.config.vision_aspect_ratio
    )

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_sizes=image_sizes,
        image_sizes_videos=image_sizes_videos,
        vision_aspect_ratio=vision_aspect_ratio,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy=vision_feature_select_strategy,
        batch_num_images=batch_num_images,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **kwargs,
    )
    if use_rmpad:
        input_ids, indices, cu_seq_lens, max_seqlen_in_batch = _unpad_input(
            input_ids, attention_mask
        )
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
        shift_hidden_states = shift_hidden_states.view(
            -1, self.language_model.config.hidden_size
        )
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

    return LlavaOnevisionCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        video_hidden_states=outputs.video_hidden_states,
    )
