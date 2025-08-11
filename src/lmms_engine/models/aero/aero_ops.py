from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from lmms_engine.models.aero.modeling_aero import AeroCausalLMOutputWithPast

from ..sequence_packing_utils import _unpad_input


def forward(
    self,
    input_ids: torch.LongTensor = None,
    audio_values: torch.FloatTensor = None,
    audio_attention_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int = 0,
) -> Union[Tuple, AeroCausalLMOutputWithPast]:
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
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if input_ids is None:
        assert (
            False
        ), "input_ids is None, please provide input_ids. To use rmpad with kino, please provide input ids. This is only used in training"

    # Unpad the input ids here
    input_ids, indices, cu_seq_lens, _ = _unpad_input(
        input_ids, attention_mask=attention_mask
    )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # Embed audio features
    if audio_values is not None:
        (
            audio_feat_lengths,
            audio_output_lengths,
        ) = self.audio_tower._get_feat_extract_output_lengths(
            audio_attention_mask.sum(-1)
        )
        if self.audio_tower_type == "qwen2_audio_encoder":
            inputs = self.prepare_inputs_for_qwen_audio_encoder(
                audio_values=audio_values,
                audio_attention_mask=audio_attention_mask,
                audio_feat_lengths=audio_feat_lengths,
                audio_output_lengths=audio_output_lengths,
            )
        elif self.audio_tower_type == "qwen2_5_omni_audio_encoder":
            inputs = self.prepare_inputs_for_qwen_5_omni_audio_encoder(
                audio_values=audio_values,
                audio_attention_mask=audio_attention_mask,
                audio_feat_lengths=audio_feat_lengths,
                audio_output_lengths=audio_output_lengths,
            )

        audio_outputs = self.audio_tower(**inputs)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.audio_modal_projector(selected_audio_feature)
        n_audio_tokens = (input_ids == self.config.audio_token_index).sum().item()
        n_audio_features = audio_output_lengths.sum()
        if n_audio_tokens != n_audio_features:
            raise ValueError(
                f"Audio features and image tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
            )
        audio_mask = (
            (input_ids == self.config.audio_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        if self.audio_tower_type == "qwen2_audio_encoder":
            audio_features = self.prepare_scattered_audio_values(
                audio_features, audio_output_lengths
            )
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

    n_audio_tokens = (input_ids == self.config.audio_token_index).sum().item()
    flops = self.calc_gpt_flops(attention_mask, n_audio_tokens)
    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        labels=labels,
        cu_seq_lens=cu_seq_lens,
        indices=indices,
    )

    logits = outputs[0]
    loss = outputs.get("loss", None)
    if labels is not None and loss is None:
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            # we use the input attention mask to shift the logits and labels, because it is 2D.
            # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
            shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(
                logits.device
            )
            shift_logits = logits[..., :-1, :][
                shift_attention_mask.to(logits.device) != 0
            ].contiguous()
            shift_labels = labels[..., 1:][
                shift_attention_mask.to(labels.device) != 0
            ].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).to(shift_logits.device),
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return AeroCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        flops=flops,
        audio_hidden_states=audio_features if audio_values is not None else None,
    )
