# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Llava-Onevision model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import AutoConfig, AutoModel, Qwen2AudioEncoder
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.image_processing_utils import select_best_resolution
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
)
from transformers.utils import add_start_docstrings, logging

from lmms_engine.utils import TrainUtilities

from .configuration_aero import AeroConfig

logger = logging.get_logger(__name__)

AutoConfig.register(
    "qwen2_5_omni_audio_encoder", Qwen2_5OmniAudioEncoderConfig, exist_ok=True
)
AutoModel.register(Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniAudioEncoder)


@dataclass
# Copied from transformers.models.llava_next_video.modeling_llava_next_video.LlavaNextVideoCausalLMOutputWithPast with LlavaNextVideo->LlavaOnevision
class AeroCausalLMOutputWithPast(ModelOutput):
    """
    Base class for LlavaOnevision causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        audio_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor`.
            audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.

    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    audio_hidden_states: Optional[torch.FloatTensor] = None
    flops: Optional[Union[float, int]] = None


def qwen_omni_audio_forward(
    self,
    input_features,
    feature_lens=None,
    aftercnn_lens=None,
):
    chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

    chunk_lengths = torch.tensor(
        [self.n_window * 2] * chunk_num.sum(),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
    chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

    chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
    padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
        chunk_list, chunk_lengths, padding_value=0, padding_side="right"
    )
    padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
    padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

    padded_embed = padded_embed + self.positional_embedding.positional_embedding[
        : padded_embed.shape[1], :
    ].unsqueeze(0).to(padded_embed.dtype)
    hidden_states = padded_embed[padded_mask_after_cnn]
    cu_seqlens = torch.cat(
        (
            torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0),
        )
    ).to(torch.int32)

    for idx, encoder_layer in enumerate(self.layers):
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                encoder_layer.__call__,
                hidden_states,
                cu_seqlens,
            )
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

        hidden_states = layer_outputs[0]

    hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
    token_audio_list = []
    for each_audio_states in hidden_states_list:
        # Remove pooling
        if each_audio_states.shape[0] <= 4:
            padded_values = torch.zeros(
                4 - each_audio_states.shape[0],
                each_audio_states.shape[1],
                dtype=each_audio_states.dtype,
                device=each_audio_states.device,
            )
            each_audio_states = torch.cat([each_audio_states, padded_values], dim=0)
        each_audio_states = self.avg_pooler(
            each_audio_states.transpose(0, 1)
        ).transpose_(0, 1)
        each_audio_states = self.ln_post(each_audio_states)
        each_audio_states = self.proj(each_audio_states)
        token_audio_list.append(each_audio_states)
    token_audio = torch.cat(token_audio_list, dim=0)
    return BaseModelOutput(last_hidden_state=token_audio)


class AeroAudioMultiModalProjector(nn.Module):
    def __init__(self, config: AeroConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.audio_config.d_model, config.text_config.hidden_size, bias=True
        )

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


class AeroQwen2_5OmniAudioProjector(nn.Module):
    def __init__(self, config: AeroConfig):
        super().__init__()
        self.act_fun = ACT2FN["gelu"]
        self.linear = nn.Linear(
            config.audio_config.output_dim, config.text_config.hidden_size, bias=True
        )

    def forward(self, audio_features):
        hidden_states = self.linear(self.act_fun(audio_features))
        return hidden_states


PROJECTOR_MAP = {
    "qwen2_audio_encoder": AeroAudioMultiModalProjector,
    "qwen2_5_omni_audio_encoder": AeroQwen2_5OmniAudioProjector,
}


class AeroPreTrainedModel(PreTrainedModel):
    config_class = AeroConfig
    base_model_prefix = "language_model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_static_cache = (
        False  # Qwen2 doesn't but llava has no reasons to not support
    )
    _supports_quantized_cache = True
    _supports_sdpa = True

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextPreTrainedModel._init_weights
    def _init_weights(self, module):
        # important: this ported version of LlavaNext isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava_next should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class AeroForConditionalGeneration(AeroPreTrainedModel, GenerationMixin):
    def __init__(self, config: AeroConfig):
        super().__init__(config)

        self.audio_tower_type = config.audio_config.model_type
        # Patch the forward function of the audio tower to remove pooling
        if self.audio_tower_type == "qwen2_5_omni_audio_encoder":
            Qwen2_5OmniAudioEncoder.forward = qwen_omni_audio_forward
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.audio_modal_projector = PROJECTOR_MAP[self.audio_tower_type](config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.post_init()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_decoder
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_decoder
    def get_decoder(self):
        return self.language_model.get_decoder()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.tie_weights
    def tie_weights(self):
        return self.language_model.tie_weights()

    def prepare_inputs_for_qwen_audio_encoder(
        self,
        audio_values: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        audio_feat_lengths: torch.FloatTensor,
        audio_output_lengths: torch.FloatTensor,
    ):
        batch_size, _, max_mel_seq_len = audio_values.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(
                0,
                max_seq_len,
                dtype=audio_feat_lengths.dtype,
                device=audio_feat_lengths.device,
            )
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype,
            device=self.audio_tower.conv1.weight.device,
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        inputs = {
            "input_features": audio_values,
            "attention_mask": audio_attention_mask,
        }
        return inputs

    def prepare_inputs_for_qwen_5_omni_audio_encoder(
        self,
        audio_values: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        audio_feat_lengths: torch.FloatTensor,
        audio_output_lengths: torch.FloatTensor,
    ):
        audio_feature_lengths = torch.sum(audio_attention_mask, dim=1)
        input_features = audio_values.permute(0, 2, 1)[
            audio_attention_mask.bool()
        ].permute(1, 0)
        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else audio_attention_mask.sum(-1)
        )
        inputs = {
            "feature_lens": feature_lens,
            "aftercnn_lens": audio_feat_lengths,
            "input_features": input_features,
        }
        return inputs

    def prepare_scattered_audio_values(
        self,
        audio_features,
        audio_output_lengths,
    ):
        # Audio feature is in (bs, max_seq_len, hidden_size)
        # If directly masked scatter, the embed will be place one by one (order is incorret)
        # We remove the padded values first
        unpadded_audio_features = [
            audio_feat[:audio_output_length]
            for audio_feat, audio_output_length in zip(
                audio_features, audio_output_lengths
            )
        ]
        # Concat the audio features
        # Should exactly have audio_mask.sum() values
        unpadded_audio_features = torch.concatenate(unpadded_audio_features, dim=0)
        return unpadded_audio_features

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
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
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
            audio_features = audio_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        audio_values=None,
        audio_attention_mask=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            model_inputs["audio_values"] = audio_values
            model_inputs["audio_attention_mask"] = audio_attention_mask

        return model_inputs

    def calc_gpt_flops(self, attention_mask, num_audio_tokens):
        tokens_count = torch.sum(attention_mask != 0).item()
        lm_flops = TrainUtilities.get_decoder_flops(
            num_layers=self.config.text_config.num_hidden_layers,
            hidden_size=self.config.text_config.hidden_size,
            vocab_size=self.config.text_config.vocab_size,
            seq_len=tokens_count,
            ffn_hidden_size=self.config.text_config.intermediate_size,
            num_key_value_heads=self.config.text_config.num_key_value_heads,
            num_heads=self.config.text_config.num_attention_heads,
            batch_size=1,
        )
        audio_encoder_flops = TrainUtilities.get_attn_flops(
            num_layers=self.config.audio_config.encoder_layers,
            hidden_size=self.config.audio_config.d_model,
            num_heads=self.config.audio_config.encoder_attention_heads,
            seq_len=num_audio_tokens,
            num_key_value_heads=None,
            ffn_hidden_size=self.config.audio_config.encoder_ffn_dim,
        )
        projector_flops = (
            self.config.audio_config.d_model * self.config.text_config.hidden_size * 6
        )
        flops = lm_flops + audio_encoder_flops + projector_flops
        return flops
