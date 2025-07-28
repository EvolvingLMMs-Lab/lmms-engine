from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import is_flash_attn_2_available, logging

logger = logging.get_logger(__name__)

from ..sequence_packing_utils import _unpad_input

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
else:
    raise ImportError("Please install flash-attn-2 to use rmpad kernels")


def prepare_causal_attn_mask(
    attention_mask: torch.Tensor,
):
    """
    Prepare Causal mask for audio encoder that fits flash-attention-2

    The original prepare process prepare it in shape (bs, 1, tgt_len, src_len), for example (2, 1, 1500, 1500)
    This is causing the bugs in flash attention. I don't really know why they prepare it in this format. But we put it back in 2d
    and also make it with bool format instead of inf
    """
    # So basically this is a 4-d mask, we squeeze the second dim first
    attention_mask = attention_mask.squeeze(1)
    # The one that equals to 0 is the True, -inf is False
    # Then we find at the col dim to make it 2d
    # lower diagonal
    causal_mask = attention_mask.eq(0).any(1)
    return causal_mask


# Forward function for Qwen2 Audio Encoder
def encoder_forward(
    self,
    input_features,
    attention_mask=None,
    head_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    expected_seq_length = (
        self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
    )
    if input_features.shape[-1] != expected_seq_length:
        raise ValueError(
            f"Qwen2Audio expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
        )

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

    # Ignore copy
    input_features = input_features.to(
        dtype=self.conv1.weight.dtype, device=self.conv1.weight.device
    )

    inputs_embeds = nn.functional.gelu(self.conv1(input_features))
    inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

    inputs_embeds = inputs_embeds.permute(0, 2, 1)
    embed_pos = self.embed_positions.weight

    hidden_states = inputs_embeds + embed_pos
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    attention_mask = prepare_causal_attn_mask(attention_mask)
    hidden_states, indices, cu_seq_lens, _ = _unpad_input(
        hidden_states, attention_mask=attention_mask
    )

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (
            len(self.layers)
        ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        to_drop = False
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:  # skip the layer
                to_drop = True

        # Ignore copy
        if to_drop:
            layer_outputs = (None, None)
        else:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                    output_attentions,
                    cu_seq_lens,
                    use_reentrant=False,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                    cu_seq_lens=cu_seq_lens,
                )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    # Ignore copy
    split_sizes = cu_seq_lens.diff().tolist()
    hidden_states = hidden_states.split(split_sizes)
    # Pad back to (bs, 1500, 1280) so that the pooling can be done
    hidden_states = pad_sequence(hidden_states, batch_first=True, padding_value=0)
    hidden_states = hidden_states.permute(0, 2, 1)
    hidden_states = self.avg_pooler(hidden_states)
    hidden_states = hidden_states.permute(0, 2, 1)

    hidden_states = self.layer_norm(hidden_states)
    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, encoder_states, all_attentions] if v is not None
        )
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=encoder_states,
        attentions=all_attentions,
    )


def encoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_head_mask: torch.Tensor,
    output_attentions: bool = False,
    cu_seq_lens: torch.Tensor = None,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states, attn_weights, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
        cu_seq_lens=cu_seq_lens,
    )
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.activation_dropout, training=self.training
    )
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )
    hidden_states = residual + hidden_states

    if hidden_states.dtype == torch.float16 and (
        torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
    ):
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.Tensor] = None,
):
    bsz = hidden_states.shape[0]
    query_states = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
    causal_mask = attention_mask

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    max_seqlen = (
        torch.diff(cu_seq_lens).max().item() if cu_seq_lens is not None else None
    )
    window_size = (-1, -1)

    # Align with flash attn forward in whisper
    # No causal mask, no softmax scale
    attn_output = flash_attn_varlen_func(
        q=query_states,
        k=key_states,
        v=value_states,
        cu_seqlens_q=cu_seq_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
        window_size=window_size,
        dropout_p=self.dropout if self.training else 0.0,
    )

    attn_output = attn_output.reshape(-1, self.embed_dim).contiguous()

    attn_output = self.out_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
