import inspect
import warnings
from typing import List, Optional, Tuple, Union

import torch
from loguru import logger
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_flash_attn_2_available, logging

from lmms_engine.parallel.sequence_parallel.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    repeat_kv,
    slice_input_tensor,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)

from ..sequence_packing_utils import (
    BaseModelOutputWithPastAndRmpad,
    _get_unpad_data,
    _unpad_input,
)

logger = logging.get_logger(__name__)


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import (
        index_first_axis,
        pad_input,
        rearrange,
        unpad_input,
    )

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
except:
    apply_rotary_emb_func = None
    logger.warning_once("fail to load faster rotary ops, use PyTorch version by default. Please check image version")


# Import apply_rotary_pos_emb from the modeling file
from .modeling_llavaonevision1_5 import apply_rotary_pos_emb


# The forward func for the base model of LLaVAOneVision1_5_TextModel
def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPastAndRmpad]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if cu_seq_lens is None and input_ids is not None:
        original_inputs = input_ids
        input_ids, indices, cu_seq_lens, max_seqlen_in_batch = _unpad_input(input_ids, attention_mask)
        if get_ulysses_sequence_parallel_world_size() > 1:
            input_ids_rmpad = input_ids.unsqueeze(0)
            input_ids, _, pad_size = ulysses_pad_and_slice_inputs(
                input_ids.unsqueeze(0),
                sp_size=get_ulysses_sequence_parallel_world_size(),
            )
            input_ids = input_ids.squeeze(0)
    elif cu_seq_lens is None and inputs_embeds is not None:
        original_inputs = inputs_embeds
        inputs_embeds, indices, cu_seq_lens, max_seqlen_in_batch = _unpad_input(inputs_embeds, attention_mask)
        if get_ulysses_sequence_parallel_world_size() > 1:
            inputs_embeds = slice_input_tensor(inputs_embeds, dim=0, padding=True)
    bs, seqlen = original_inputs.shape[:2]

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seqlen,
            device=inputs_embeds.device,
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    position_ids = position_ids.repeat_interleave(bs, dim=0)

    position_ids = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(
        0, 1
    )
    original_position_ids = position_ids

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": original_inputs,  # Use original input ids to prepare mask
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": original_position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=None,  # Not used in flash attention varlen
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cu_seq_lens=cu_seq_lens,
            indices=indices,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPastAndRmpad(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        seq_lens=cu_seq_lens,
        word_idx=indices,
    )


# The decoder forward func for LLaVAOneVision1_5
def decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cu_seq_lens=cu_seq_lens,
        indices=indices,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


# The attn forward func for LLaVAOneVision1_5
def attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)



    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

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

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    max_seqlen = torch.diff(cu_seq_lens).max().item() if cu_seq_lens is not None else None
    window_size = (-1, -1)
    attn_output = flash_attn_varlen_func(
        q=query_states,
        k=key_states,
        v=value_states,
        cu_seqlens_q=cu_seq_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
        window_size=window_size,
        softmax_scale=self.head_dim**-0.5,
        dropout_p=dropout_rate,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
