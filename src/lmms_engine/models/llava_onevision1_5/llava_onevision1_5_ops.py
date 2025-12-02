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

    # LLaVA OneVision uses 3D position_ids for temporal, height, width
    if position_ids is None:
        # the hard coded `3` is for temporal, height and width.
        position_ids = cache_position.view(1, 1, -1).expand(3, bs, -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, -1, -1)

    position_ids = position_ids.permute(1, 2, 0)

    position_ids = rearrange(position_ids, "b s d -> (b s) d")

    position_ids = index_first_axis(position_ids, indices)  # -> (packed_len, 3)

    position_ids = position_ids.transpose(0, 1)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=None,  # Not used in flash attention varlen
            position_ids=position_ids,
            past_key_value=past_key_values,
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
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.FloatTensor:
    """
    Simplified decoder layer forward for sequence packing.
    Returns only hidden_states (not a tuple) to match the usage in model_forward.
    """
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
        cache_position=cache_position,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


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
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    Flash Attention forward for LLaVA OneVision 1.5 with sequence packing support.
    This version uses flash_attn_varlen_func for variable-length sequences.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Project and normalize Q/K
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    cos, sin = position_embeddings

    ########## AlltoAll for Ulysses ##########
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    if ulysses_sp_size > 1:
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"

        # NOTE: repeat kv heads to be divided by sequence parallel
        repeats = max(ulysses_sp_size // key_states.size(1), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (seq_len/n, n_head, head_dim) -> (seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=0, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=0, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=0, head_dim=1)

        # Update cu_seq_lens if padding is used
        if cu_seq_lens.max().item() < query_states.shape[0]:
            cu_seq_lens = torch.cat(
                [
                    cu_seq_lens,
                    torch.tensor(
                        [query_states.shape[0]],
                        device=cu_seq_lens.device,
                        dtype=cu_seq_lens.dtype,
                    ),
                ]
            )

    # Add batch dimension and transpose for RoPE
    query_states = query_states.unsqueeze(0).transpose(1, 2)
    key_states = key_states.unsqueeze(0).transpose(1, 2)

    # Apply rotary position embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Remove batch dimension and transpose back
    query_states = query_states.transpose(1, 2).squeeze(0)
    key_states = key_states.transpose(1, 2).squeeze(0)

    # Handle past_key_value cache (for inference)
    if past_key_value is not None:
        # For training with sequence packing, we don't use cache
        # This is mainly for inference/generation
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states.unsqueeze(0).transpose(1, 2),
            value_states.unsqueeze(0).transpose(1, 2),
            self.layer_idx,
            cache_kwargs,
        )
        key_states = key_states.transpose(1, 2).squeeze(0)
        value_states = value_states.transpose(1, 2).squeeze(0)

    # Determine sliding window size
    window_size = (-1, -1)
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        window_size = (self.config.sliding_window, self.config.sliding_window)

    # Calculate max sequence length
    max_seqlen = torch.diff(cu_seq_lens).max().item() if cu_seq_lens is not None else None

    # Flash Attention with variable-length sequences
    dropout_rate = 0.0 if not self.training else self.attention_dropout
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

    # Reshape and project output
    attn_output = attn_output.reshape(-1, self.config.num_attention_heads * self.head_dim).contiguous()
    attn_output = self.o_proj(attn_output)

    attn_weights = None if not output_attentions else "NotImplemented"

    return attn_output, attn_weights, past_key_value
