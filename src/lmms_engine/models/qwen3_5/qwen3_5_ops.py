from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5DynamicCache,
    Qwen3_5TextModel,
    apply_rotary_pos_emb,
)
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

from ..sequence_packing_utils import BaseModelOutputWithPastAndRmpad, _unpad_input

logger = logging.get_logger(__name__)


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, rearrange


def model_forward(
    self: Qwen3_5TextModel,
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
            input_ids_rmpad = torch.zeros(1, inputs_embeds.shape[0], dtype=torch.long, device=inputs_embeds.device)
            inputs_embeds = slice_input_tensor(inputs_embeds, dim=0, padding=True)
    bs, seqlen = original_inputs.shape[:2]

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = Qwen3_5DynamicCache(config=self.config)

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

    if get_ulysses_sequence_parallel_world_size() > 1:
        _, position_ids, pad_size = ulysses_pad(
            input_ids_rmpad,
            original_position_ids,
            sp_size=get_ulysses_sequence_parallel_world_size(),
        )

    # Qwen3.5 uses 4-component position IDs: text + temporal + height + width
    # For text-only: expand 1D position_ids to 4 components
    if position_ids.ndim == 2 and position_ids.shape[0] == 1:
        position_ids = position_ids.expand(4, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        mrope_position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
        mrope_position_ids = position_ids

    hidden_states = inputs_embeds

    position_embeddings = self.rotary_emb(hidden_states, mrope_position_ids)

    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
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


def decoder_layer_forward(
    self: Qwen3_5DecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        # GatedDeltaNet expects 3D (batch, seq_len, hidden) but rmpad
        # flattens to 2D (total_tokens, hidden). Add a batch dim of 1.
        needs_squeeze = hidden_states.ndim == 2
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=None,
        )
        if needs_squeeze:
            hidden_states = hidden_states.squeeze(0)
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cu_seq_lens=cu_seq_lens,
            indices=indices,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def attn_forward(
    self: Qwen3_5Attention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: bool = False,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Qwen3.5 uses gated attention: q_proj outputs query + gate (2x size)
    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
    value_states = self.v_proj(hidden_states).view(hidden_shape)
    cos, sin = position_embeddings

    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    if ulysses_sp_size > 1:
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"

        repeats = max(ulysses_sp_size // key_states.size(1), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        query_states = gather_seq_scatter_heads(query_states, seq_dim=0, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=0, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=0, head_dim=1)

        if cu_seq_lens is not None:
            seq_len_tensor = torch.tensor(
                query_states.shape[0],
                device=cu_seq_lens.device,
                dtype=cu_seq_lens.dtype,
            )
            needs_append = (cu_seq_lens.max() < seq_len_tensor).item()
            if needs_append:
                cu_seq_lens = torch.cat([cu_seq_lens, seq_len_tensor.unsqueeze(0)])

    query_states = query_states.unsqueeze(0).transpose(1, 2)
    key_states = key_states.unsqueeze(0).transpose(1, 2)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None and hasattr(past_key_values, "update"):
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": None}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    query_states = query_states.transpose(1, 2).squeeze(0)
    key_states = key_states.transpose(1, 2).squeeze(0)

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
        dropout_p=0.0,
    )

    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    # Apply the gated attention mechanism
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)

    return attn_output, None
