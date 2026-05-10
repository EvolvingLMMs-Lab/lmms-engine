from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextModel,
    apply_mask_to_padding_states,
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

try:
    from causal_conv1d import causal_conv1d_fn

    _HAS_CAUSAL_CONV1D = True
except ImportError:
    causal_conv1d_fn = None
    _HAS_CAUSAL_CONV1D = False

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    _HAS_FLA = True
except ImportError:
    chunk_gated_delta_rule = None
    _HAS_FLA = False


def _seq_idx_from_cu_seqlens(cu_seqlens: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """Build per-token sample index (int32) from cumulative seqlens.

    cu_seqlens shape ``[N+1]``; returns shape ``[1, total_tokens]`` int32 with
    each token labelled by its sample id, as required by ``causal_conv1d_fn``.
    """
    lens = torch.diff(cu_seqlens.to(torch.long))
    seq_idx = torch.repeat_interleave(
        torch.arange(lens.numel(), device=cu_seqlens.device, dtype=torch.int32),
        lens,
    )
    # Pad / truncate defensively in case cu_seqlens didn't end exactly at total_tokens
    if seq_idx.numel() != total_tokens:
        if seq_idx.numel() < total_tokens:
            pad = total_tokens - seq_idx.numel()
            seq_idx = torch.cat(
                [seq_idx, torch.full((pad,), seq_idx[-1].item() + 1, device=seq_idx.device, dtype=torch.int32)]
            )
        else:
            seq_idx = seq_idx[:total_tokens]
    return seq_idx.unsqueeze(0).contiguous()


def linear_attn_forward(
    self: Qwen3_5GatedDeltaNet,
    hidden_states: torch.Tensor,
    cache_params: Optional[Cache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cu_seq_lens: Optional[torch.Tensor] = None,
    **kwargs,  # absorb e.g. cache_position passed by upstream decoder layer
) -> torch.Tensor:
    """Packed/varlen ``Qwen3_5GatedDeltaNet.forward``.

    This patch is only installed when ``use_rmpad=True``, so we always go
    through the packed path. ``cu_seq_lens`` must be provided.

    * ``causal_conv1d_fn(..., seq_idx=...)`` if causal_conv1d is installed,
      otherwise ``nn.Conv1d`` (leaks up to ``conv_kernel_size - 1`` tokens
      across sample boundaries — accepted as a soft compromise).
    * ``fla.ops.gated_delta_rule.chunk_gated_delta_rule(..., cu_seqlens=...)``
      so the recurrent state resets per sample. fla is required.
    """
    assert cu_seq_lens is not None, "linear_attn_forward requires cu_seq_lens (rmpad must be on)"
    if not _HAS_FLA:
        raise RuntimeError(
            "Packed linear attention requires `fla` (flash-linear-attention). "
            "Install it via `pip install flash-linear-attention`."
        )

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    assert batch_size == 1, (
        f"packed linear_attn_forward expects batch_size=1, got {batch_size}. "
        "Caller must squeeze rmpad inputs to (1, total_tokens, hidden)."
    )

    seq_idx = _seq_idx_from_cu_seqlens(cu_seq_lens, total_tokens=seq_len)

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, conv_dim, T)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if _HAS_CAUSAL_CONV1D:
        mixed_qkv = causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
    else:
        # Fallback: plain nn.Conv1d. Leaks up to (kernel-1) tokens across
        # sample boundaries. Accepted to avoid the causal_conv1d build dep.
        logger.warning_once(
            f"Packed linear attention without causal_conv1d_fn; up to "
            f"{self.conv_kernel_size - 1} tokens will leak across sample "
            f"boundaries in the input conv. Install causal_conv1d to avoid."
        )
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    core_attn_out, _ = chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seq_lens.to(torch.long),
    )

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    return self.out_proj(core_attn_out)


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
            cu_seq_lens=cu_seq_lens,
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
