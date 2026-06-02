from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.voxtral_realtime.modeling_voxtral_realtime import (
    apply_rotary_pos_emb,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from lmms_engine.kernels.attention import varlen_attn
from lmms_engine.parallel.sequence_parallel.ulysses import (
    gather_heads_scatter_seq,
    gather_outputs_and_unpad,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
    validate_ulysses_config,
)

from .modeling_aero_realtime import (
    AeroRealtimeAudioAttention,
    AeroRealtimeAudioConv1dPaddingCache,
    AeroRealtimeAudioEncoder,
    AeroRealtimeAudioEncoderOutput,
)


def _get_unpad_data(attention_mask: torch.Tensor):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen


def _pad_hidden_states(hidden_states: torch.Tensor, position_ids: torch.Tensor, sp_size: int):
    pad_size = (sp_size - hidden_states.shape[0] % sp_size) % sp_size
    if pad_size == 0:
        return hidden_states, position_ids, pad_size

    hidden_pad = hidden_states.new_zeros((pad_size, hidden_states.shape[-1]))
    pos_pad = torch.arange(pad_size, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(0)
    return torch.cat([hidden_states, hidden_pad], dim=0), torch.cat([position_ids, pos_pad], dim=-1), pad_size


def attention_forward(
    self: AeroRealtimeAudioAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert isinstance(attention_mask, dict), "AeroRealtime rmpad attention requires packed attention metadata"
    return _varlen_attention_forward(
        self,
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        cu_seq_lens=attention_mask["cu_seq_lens"],
        max_seqlen=attention_mask["max_seqlen"],
    )


def _varlen_attention_forward(
    self: AeroRealtimeAudioAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cu_seq_lens: torch.Tensor,
    max_seqlen: int,
) -> tuple[torch.Tensor, None]:
    sp_size = get_ulysses_sequence_parallel_world_size()
    total = hidden_states.shape[0]
    query_states = self.q_proj(hidden_states).view(total, self.config.num_attention_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(total, self.config.num_key_value_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(total, self.config.num_key_value_heads, self.head_dim)

    cos, sin = position_embeddings
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = query_states.squeeze(0).transpose(0, 1).contiguous()
    key_states = key_states.squeeze(0).transpose(0, 1).contiguous()

    if sp_size > 1:
        validate_ulysses_config(query_states.size(1), sp_size)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=0, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=0, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=0, head_dim=1)

    window_left = int(getattr(self.config, "attention_window_left", -1))
    window_right = int(getattr(self.config, "attention_window_right", -1))
    window_size = (window_left, window_right)
    attn_output = varlen_attn(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seq_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=bool(getattr(self.config, "is_causal", False)),
        softmax_scale=self.scaling,
        window_size=window_size,
        dropout_p=0.0 if not self.training else self.attention_dropout,
        backend=self.config._attn_implementation,
    )
    if sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1)

    attn_output = self.o_proj(attn_output.reshape(total, -1))
    return attn_output, None


def encoder_forward(
    self: AeroRealtimeAudioEncoder,
    input_features: torch.FloatTensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    padding_cache: AeroRealtimeAudioConv1dPaddingCache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    use_padding_cache: bool | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | BaseModelOutputWithPooling:
    if (input_features is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_features or inputs_embeds")

    if use_padding_cache and padding_cache is None:
        padding_cache = AeroRealtimeAudioConv1dPaddingCache()

    if inputs_embeds is None:
        inputs_embeds = self.embedder(input_features, padding_cache)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    hidden_states = inputs_embeds
    if attention_mask is None:
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    indices, cu_seq_lens, max_seqlen = _get_unpad_data(attention_mask)
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])[indices]
    position_ids = position_ids.expand(inputs_embeds.shape[0], -1).reshape(-1)[indices].unsqueeze(0)
    sp_size = get_ulysses_sequence_parallel_world_size()
    pad_size = 0
    if sp_size > 1:
        hidden_states, position_ids, pad_size = _pad_hidden_states(hidden_states, position_ids, sp_size)
        hidden_states = slice_input_tensor(hidden_states, dim=0, padding=False)
        position_ids = slice_input_tensor(position_ids, dim=-1, padding=False)
        if pad_size > 0:
            cu_seq_lens = torch.cat([cu_seq_lens, cu_seq_lens.new_tensor([cu_seq_lens[-1].item() + pad_size])])
    causal_mask = {
        "cu_seq_lens": cu_seq_lens,
        "max_seqlen": max_seqlen,
    }
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    for encoder_layer in self.layers:
        hidden_states = encoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    if sp_size > 1:
        hidden_states = gather_outputs_and_unpad(hidden_states, gather_dim=0, unpad_dim=0, padding_size=pad_size)

    return AeroRealtimeAudioEncoderOutput(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        padding_cache=padding_cache,
    )


aero_realtime_attention_forward = attention_forward
aero_realtime_encoder_forward = encoder_forward

__all__ = [
    "attention_forward",
    "encoder_forward",
    "aero_realtime_attention_forward",
    "aero_realtime_encoder_forward",
]
