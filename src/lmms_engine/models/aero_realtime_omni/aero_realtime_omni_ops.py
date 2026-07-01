"""Talker rmpad + Ulysses ops for aero_realtime_omni.

Derived from ``qwen3_vl_ops`` (text_model_forward / decoder_layer_forward /
attn_forward). The talker is a Qwen-style causal LM with 3-axis mrope; here we:

1. Feed the trunk a **packed** sequence (``inputs_embeds [N, D]``,
   ``cu_seq_lens`` marking sample boundaries) instead of the current
   ``[B, max_len, D]`` padded batch.
2. Under Ulysses (``sp_size > 1``) all-to-all Q/K/V along heads so each rank
   sees full-length Q/K/V of a head slice.
3. Use ``varlen_attn`` (flash / sdpa) with ``cu_seqlens_q=cu_seqlens_k=cu_seq_lens``.

The mrope apply lives in ``AeroRealtimeTalkerRotaryEmbedding.forward`` (see
``modeling_aero_realtime_talker.py``), so the cos/sin returned here are the
already-mroped tensors, matching ``apply_rotary_pos_emb``'s expectation.
"""

from typing import Optional, Union

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_flash_attn_2_available, logging

from lmms_engine.parallel.sequence_parallel.ulysses import (
    gather_heads_scatter_seq,
    gather_outputs_and_unpad,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    repeat_kv,
    slice_input_tensor,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)

from ..sequence_packing_utils import _unpad_input

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, rearrange

from lmms_engine.kernels.attention import varlen_attn

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )

    _HAS_LIGER = True
except Exception:
    _HAS_LIGER = False

logger = logging.get_logger(__name__)


def talker_model_forward(
    self,  # AeroRealtimeTalkerModel
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[tuple, BaseModelOutputWithPast]:
    """RMPad + Ulysses forward for ``AeroRealtimeTalkerModel``.

    Expects the caller (``compute_talker_loss``) to have already packed the
    per-sample sequences into ``inputs_embeds [N, D]`` and to pass the
    matching ``cu_seq_lens`` and ``position_ids`` (3-axis mrope). No padded
    ``attention_mask`` is required.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[0],
            device=inputs_embeds.device,
        )

    # Talker uses 3-axis mrope. If position_ids weren't provided, replicate
    # cache_position along the 3 axes. Callers that pack samples should pass
    # position_ids explicitly (shape ``(3, N)`` or ``(3, 1, N)``).
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    text_position_ids = position_ids[0]

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            cu_seq_lens=cu_seq_lens,
            indices=indices,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def decoder_layer_forward(
    self,  # AeroRealtimeTalkerDecoderLayer
    hidden_states: torch.Tensor,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        indices=indices,
        cu_seq_lens=cu_seq_lens,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


def attn_forward(
    self,  # AeroRealtimeTalkerAttention
    hidden_states: torch.Tensor,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    from .modeling_aero_realtime_talker import apply_multimodal_rotary_pos_emb

    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    cos, sin = position_embeddings
    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        assert position_ids is not None, (
            f"position_ids is required for Ulysses sequence parallelism " f"(sp_size={ulysses_sp_size}). Got None."
        )

        repeats = max(ulysses_sp_size // key_states.size(1), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (seq_len/n, n_head, head_dim) -> (seq_len, n_head/n, head_dim)
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
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
        self.rope_scaling["interleaved"],
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if cu_seq_lens is not None:
        max_seqlen = torch.diff(cu_seq_lens).max().item()
    else:
        max_seqlen = None

    query_states = query_states.transpose(1, 2).squeeze(0)
    key_states = key_states.transpose(1, 2).squeeze(0)

    attn_output = varlen_attn(
        q=query_states,
        k=key_states,
        v=value_states,
        cu_seqlens_q=cu_seq_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
        window_size=(-1, -1),
        softmax_scale=self.head_dim**-0.5,
        dropout_p=0.0,
        backend=self.config._attn_implementation,
    )

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def compute_talker_loss(
    self,  # AeroRealtimeOmniForConditionalGeneration
    packed_last_hidden_state: torch.Tensor,
    packed_input_ids: torch.Tensor,
    codec_labels_flat: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Packed teacher-forced talker loss (group-0 + residual).

    Replaces the padded-batch method on ``AeroRealtimeOmniForConditionalGeneration``.
    Cats every sample's ``[cond(3) + body(n_i)]`` into a single packed sequence
    and feeds it to the patched ``talker.model`` with matching ``cu_seq_lens``
    and 3-axis ``position_ids``.
    """
    talker = self.talker
    talker_cfg = self.config.talker_config
    audio_token_id = self.config.thinker_config.audio_token_index
    speaker_id = self._default_speaker_id
    codec_bos_id = talker_cfg.codec_bos_id
    codec_nothink_id = talker_cfg.codec_nothink_id

    device = packed_last_hidden_state.device
    codec_emb = talker.get_input_embeddings()

    audio_mask = packed_input_ids == audio_token_id  # [T]
    S = cu_seq_lens.numel() - 1

    body_hidden_thinker = packed_last_hidden_state[audio_mask]  # [N_body, D_thinker]
    body_codes = codec_labels_flat[audio_mask]  # [N_body, G]  (keeps -100)
    N_body = body_hidden_thinker.shape[0]

    if N_body == 0:
        return packed_last_hidden_state.sum() * 0.0

    thinker_seg = torch.repeat_interleave(
        torch.arange(S, device=device, dtype=cu_seq_lens.dtype),
        torch.diff(cu_seq_lens),
    )  # [T]
    body_lens = torch.zeros(S, dtype=torch.long, device=device).scatter_add_(
        0,
        thinker_seg[audio_mask].long(),
        torch.ones(N_body, dtype=torch.long, device=device),
    )  # [S]

    text_h = talker.text_projection(body_hidden_thinker)  # [N_body, D_talker]
    sample_body_seg = torch.repeat_interleave(torch.arange(S, device=device, dtype=torch.long), body_lens)  # [N_body]
    is_first_in_sample = torch.zeros_like(sample_body_seg, dtype=torch.bool)
    is_first_in_sample[0] = True
    is_first_in_sample[1:] = sample_body_seg[1:] != sample_body_seg[:-1]
    prev_group0 = torch.roll(body_codes[:, 0].clamp(min=0), 1)  # [N_body]
    prev_group0[is_first_in_sample] = codec_bos_id
    body_emb = text_h + codec_emb(prev_group0)  # [N_body, D_talker]

    cond_ids = torch.tensor(
        [codec_bos_id, codec_nothink_id, speaker_id],
        device=device,
        dtype=torch.long,
    )
    cond_ids_all = cond_ids.unsqueeze(0).expand(S, -1).reshape(-1)  # [3S]
    cond_emb_all = codec_emb(cond_ids_all)  # [3S, D_talker]

    seg_lens = body_lens + 3  # [S]
    N = int(seg_lens.sum().item())
    seg_starts = torch.cat([seg_lens.new_zeros(1), torch.cumsum(seg_lens[:-1], 0)])
    talker_seg = torch.repeat_interleave(torch.arange(S, device=device, dtype=torch.long), seg_lens)  # [N]
    local_pos = torch.arange(N, device=device, dtype=torch.long) - seg_starts[talker_seg]
    is_cond_token = local_pos < 3  # [N]
    is_body_token = ~is_cond_token  # [N]

    d_talker = talker.config.hidden_size
    packed = body_emb.new_empty(N, d_talker)
    packed[is_cond_token] = cond_emb_all.to(packed.dtype)
    packed[is_body_token] = body_emb

    position_ids = local_pos.view(1, 1, -1).expand(3, 1, -1).contiguous()  # [3, 1, N]
    cu_seq_lens_talker = torch.cat([seg_lens.new_zeros(1), torch.cumsum(seg_lens, 0)]).to(torch.int32)  # [S+1]

    # Pad the packed sequence to be divisible by sp_size (mirrors the thinker
    # rmpad path in ``aero_realtime_liger.py``). The wrapping
    # ``patch_vlm_for_ulysses_input_slicing`` will slice ``inputs_embeds`` per
    # rank, and ``gather_seq_scatter_heads`` inside ``attn_forward`` will
    # regather to the full padded length.
    sp_size = get_ulysses_sequence_parallel_world_size()
    pad_size = 0
    if sp_size > 1:
        pad_size = (sp_size - N % sp_size) % sp_size
        if pad_size > 0:
            packed = torch.nn.functional.pad(packed, (0, 0, 0, pad_size), value=0.0)
            pad_pos = torch.arange(pad_size, device=device, dtype=position_ids.dtype)
            pad_pos = pad_pos.view(1, 1, -1).expand(3, 1, -1)
            position_ids = torch.cat([position_ids, pad_pos], dim=-1)
            # Mark pad span as its own "sample" so varlen attention masks it out
            # from the last real sample.
            cu_seq_lens_talker = torch.cat([cu_seq_lens_talker, cu_seq_lens_talker.new_tensor([N + pad_size])])

    trunk_out = talker.model(
        inputs_embeds=packed,
        cu_seq_lens=cu_seq_lens_talker,
        position_ids=position_ids,
    )
    trunk_hidden_all = trunk_out.last_hidden_state  # rank-shard, [N_padded/sp, D]

    # Gather sp-sharded hidden back to full N (padded). The wrapping
    # `patch_vlm_for_ulysses_input_slicing` sliced inputs_embeds on the way in;
    # here we re-gather+unpad to recover the un-padded N.
    if sp_size > 1:
        trunk_hidden_all = gather_outputs_and_unpad(trunk_hidden_all, gather_dim=0, unpad_dim=0, padding_size=pad_size)
    trunk_hidden = trunk_hidden_all[is_body_token]  # [N_body, D_talker]

    if _HAS_LIGER:
        lce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        group0_loss = lce(talker.codec_head.weight, trunk_hidden, body_codes[:, 0])
    else:
        group0_logits = talker.codec_head(trunk_hidden)
        group0_loss = F.cross_entropy(group0_logits, body_codes[:, 0], ignore_index=-100)

    _, residual_loss = talker.forward_sub_talker_finetune(body_codes, trunk_hidden)

    return group0_loss + residual_loss
