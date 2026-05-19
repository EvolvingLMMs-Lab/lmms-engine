from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ModelOutputWithPast as HFQwen3_5ModelOutputWithPast,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextModel,
    Qwen3_5VisionPatchEmbed,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeModel,
    Qwen3_5MoeTextModel,
)
from transformers.utils import is_flash_attn_2_available, logging

from ...parallel.sequence_parallel.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    repeat_kv,
    ulysses_pad,
    validate_ulysses_config,
)
from ..common_ops.rope import qwen3_vl_get_rope_index


@dataclass
class Qwen3_5ModelOutputWithPast(HFQwen3_5ModelOutputWithPast):
    """Extends the upstream output with packed-mode bookkeeping fields used by
    the patched LCE forward (``seq_lens`` and ``word_idx``)."""

    seq_lens: Optional[torch.IntTensor] = None
    word_idx: Optional[torch.IntTensor] = None


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

try:
    from fla.modules.conv.cp.ops import causal_conv1d_cp
    from fla.ops.cp.context import build_cp_context

    _HAS_FLA_CP = True
except ImportError:
    causal_conv1d_cp = None
    build_cp_context = None
    _HAS_FLA_CP = False


def patch_embed_forward(
    self: Qwen3_5VisionPatchEmbed,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Replacement for ``Qwen3_5VisionPatchEmbed.forward``.

    Mathematically equivalent to the upstream ``Conv3d`` (kernel == stride), but
    avoids cudnn occasionally falling back to a very slow Conv3d kernel on
    packed varlen ViT inputs. Done in fp32 for numerical stability, then cast
    back to the proj weight dtype.
    """
    target_dtype = self.proj.weight.dtype
    proj_weight = self.proj.weight
    proj_bias = self.proj.bias
    with torch.amp.autocast(device_type="cuda", enabled=False):
        hidden_states_fp32 = hidden_states.float()
        weight_fp32 = proj_weight.view(self.embed_dim, -1).float()
        bias_fp32 = proj_bias.float() if proj_bias is not None else None
        hidden_states = F.linear(hidden_states_fp32, weight_fp32, bias_fp32)
    return hidden_states.to(dtype=target_dtype)


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
    self: Union[Qwen3_5GatedDeltaNet, Qwen3_5MoeGatedDeltaNet],
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

    Sequence parallel (Ulysses SP world size > 1):

    * Linear attention can **not** use the Ulysses all-to-all trick (the
      recurrent state runs sequentially along the seq dim, so swapping
      heads <-> seq would collapse all tokens onto a single rank). Instead
      we use fla's Context Parallel: each rank keeps its local seq shard,
      builds an ``FLACPContext`` from the *global* ``cu_seq_lens`` so the
      kernel knows where the sample boundaries fall on every rank, and we
      manually do the causal-conv1d halo exchange (``conv_kernel_size - 1``
      tokens from the previous rank are prepended before the conv and
      stripped off afterwards).
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

    # ---- (optional) FLA context-parallel setup ----
    # Under SP, build an ``FLACPContext`` from the *global* ``cu_seq_lens``.
    # The context carries local cu_seqlens, sample-boundary-aware conv-halo
    # metadata, and the SP process group. Both ``causal_conv1d_cp`` and
    # ``chunk_gated_delta_rule`` consume the context directly.
    if get_ulysses_sequence_parallel_world_size() > 1:
        if not _HAS_FLA_CP:
            raise RuntimeError(
                "Sequence parallel for Qwen3.5 linear attention requires fla>=0.4 "
                "with `fla.ops.cp` / `fla.modules.conv.cp`. Please upgrade fla."
            )
        cp_ctx = build_cp_context(
            cu_seqlens=cu_seq_lens.to(torch.long),
            group=get_ulysses_sequence_parallel_group(),
            conv1d_kernel_size=self.conv_kernel_size,
        )
    else:
        cp_ctx = None

    mixed_qkv = self.in_proj_qkv(hidden_states)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # ---- causal conv1d ----
    # Under SP: use fla's CP-aware conv (it talks to the previous rank to set
    # up the conv1d ``initial_state`` and handles backward gradient sync).
    # Without SP: keep the fast Tri Dao ``causal_conv1d_fn`` (with ``seq_idx``
    # to respect packed sample boundaries).
    if cp_ctx is not None:
        # fla wants (B, T, D); Tri Dao wants (B, D, T). We never did the
        # transpose, so just pass mixed_qkv straight through.
        mixed_qkv = causal_conv1d_cp(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            cp_context=cp_ctx,
        )
    elif _HAS_CAUSAL_CONV1D:
        # Tri Dao kernel: needs channel-last (B, D, T)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        seq_idx = _seq_idx_from_cu_seqlens(cu_seq_lens, total_tokens=seq_len)
        mixed_qkv = causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)
    else:
        # Fallback: plain nn.Conv1d. Leaks up to (kernel-1) tokens across
        # sample boundaries. Accepted to avoid the causal_conv1d build dep.
        logger.warning_once(
            f"Packed linear attention without causal_conv1d_fn; up to "
            f"{self.conv_kernel_size - 1} tokens will leak across sample "
            f"boundaries in the input conv. Install causal_conv1d to avoid."
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)
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

    if cp_ctx is not None:
        # When `cp_context` is given, fla overrides `cu_seqlens` internally
        # with `cp_context.cu_seqlens` -- do NOT pass cu_seqlens again.
        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cp_context=cp_ctx,
        )
    else:
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


def text_model_forward(
    self: Union[Qwen3_5TextModel, Qwen3_5MoeTextModel],
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPastAndRmpad]:
    """Packed forward for ``Qwen3_5TextModel``.

    Caller is expected to have already done un-padding (and any SP slicing)
    upstream — this mirrors the Qwen3-VL design where the parent
    ``Qwen3_5Model.forward`` does the un-pad and feeds packed tensors here.
    ``position_ids`` is expected as ``(3, B, packed_seq)`` mrope.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        # inputs_embeds is packed 2D ``(total_tokens, hidden)`` here, so use
        # ``shape[0]`` (NOT ``shape[1]`` — that's hidden_size).
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[0],
            device=inputs_embeds.device,
        )

    # Qwen3.5 expects 4-component position_ids ``(text, t, h, w)``. Build the
    # 4-dim layout (mirrors upstream ``Qwen3_5TextModel.forward``):
    #   * None -> arange expanded to (4, B, S)
    #   * 2D   -> expand to (4, B, S)
    #   * 3D shape[0]==3 (true 3-axis mrope) -> prepend a text axis (cumulative
    #     ``arange`` along packed seq) and stack to (4, B, S)
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(4, 1, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
    elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
        text_axis = (
            torch.arange(position_ids.shape[-1], device=position_ids.device, dtype=position_ids.dtype)
            .view(1, 1, -1)
            .expand(1, position_ids.shape[1], -1)
        )
        position_ids = torch.cat([text_axis, position_ids], dim=0)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    hidden_states = inputs_embeds

    position_embeddings = self.rotary_emb(hidden_states, position_ids)

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
    self: Union[Qwen3_5Attention, Qwen3_5MoeAttention],
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
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Qwen3.5 uses gated attention: q_proj outputs query + gate (2x size).
    # The gate is per-token and stays seq-sharded -- it is *not* part of the
    # Ulysses all-to-all dance (which only swaps seq <-> head on Q/K/V).
    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
    value_states = self.v_proj(hidden_states).view(hidden_shape)
    cos, sin = position_embeddings

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        validate_ulysses_config(query_states.size(-2), ulysses_sp_size)
        # Repeat KV heads so they divide sp_size (flash-attn handles MQA/GQA
        # in the kernel; we just need n_heads % sp_size == 0 after this).
        repeats = max(ulysses_sp_size // key_states.size(-2), 1)
        if repeats > 1:
            key_states = key_states.repeat_interleave(repeats, dim=-2)
            value_states = value_states.repeat_interleave(repeats, dim=-2)

        # (seq/sp, n_head, head_dim) -> (seq, n_head/sp, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=0, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=0, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=0, head_dim=1)

        # `cu_seq_lens` here is the *global* (already-ulysses-padded) one,
        # so it lines up with the gathered seq. Nothing to fix.

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

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # (seq, n_head/sp, head_dim) -> (seq/sp, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    # Apply the gated attention mechanism (gate stays seq-sharded throughout).
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def model_forward(
    self: Union[Qwen3_5Model, Qwen3_5MoeModel],
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[tuple, Qwen3_5ModelOutputWithPast]:
    """Replacement for ``Qwen3_5Model.forward``.

    Differences from upstream:
      * Uses ``qwen3_vl_get_rope_index`` instead of
        ``self.compute_3d_position_ids(..., mm_token_type_ids=...)``, so we
        don't need the processor to emit ``mm_token_type_ids``.
      * Does the rmpad un-pad here and feeds packed tensors (with
        ``cu_seq_lens`` / ``indices``) to the patched
        ``Qwen3_5TextModel.forward``. Mirrors the Qwen3-VL ops layout.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # ---- un-pad input_ids / inputs_embeds ----
    if input_ids is not None:
        original_input_ids = input_ids
        input_ids, indices, cu_seq_lens, _ = _unpad_input(input_ids, attention_mask=attention_mask)
        batch_size, seq_length = original_input_ids.shape
    else:
        original_input_ids = None
        original_inputs_embeds = inputs_embeds
        inputs_embeds, indices, cu_seq_lens, _ = _unpad_input(inputs_embeds, attention_mask=attention_mask)
        batch_size, seq_length, _ = original_inputs_embeds.shape

    # ---- compute 3D position ids from padded layout, then gather to packed ----
    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        position_ids, rope_deltas = qwen3_vl_get_rope_index(
            self,
            original_input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask=attention_mask_tensor,
        )
        self.rope_deltas = rope_deltas

    # position_ids: (c, B, S) -> packed (c, 1, total_tokens)
    position_ids = (
        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
    )
    if get_ulysses_sequence_parallel_world_size() > 1:
        # Pad packed seq to a multiple of sp_size. Both Ulysses (full-attn)
        # and FLA CP (linear-attn) assume each rank holds the same number of
        # contiguous tokens, so we apply the same padding for both paths.
        input_ids, position_ids, pad_size = ulysses_pad(
            input_ids.unsqueeze(0),
            position_ids,
            sp_size=get_ulysses_sequence_parallel_world_size(),
        )
        input_ids = input_ids.squeeze(0)
        if pad_size > 0:
            # Mark the pad span as its own sample so linear-attn / causal-conv
            # don't leak the pad region back into real samples.
            cu_seq_lens = torch.cat([cu_seq_lens, cu_seq_lens.new_tensor([cu_seq_lens[-1].item() + pad_size])])

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # ---- visual feature injection (still on packed inputs_embeds) ----
    if pixel_values is not None:
        image_outputs: BaseModelOutputWithPooling = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeds = image_outputs.pooler_output
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_outputs: BaseModelOutputWithPooling = self.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True
        )
        video_embeds = video_outputs.pooler_output
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # `cu_seq_lens` / `indices` may already be in **kwargs from the collator
    # (we compute fresh ones from attention_mask); drop them to avoid duplicate kwargs.
    kwargs.pop("cu_seq_lens", None)
    kwargs.pop("indices", None)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        indices=indices,
        cu_seq_lens=cu_seq_lens,
        **kwargs,
    )

    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
        seq_lens=cu_seq_lens,
        word_idx=indices,
    )
