"""Patched forwards for transformers.models.qwen3_5_moe.

Attention-side forwards (`attn_forward`, `linear_attn_forward`) and the vision
patch-embed are reused as-is from the dense qwen3_5 ops — the gated-attention
and gated-delta-net layers are structurally identical between the dense and
MoE variants.

MoE-specific paths (kept local):
- `decoder_layer_forward` — handles the SparseMoeBlock tuple return shape and
  propagates router_logits when ``output_router_logits`` is requested.
- `text_model_forward` / `model_forward` — collect router_logits across layers
  and surface them on ``BaseModelOutputWithPastAndRmpad`` so ``lce_forward``
  can compute the load-balancing aux loss.
- `moe_sparse_layer_forward` — routed experts + shared_expert combine.
- `experts_forward` — stacked-parameter experts (gate_up_proj + down_proj).
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeModel,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextModel,
)
from transformers.utils import is_flash_attn_2_available

# ---- reused as-is from qwen3_5 (dense) ----
from lmms_engine.models.qwen3_5.qwen3_5_ops import attn_forward
from lmms_engine.models.qwen3_5.qwen3_5_ops import (  # noqa: F401
    linear_attn_forward as gated_delta_net_forward,
)
from lmms_engine.models.qwen3_5.qwen3_5_ops import patch_embed_forward

from ..common_ops.rope import qwen3_vl_get_rope_index
from ..sequence_packing_utils import BaseModelOutputWithPastAndRmpad, _unpad_input

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, rearrange


# ---------------------------------------------------------------------------
# decoder_layer_forward — same attention dispatch as qwen3_5, but MoE MLP
# returns (hidden, router_logits) tuple instead of a plain tensor.
# ---------------------------------------------------------------------------
def decoder_layer_forward(
    self: Qwen3_5MoeDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    output_router_logits: bool = False,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
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
            cache_position=cache_position,
            **kwargs,
        )
    else:
        raise ValueError(f"unknown layer_type={self.layer_type!r}")

    hidden_states = residual + hidden_states

    # MoE block — wraps add batch dim if rmpad flattened to 2D
    residual = hidden_states
    needs_squeeze = hidden_states.ndim == 2
    if needs_squeeze:
        hidden_states = hidden_states.unsqueeze(0)
    hidden_states = self.post_attention_layernorm(hidden_states)
    mlp_output = self.mlp(hidden_states)

    # Qwen3_5MoeSparseMoeBlock returns (Tensor, router_logits)
    router_logits = None
    if isinstance(mlp_output, tuple):
        hidden_states, router_logits = mlp_output
    else:
        hidden_states = mlp_output

    if needs_squeeze:
        hidden_states = hidden_states.squeeze(0)
    hidden_states = residual + hidden_states

    if output_router_logits and router_logits is not None:
        return hidden_states, router_logits
    return hidden_states


# ---------------------------------------------------------------------------
# text_model_forward — like qwen3_5 dense, but collects router_logits across
# layers when requested.
# ---------------------------------------------------------------------------
def text_model_forward(
    self: Qwen3_5MoeTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    output_router_logits: Optional[bool] = None,
    **kwargs,
) -> BaseModelOutputWithPastAndRmpad:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else getattr(self.config, "output_router_logits", False)
    )

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

    # Qwen3.5 expects 4-component position_ids ``(text, t, h, w)``.
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

    all_router_logits = () if output_router_logits else None

    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cu_seq_lens=cu_seq_lens,
            indices=indices,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        if isinstance(layer_outputs, tuple):
            hidden_states, router_logits = layer_outputs
            if output_router_logits and router_logits is not None:
                all_router_logits += (router_logits,)
        else:
            hidden_states = layer_outputs

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPastAndRmpad(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        seq_lens=cu_seq_lens,
        word_idx=indices,
        router_logits=all_router_logits if output_router_logits else None,
    )


# ---------------------------------------------------------------------------
# model_forward — outer multimodal wrapper. Mirrors qwen3_5 dense
# model_forward, but plumbs output_router_logits through and surfaces
# router_logits on the returned BaseModelOutputWithPastAndRmpad.
# ---------------------------------------------------------------------------
def model_forward(
    self: Qwen3_5MoeModel,
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
    output_router_logits: Optional[bool] = None,
    **kwargs,
) -> BaseModelOutputWithPastAndRmpad:
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
        output_router_logits=output_router_logits,
        **kwargs,
    )

    return BaseModelOutputWithPastAndRmpad(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        seq_lens=cu_seq_lens,
        word_idx=indices,
        router_logits=getattr(outputs, "router_logits", None),
    )


# ---------------------------------------------------------------------------
# moe_sparse_layer_forward — routed experts + shared_expert combine
# ---------------------------------------------------------------------------
def moe_sparse_layer_forward(
    self: Qwen3_5MoeSparseMoeBlock,
    hidden_states: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)

    # Shared expert path
    shared_out = self.shared_expert(hidden_states_flat)
    shared_out = torch.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_out

    # Router (returns logits, normalized weights, indices)
    router_logits, routing_weights, selected_experts = self.gate(hidden_states_flat)
    num_experts = self.gate.num_experts
    top_k = self.gate.top_k

    # Build per-expert routing tensors (same shape qwen3_moe uses so the EP
    # dispatch in Qwen3_5MoeParallelStyle._input_fn is identical)
    selected_experts = selected_experts.to(torch.float32)
    num_tokens_per_expert = torch.histc(selected_experts, bins=num_experts, min=0, max=num_experts)
    selected_experts = selected_experts.to(torch.int64)
    num_tokens_per_expert = num_tokens_per_expert.to(torch.int64)

    token_indices_experts_sorted = torch.argsort(selected_experts.view(-1), stable=True)
    top_scores_experts_sorted = routing_weights.view(-1)[token_indices_experts_sorted]
    token_indices_experts_sorted = token_indices_experts_sorted // top_k

    token_indices_experts_sorted = token_indices_experts_sorted.reshape(-1, 1).expand(-1, hidden_dim)
    routed_input = torch.gather(hidden_states_flat, dim=0, index=token_indices_experts_sorted)

    out_experts_split = self.experts(routed_input, num_tokens_per_expert)

    routed_output = out_experts_split * top_scores_experts_sorted.reshape(-1, 1)
    final_hidden_states = torch.zeros_like(hidden_states_flat)
    final_hidden_states = final_hidden_states.scatter_add(dim=0, index=token_indices_experts_sorted, src=routed_output)

    # Combine routed + shared
    final_hidden_states = final_hidden_states + shared_out
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


# ---------------------------------------------------------------------------
# experts_forward — stacked-parameter experts (same shape as qwen3_moe T>=5)
# ---------------------------------------------------------------------------
def experts_forward(self, *routed_input):
    if len(routed_input) == 2 and routed_input[1].ndim == 1:
        routed_input = torch.split(
            routed_input[0],
            split_size_or_sections=routed_input[1].tolist(),
            dim=0,
        )

    if isinstance(self.down_proj, DTensor):
        down_proj = self.down_proj.to_local()
        gate_up_proj = self.gate_up_proj.to_local()
    else:
        down_proj = self.down_proj
        gate_up_proj = self.gate_up_proj

    out_experts_split = []
    for idx, x in enumerate(routed_input):
        gate_up = F.linear(x, gate_up_proj[idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = self.act_fn(gate) * up
        hidden = F.linear(hidden, down_proj[idx])
        out_experts_split.append(hidden)

    return torch.cat(out_experts_split, dim=0)
