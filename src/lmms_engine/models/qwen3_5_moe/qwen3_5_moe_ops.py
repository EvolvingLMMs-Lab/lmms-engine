"""Patched forwards for transformers.models.qwen3_5_moe.

Most forwards reuse the qwen3_5 (dense) implementations directly via import —
the attention path (`attn_forward`, `linear_attn_forward`) and the outer model
wrappers (`text_model_forward`, `model_forward`) are structurally identical
between the dense and MoE variants.

Only the MLP-side logic is MoE-specific:
- `decoder_layer_forward` — handles the SparseMoeBlock tuple return shape.
- `moe_sparse_layer_forward` — routed experts + shared_expert combine.
- `experts_forward` — stacked-parameter experts (gate_up_proj + down_proj).
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeSparseMoeBlock,
)

# ---- reused as-is from qwen3_5 (dense) ----
from lmms_engine.models.qwen3_5.qwen3_5_ops import attn_forward
from lmms_engine.models.qwen3_5.qwen3_5_ops import (  # noqa: F401
    linear_attn_forward as gated_delta_net_forward,
)
from lmms_engine.models.qwen3_5.qwen3_5_ops import (
    model_forward,
    patch_embed_forward,
    text_model_forward,
)


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
    output_router_logits: bool = True,
    **kwargs,
) -> torch.Tensor:
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
