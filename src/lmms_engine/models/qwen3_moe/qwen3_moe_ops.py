from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    MoeModelOutputWithPast,
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
    apply_rotary_pos_emb,
)

from lmms_engine.utils.import_utils import is_transformers_version_greater_or_equal_to

_IS_TRANSFORMERS_5 = is_transformers_version_greater_or_equal_to("5.0")
if _IS_TRANSFORMERS_5:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
from transformers.utils import is_flash_attn_2_available

from lmms_engine.models.sequence_packing_utils import (
    BaseModelOutputWithPastAndRmpad,
    _unpad_input,
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import (
        index_first_axis,
        pad_input,
        rearrange,
        unpad_input,
    )


def model_forward(
    self: Qwen3MoeModel,
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
) -> MoeModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if cu_seq_lens is None and input_ids is not None:
        original_inputs = input_ids
        input_ids, indices, cu_seq_lens, max_seqlen_in_batch = _unpad_input(input_ids, attention_mask)
    elif cu_seq_lens is None and inputs_embeds is not None:
        original_inputs = inputs_embeds
        inputs_embeds, indices, cu_seq_lens, max_seqlen_in_batch = _unpad_input(inputs_embeds, attention_mask)
    bs, seqlen = original_inputs.shape[:2]

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

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

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    output_router_logits = (
        output_router_logits if output_router_logits is not None else getattr(self.config, "output_router_logits", True)
    )
    all_router_logits = () if output_router_logits else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        layer_outputs = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            cu_seq_lens=cu_seq_lens,
            indices=indices,
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


def decoder_layer_forward(
    self: Qwen3MoeDecoderLayer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    output_router_logits: bool = True,
    **kwargs,
) -> torch.FloatTensor:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        output_router_logits (`bool`, *optional*):
            Whether or not to return the logits of all the routers. They are useful for computing the router loss,
            and should not be returned during inference.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_values (`Cache`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence.
        position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
        cu_seq_lens=cu_seq_lens,
        indices=indices,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    # Unsqueeze to unpack shape for the MoE sparse layer
    hidden_states = hidden_states.unsqueeze(0)
    hidden_states = self.post_attention_layernorm(hidden_states)
    mlp_output = self.mlp(hidden_states)

    router_logits = None
    if isinstance(mlp_output, tuple):
        hidden_states, router_logits = mlp_output
    else:
        hidden_states = mlp_output

    # Squeeze to pack shape for later
    hidden_states = hidden_states.squeeze(0)
    hidden_states = residual + hidden_states

    if output_router_logits and router_logits is not None:
        return hidden_states, router_logits
    return hidden_states


def attn_forward(
    self: Qwen3MoeAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    cos, sin = position_embeddings
    query_states = query_states.unsqueeze(0)
    key_states = key_states.unsqueeze(0)
    query_states = query_states.permute(0, 3, 1, 2)
    key_states = key_states.permute(0, 3, 1, 2)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def moe_sparse_layer_forward(self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    if hasattr(self.gate, "num_experts"):
        # transformers >= 5.0: TopKRouter
        num_experts = self.gate.num_experts
        top_k = self.gate.top_k
        router_logits, routing_weights, selected_experts = self.gate(hidden_states)
    else:
        # transformers < 5.0: nn.Linear gate
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        num_experts = self.num_experts
        top_k = self.top_k

    selected_experts = selected_experts.to(torch.float32)
    num_tokens_per_expert = torch.histc(selected_experts, bins=num_experts, min=0, max=num_experts)
    selected_experts = selected_experts.to(torch.int64)
    num_tokens_per_expert = num_tokens_per_expert.to(torch.int64)

    token_indices_experts_sorted = torch.argsort(selected_experts.view(-1), stable=True)
    top_scores_experts_sorted = routing_weights.view(-1)[token_indices_experts_sorted]
    token_indices_experts_sorted = token_indices_experts_sorted // top_k

    token_indices_experts_sorted = token_indices_experts_sorted.reshape(-1, 1).expand(-1, hidden_dim)
    routed_input = torch.gather(hidden_states, dim=0, index=token_indices_experts_sorted)

    out_experts_split = self.experts(routed_input, num_tokens_per_expert)

    routed_output = out_experts_split * top_scores_experts_sorted.reshape(-1, 1)
    final_hidden_states = torch.zeros_like(hidden_states)
    final_hidden_states = final_hidden_states.scatter_add(dim=0, index=token_indices_experts_sorted, src=routed_output)

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def experts_forward(self, *routed_input):
    if len(routed_input) == 2 and routed_input[1].ndim == 1:
        routed_input = torch.split(
            routed_input[0],
            split_size_or_sections=routed_input[1].tolist(),
            dim=0,
        )

    out_experts_split = []
    if isinstance(self.down_proj, DTensor):
        down_proj = self.down_proj.to_local()
        gate_up_proj = self.gate_up_proj.to_local()
    else:
        down_proj = self.down_proj
        gate_up_proj = self.gate_up_proj

    for idx, x in enumerate(routed_input):
        gate_up = F.linear(x, gate_up_proj[idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = self.act_fn(gate) * up
        hidden = F.linear(hidden, down_proj[idx])
        out_experts_split.append(hidden)

    return torch.cat(out_experts_split, dim=0)
