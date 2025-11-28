import inspect
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerCausalLMOutputWithPast as HFQwen3OmniMoeModelOutputWithPast,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextAttention,
    Qwen3OmniMoeThinkerTextDecoderLayer,
    Qwen3OmniMoeThinkerTextModel,
    apply_rotary_pos_emb,
    rotate_half,
)
from transformers.utils import is_flash_attn_2_available

from lmms_engine.parallel.sequence_parallel.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    repeat_kv,
    ulysses_pad,
)
from lmms_engine.utils import Logging

from ..sequence_packing_utils import (
    BaseModelOutputWithPastAndRmpad,
    _get_unpad_data,
    _unpad_input,
)

if is_flash_attn_2_available():
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import (
            index_first_axis,
            pad_input,
            rearrange,
            unpad_input,
        )

        _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
    except:
        raise ModuleNotFoundError("flash_attn is not available. Please install it via `pip install flash_attn`.")


def _get_module_attr(module, attr_name):
    """
    Safely get attribute from a module that may be wrapped by FSDP.
    """
    if attr_name in module.__dict__:
        return module.__dict__[attr_name]

    # Try normal attribute access
    if hasattr(module, attr_name):
        return getattr(module, attr_name)

    # Try accessing through parent class attributes (for class-level defaults)
    for cls in type(module).__mro__:
        if attr_name in cls.__dict__:
            return cls.__dict__[attr_name]

    # Try accessing through FSDP wrapped module (FSDP1 style)
    if hasattr(module, "_fsdp_wrapped_module"):
        return getattr(module._fsdp_wrapped_module, attr_name)

    # If still not found, raise error with helpful debugging info
    available_attrs = [x for x in dir(module) if not x.startswith("__")][:20]
    raise AttributeError(
        f"Module {type(module).__name__} has no attribute '{attr_name}'. "
        f"Available attributes (first 20): {available_attrs}"
    )


@dataclass
class Qwen3OmniMoeModelOutputWithPast(HFQwen3OmniMoeModelOutputWithPast):
    seq_lens: Optional[torch.IntTensor] = None
    word_idx: Optional[torch.IntTensor] = None


def text_model_forward(
    self: Qwen3OmniMoeThinkerTextModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPastAndRmpad]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    output_router_logits = (
        output_router_logits if output_router_logits is not None else getattr(self.config, "output_router_logits", True)
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if self.gradient_checkpointing and self.training:
        if use_cache:
            Logging.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cu_seq_lens is not None and indices is not None:
            seq_len_for_cache = inputs_embeds.shape[0]  # 1D case, total unpadded tokens
        else:
            seq_len_for_cache = inputs_embeds.shape[1]  # 2D case, sequence length dimension
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seq_len_for_cache,
            device=inputs_embeds.device,
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        if cu_seq_lens is not None and indices is not None:
            # if use rmpad, position ids is [3, 1, total_non_pad_tokens]
            # but lce_forward already provides position_ids
            position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        else:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        # if position_ids is provided but only 2D [batch, seq_len], expand to 3D [3, batch, seq_len]
        # by adding the TMRoPE dimension at the front
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_router_logits = () if output_router_logits else None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(
                decoder_layer.__call__,
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                past_key_values,
                cache_position,
                cu_seq_lens,
                indices,
                output_router_logits,
                use_reentrant=False,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
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

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_attentions] if v is not None)

    return BaseModelOutputWithPastAndRmpad(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        seq_lens=cu_seq_lens,
        word_idx=indices,
        router_logits=all_router_logits if output_router_logits else None,
    )


def decoder_layer_forward(
    self: Qwen3OmniMoeThinkerTextDecoderLayer,
    hidden_states: torch.Tensor,  # should be 2D with rmpad
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
    residual = hidden_states
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
    self: Qwen3OmniMoeThinkerTextAttention,
    hidden_states: torch.Tensor,  # should be 2D with rmpad
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    **kwargs,
):
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    bsz = hidden_states.shape[0]
    if cu_seq_lens is not None:
        q_len = (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()
    else:
        q_len = hidden_states.shape[0] if hidden_states.ndim == 2 else hidden_states.shape[1]
    kv_seq_len = q_len

    head_dim = _get_module_attr(self, "head_dim")
    config = _get_module_attr(self, "config")

    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    # Project and normalize queries/keys
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
    value_states = self.v_proj(hidden_states).view(hidden_shape)
    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"
        # query_states, key_states, value_states are (total_tokens, num_heads, head_dim)
        repeats = max(ulysses_sp_size // key_states.size(1), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=0, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=0, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=0, head_dim=1)

    query_states = query_states.unsqueeze(0).transpose(1, 2)
    key_states = key_states.unsqueeze(0).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    max_seqlen = torch.diff(cu_seq_lens).max().item() if cu_seq_lens is not None else None

    query_states = query_states.transpose(1, 2).squeeze(0)
    key_states = key_states.transpose(1, 2).squeeze(0)

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
        softmax_scale=head_dim**-0.5,
        dropout_p=0.0,
    )

    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, None


def moe_sparse_layer_forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    import torch.nn.functional as F

    gate = _get_module_attr(self, "gate")
    num_experts = gate.out_features
    num_experts_per_tok = _get_module_attr(self, "num_experts_per_tok")
    norm_topk_prob = _get_module_attr(self, "norm_topk_prob")

    # Handle both 3D [batch, seq, hidden] and 1D [total_tokens, hidden] inputs
    is_3d = hidden_states.ndim == 3
    if is_3d:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
    else:
        hidden_dim = hidden_states.shape[-1]

    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)

    # Convert to float32 for histogram
    selected_experts = selected_experts.to(torch.float32)

    if norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

    routing_weights = routing_weights.to(hidden_states.dtype)

    num_tokens_per_expert = torch.histc(selected_experts, bins=num_experts, min=0, max=num_experts)
    selected_experts = selected_experts.to(torch.int64)
    num_tokens_per_expert = num_tokens_per_expert.to(torch.int64)

    # tokens need to be reordered so all tokens for expert_0 come first,then expert_1, etc.
    token_indices_experts_sorted = torch.argsort(selected_experts.view(-1), stable=True)

    top_scores_experts_sorted = routing_weights.view(-1)[token_indices_experts_sorted]
    token_indices_experts_sorted = token_indices_experts_sorted // num_experts_per_tok
    token_indices_experts_sorted = token_indices_experts_sorted.reshape(-1, 1).expand(-1, hidden_dim)

    routed_input = torch.gather(hidden_states, dim=0, index=token_indices_experts_sorted)

    # Check if EP is enabled by checking if expert params are DTensors
    from torch.distributed._tensor import DTensor

    if isinstance(self.experts.gate_proj, DTensor):
        # EP is enabled - ParallelStyle._input_fn will handle the split
        out_experts_split = self.experts(routed_input, num_tokens_per_expert)
    else:
        # EP is disabled, need to split routed_input manually
        routed_input_split = torch.split(
            routed_input,
            split_size_or_sections=num_tokens_per_expert.tolist(),
            dim=0,
        )
        out_experts_split = self.experts(*routed_input_split)

    routed_output = out_experts_split * top_scores_experts_sorted.reshape(-1, 1)
    final_hidden_states = torch.zeros_like(hidden_states)
    final_hidden_states = final_hidden_states.scatter_add(dim=0, index=token_indices_experts_sorted, src=routed_output)

    # If input was 3D, reshape back
    if is_3d:
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    return final_hidden_states, router_logits
