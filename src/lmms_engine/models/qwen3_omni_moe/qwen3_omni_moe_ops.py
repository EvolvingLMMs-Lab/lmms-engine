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

# Import Qwen3 Omni MoE classes from transformers
# Note: Using Thinker component classes as they contain the MoE layers
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextAttention as Qwen3OmniMoeAttention,
    Qwen3OmniMoeThinkerTextDecoderLayer as Qwen3OmniMoeDecoderLayer,
    Qwen3OmniMoeThinkerForConditionalGeneration as Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel as Qwen3OmniMoeTextModel,
    apply_rotary_pos_emb,
    rotate_half,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerCausalLMOutputWithPast as HFQwen3OmniMoeModelOutputWithPast,
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

        _flash_supports_window_size = "window_size" in list(
            inspect.signature(flash_attn_func).parameters
        )
    except:
        raise ModuleNotFoundError(
            "flash_attn is not available. Please install it via `pip install flash_attn`."
        )


def _get_module_attr(module, attr_name):
    """
    Safely get attribute from a module that may be wrapped by FSDP.

    Args:
        module: The module (possibly FSDP-wrapped)
        attr_name: Name of the attribute to retrieve

    Returns:
        The attribute value
    """
    # FSDP2 uses a state-based pattern, not a wrapper pattern
    # The module's __dict__ should still contain the original attributes
    # Try direct access via __dict__ first (bypasses descriptor protocol)
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
    if hasattr(module, '_fsdp_wrapped_module'):
        return getattr(module._fsdp_wrapped_module, attr_name)

    # If still not found, raise error with helpful debugging info
    available_attrs = [x for x in dir(module) if not x.startswith('__')][:20]
    raise AttributeError(
        f"Module {type(module).__name__} has no attribute '{attr_name}'. "
        f"Available attributes (first 20): {available_attrs}"
    )


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.

    This function is adapted from Qwen2.5-Omni for compatibility with the existing codebase.
    In Qwen3-Omni, the multimodal logic is handled differently within the RoPE embedding class,
    but we maintain this interface for consistency with the current implementation.

    Multimodal 3D rotary position embedding extends 1D RoPE for vision and text embeddings:
    - Vision (images/videos): Apply RoPE on temporal, height, and width dimensions separately
    - Text: Standard 1D RoPE (T, H, W indices are identical)

    The channel dimension is split into 3 chunks for temporal, height, and width RoPE.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine part of rotary embedding
        sin: Sine part of rotary embedding
        mrope_section: List of channel dimensions [temporal, height, width] for multimodal RoPE
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting (default: 1 for head dim)

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied
    """
    mrope_section = mrope_section * 2  # Double for cos/sin interleaving
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    # Apply rotary embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@dataclass
class Qwen3OmniMoeModelOutputWithPast(HFQwen3OmniMoeModelOutputWithPast):
    seq_lens: Optional[torch.IntTensor] = None
    word_idx: Optional[torch.IntTensor] = None


def text_model_forward(
    self: Qwen3OmniMoeTextModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPastAndRmpad]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

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
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if cu_seq_lens is not None and indices is not None:
            seq_len_for_cache = inputs_embeds.shape[0]  # 1D case, total unpadded tokens
        else:
            seq_len_for_cache = inputs_embeds.shape[
                1
            ]  # 2D case, sequence length dimension
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
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
    elif position_ids.dim() == 2:
        # if position_ids is provided but only 2D [batch, seq_len], expand to 3D [3, batch, seq_len]
        # by adding the TMRoPE dimension at the front
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cu_seq_lens,
                indices,
                position_embeddings,
                use_reentrant=False,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cu_seq_lens=cu_seq_lens,
                indices=indices,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                **kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, past_key_values, all_hidden_states, all_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndRmpad(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        seq_lens=cu_seq_lens,
        word_idx=indices,
    )


def decoder_layer_forward(
    self: Qwen3OmniMoeDecoderLayer,
    hidden_states: torch.Tensor,  # should be 2D with rmpad
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cu_seq_lens=cu_seq_lens,
        indices=indices,
        position_embeddings=position_embeddings,
    )

    hidden_states = residual + hidden_states

    # MoE block (different from regular MLP)
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    # Check if MoE block - may return router logits
    if hasattr(self.mlp, 'gate') or hasattr(self.mlp, 'router'):
        # MoE forward - may return (hidden_states, router_logits)
        mlp_output = self.mlp(hidden_states)
        if isinstance(mlp_output, tuple):
            hidden_states, router_logits = mlp_output
        else:
            hidden_states = mlp_output
            router_logits = None
    else:
        # Regular MLP
        hidden_states = self.mlp(hidden_states)
        router_logits = None

    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    if router_logits is not None:
        outputs += (router_logits,)

    return outputs


def attn_forward(
    self: Qwen3OmniMoeAttention,
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
        q_len = (
            hidden_states.shape[0]
            if hidden_states.ndim == 2
            else hidden_states.shape[1]
        )
    kv_seq_len = q_len

    # Get attributes safely (handles FSDP wrapping)
    # These attributes are set in the attention module's __init__
    head_dim = _get_module_attr(self, 'head_dim')
    config = _get_module_attr(self, 'config')

    # These come from config, not set as instance attributes
    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads

    # Compute shapes for projections
    # input_shape tracks the original input dimensions (before head splitting)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    # Project and normalize queries/keys (Qwen3 uses per-head normalization)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
    value_states = self.v_proj(hidden_states).view(hidden_shape)
    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        assert (
            position_ids is not None
        ), "position_ids is required for Ulysses sequence parallelism"
        # query_states, key_states, value_states are (total_tokens, num_heads, head_dim)
        repeats = max(ulysses_sp_size // key_states.size(1), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)
        # Scatter sequence across ranks, gather heads
        # before all to all Q: torch.Size([22541, 28, 128]), K: torch.Size([22541, 4, 128]), V: torch.Size([22541, 4, 128])
        # after all to all Q: torch.Size([45082, 14, 128]), K: torch.Size([45082, 2, 128]), V: torch.Size([45082, 2, 128])
        query_states = gather_seq_scatter_heads(query_states, seq_dim=0, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=0, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=0, head_dim=1)

    # Reshape for RoPE application: add batch dim and transpose to (1, num_heads, seq, head_dim)
    query_states = query_states.unsqueeze(0).transpose(1, 2)
    key_states = key_states.unsqueeze(0).transpose(1, 2)

    cos, sin = position_embeddings
    # Note: Qwen3-Omni MoE uses standard apply_rotary_pos_emb, not multimodal version
    # The rotary embedding class already handles multimodal aspect with apply_interleaved_mrope
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    max_seqlen = (
        torch.diff(cu_seq_lens).max().item() if cu_seq_lens is not None else None
    )

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
        # [45082, 14, 128] -> [22541, 28, 128]
        # Restore original sequence distribution
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1)

    # Reshape back to original input shape with concatenated heads
    # After flash attention: (total_tokens, num_heads, head_dim)
    # Target: (total_tokens, num_heads * head_dim) for o_proj
    # Use input_shape to handle both 2D and 3D inputs gracefully
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    # Project concatenated heads to hidden_size
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def moe_sparse_layer_forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for Qwen3-Omni MoE sparse layer with expert parallelism support.

    This function handles token routing and expert computation for Mixture-of-Experts layers.
    It supports both standard 3D tensors and rmpad 1D tensors for sequence packing.

    Algorithm:
    1. Compute router logits for all tokens
    2. Select top-K experts per token via softmax + topk
    3. Count tokens assigned to each expert (using histogram)
    4. Sort tokens by expert assignment for batched processing
    5. Dispatch to experts (with optional all-to-all for expert parallelism)
    6. Apply routing weights and scatter results back to original positions
    7. Return hidden states and router logits (for auxiliary loss)

    Args:
        self: Qwen3OmniMoeThinkerTextSparseMoeBlock instance
        hidden_states: Input tensor, shape [batch, seq_len, hidden_dim] or [total_tokens, hidden_dim] (rmpad)
        **kwargs: Additional arguments (unused, for compatibility)

    Returns:
        Tuple of (final_hidden_states, router_logits):
            final_hidden_states: Tensor with same shape as input
            router_logits: Tensor [batch*seq_len, num_experts] for load balancing loss

    Expert Parallelism:
        When EP is enabled, the `self.experts` forward is intercepted by
        `Qwen3OmniMoeParallelStyle`, which performs all-to-all token exchange
        across EP ranks before expert computation.

    Multimodal Support:
        This function treats all tokens identically (text, vision, audio) - the routing
        mechanism doesn't distinguish between modalities. Tokens are routed purely based
        on their hidden representations.
    """
    # Import here to avoid circular dependency
    import torch.nn.functional as F

    # Get MoE configuration attributes safely (handles FSDP wrapping)
    # Note: num_experts is not stored as instance attr, get from gate output features
    gate = _get_module_attr(self, 'gate')
    num_experts = gate.out_features
    num_experts_per_tok = _get_module_attr(self, 'num_experts_per_tok')
    norm_topk_prob = _get_module_attr(self, 'norm_topk_prob')

    # Handle both 3D [batch, seq, hidden] and 1D [total_tokens, hidden] inputs
    # rmpad mode uses 1D, standard mode uses 3D
    is_3d = hidden_states.ndim == 3
    if is_3d:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)  # Flatten to [batch*seq, hidden]
    else:
        hidden_dim = hidden_states.shape[-1]

    # === Step 1: Compute Router Logits ===
    # router_logits: [num_tokens, num_experts]
    # Determines which experts each token should be routed to
    router_logits = self.gate(hidden_states)

    # === Step 2: Top-K Expert Selection ===
    # Compute routing weights via softmax and select top-K experts per token
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, num_experts_per_tok, dim=-1
    )

    # Convert to float32 for histogram (histc doesn't support half precision)
    selected_experts = selected_experts.to(torch.float32)

    # === Step 3: Normalize Top-K Probabilities (Optional) ===
    # Some MoE variants normalize the routing weights after top-K selection
    # This ensures routing weights sum to 1.0 for each token
    if norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

    routing_weights = routing_weights.to(hidden_states.dtype)

    # === Step 4: Count Tokens Per Expert ===
    # Use histogram to efficiently count how many tokens are assigned to each expert
    # This is used for splitting tokens in the next step
    num_tokens_per_expert = torch.histc(
        selected_experts, bins=num_experts, min=0, max=num_experts
    )
    # Cast back to int64 for indexing operations
    selected_experts = selected_experts.to(torch.int64)
    num_tokens_per_expert = num_tokens_per_expert.to(torch.int64)

    # === Step 5: Sort Tokens by Expert Assignment ===
    # Tokens need to be reordered so all tokens for expert_0 come first,
    # then expert_1, etc. This enables efficient batched expert computation
    # Note: Each token appears top_k times (once for each selected expert)
    token_indices_experts_sorted = torch.argsort(selected_experts.view(-1), stable=True)

    # Get routing weights in the same sorted order
    top_scores_experts_sorted = routing_weights.view(-1)[token_indices_experts_sorted]

    # Recover original token indices
    # Division by top_k maps back: if token_i selects 2 experts, it appears twice,
    # dividing by top_k gives the original token index
    token_indices_experts_sorted = token_indices_experts_sorted // num_experts_per_tok

    # === Step 6: Gather Tokens in Expert-Sorted Order ===
    # Expand indices to match hidden_dim for gathering
    token_indices_experts_sorted = token_indices_experts_sorted.reshape(-1, 1).expand(
        -1, hidden_dim
    )
    routed_input = torch.gather(hidden_states, dim=0, index=token_indices_experts_sorted)

    # === Step 7: Expert Computation ===
    # Call experts module - this is where expert parallelism happens if enabled
    # Input: (routed_input, num_tokens_per_expert)
    #   - routed_input: [total_routed_tokens, hidden_dim] (tokens sorted by expert)
    #   - num_tokens_per_expert: [num_experts] token counts per expert
    # If EP enabled: Qwen3OmniMoeParallelStyle._input_fn intercepts, does all-to-all and splits
    # If EP disabled: We need to split manually before calling experts
    # Output: [total_routed_tokens, hidden_dim] (expert outputs)

    # Check if EP is enabled by checking if expert params are DTensors
    from torch.distributed._tensor import DTensor
    if isinstance(self.experts.gate_proj, DTensor):
        # EP is enabled - ParallelStyle._input_fn will handle the split
        out_experts_split = self.experts(routed_input, num_tokens_per_expert)
    else:
        # EP is disabled - need to split routed_input manually
        # Split into separate tensors for each expert based on token counts
        routed_input_split = torch.split(
            routed_input,
            split_size_or_sections=num_tokens_per_expert.tolist(),
            dim=0,
        )
        # Pass split tensors as separate arguments
        out_experts_split = self.experts(*routed_input_split)

    # === Step 8: Apply Routing Weights ===
    # Weight each expert's output by its routing weight for the token
    routed_output = out_experts_split * top_scores_experts_sorted.reshape(-1, 1)

    # === Step 9: Scatter-Add to Combine Results ===
    # Each token has outputs from top_k experts - combine them via scatter-add
    # This accumulates contributions from all selected experts per token
    final_hidden_states = torch.zeros_like(hidden_states)
    final_hidden_states = final_hidden_states.scatter_add(
        dim=0, index=token_indices_experts_sorted, src=routed_output
    )

    # === Step 10: Restore Original Shape ===
    # If input was 3D, reshape back
    if is_3d:
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

    # Return both hidden states and router logits
    # Router logits are used for auxiliary load balancing loss
    return final_hidden_states, router_logits
