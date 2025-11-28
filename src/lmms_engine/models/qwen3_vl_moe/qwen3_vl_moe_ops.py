import inspect
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributed._tensor import DTensor
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModel,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModelOutputWithPast as HFQwen3VLMoeModelOutputWithPast,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextAttention,
    Qwen3VLMoeTextDecoderLayer,
    Qwen3VLMoeTextExperts,
    Qwen3VLMoeTextModel,
    Qwen3VLMoeTextSparseMoeBlock,
    apply_rotary_pos_emb,
    rotate_half,
)
from transformers.utils import is_flash_attn_2_available, is_torchdynamo_compiling

from lmms_engine.parallel.sequence_parallel.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    get_visual_embeds_for_rank,
    pad_and_mask_visual_for_ulysses,
    repeat_kv,
    slice_input_tensor,
    ulysses_pad,
)

from ..sequence_packing_utils import BaseModelOutputWithPastAndRmpad, _unpad_input

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


def _distribute_deepstack_embeds_for_rank(deepstack_embeds, original_mask, sp_size):
    """
    Distribute deepstack embeddings for the current rank based on sequence parallel split.

    Args:
        deepstack_embeds: List of embeddings to distribute
        original_mask: Original mask before padding
        sp_size: Sequence parallel size

    Returns:
        List of distributed embeddings for current rank
    """
    if sp_size <= 1:
        return deepstack_embeds

    return [
        get_visual_embeds_for_rank(
            embed,
            original_mask[..., 0].bool(),
            sp_size=sp_size,
        )
        for embed in deepstack_embeds
    ]


def _aggregate_visual_masks_and_embeds(
    image_mask,
    video_mask,
    deepstack_image_embeds,
    deepstack_video_embeds,
    original_image_mask,
    original_video_mask,
    sp_size,
):
    """
    Aggregate visual position masks and deepstack visual embeddings for both image and video.

    Args:
        image_mask: Image mask tensor
        video_mask: Video mask tensor
        deepstack_image_embeds: Deepstack image embeddings
        deepstack_video_embeds: Deepstack video embeddings
        original_image_mask: Original image mask before rank-specific masking
        original_video_mask: Original video mask before rank-specific masking
        sp_size: Sequence parallel size

    Returns:
        Tuple of (visual_pos_masks, deepstack_visual_embeds)
    """
    image_mask = image_mask[..., 0]
    video_mask = video_mask[..., 0]
    visual_pos_masks = image_mask | video_mask

    # Distribute deepstack embeds for this rank based on original masks
    deepstack_visual_embeds = []
    if sp_size > 1:
        deepstack_image_embeds = _distribute_deepstack_embeds_for_rank(
            deepstack_image_embeds, original_image_mask, sp_size
        )
        deepstack_video_embeds = _distribute_deepstack_embeds_for_rank(
            deepstack_video_embeds, original_video_mask, sp_size
        )

    # Merge image and video embeddings
    image_mask_joint = image_mask[visual_pos_masks]
    video_mask_joint = video_mask[visual_pos_masks]
    for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
        embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
        embed_joint[image_mask_joint, :] = img_embed
        embed_joint[video_mask_joint, :] = vid_embed
        deepstack_visual_embeds.append(embed_joint)

    return visual_pos_masks, deepstack_visual_embeds


def _process_single_visual_modality(mask, deepstack_embeds, original_mask, sp_size):
    """
    Process visual embeddings for a single modality (image or video).

    Args:
        mask: Visual mask tensor
        deepstack_embeds: Deepstack embeddings
        original_mask: Original mask before rank-specific masking
        sp_size: Sequence parallel size

    Returns:
        Tuple of (visual_pos_masks, deepstack_visual_embeds)
    """
    mask = mask[..., 0]
    visual_pos_masks = mask

    # Distribute deepstack embeds for this rank based on original mask
    if sp_size > 1:
        deepstack_embeds = _distribute_deepstack_embeds_for_rank(deepstack_embeds, original_mask, sp_size)

    return visual_pos_masks, deepstack_embeds


@dataclass
class Qwen3VLMoeModelOutputWithPast(HFQwen3VLMoeModelOutputWithPast):
    seq_lens: Optional[torch.IntTensor] = None
    word_idx: Optional[torch.IntTensor] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


def model_forward(
    self: Qwen3VLMoeModel,
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
) -> Union[tuple, Qwen3VLMoeModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if input_ids is not None:
        original_input_ids = input_ids
        input_ids, indices, cu_seq_lens, _ = _unpad_input(input_ids, attention_mask=attention_mask)
        batch_size, seq_length = original_input_ids.shape
    elif inputs_embeds is not None:
        original_inputs_embeds = inputs_embeds
        inputs_embeds, indices, cu_seq_lens, _ = _unpad_input(inputs_embeds, attention_mask=attention_mask)
        batch_size, seq_length, _ = original_inputs_embeds.shape

    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (original_input_ids is not None and original_input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                original_input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    position_ids = (
        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
    )
    if get_ulysses_sequence_parallel_world_size() > 1:
        input_ids, position_ids, pad_size = ulysses_pad(
            input_ids.unsqueeze(0),
            position_ids,
            sp_size=get_ulysses_sequence_parallel_world_size(),
        )
        input_ids = input_ids.squeeze(0)
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    original_image_mask = image_mask.clone() if image_mask is not None else None
    original_video_mask = video_mask.clone() if video_mask is not None else None

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if get_ulysses_sequence_parallel_world_size() > 1:
        sp_size = get_ulysses_sequence_parallel_world_size()
        if image_mask is not None:
            image_mask = pad_and_mask_visual_for_ulysses(image_mask, sp_size=sp_size)
        if video_mask is not None:
            video_mask = pad_and_mask_visual_for_ulysses(video_mask, sp_size=sp_size)

    if image_mask is not None and video_mask is not None:
        visual_pos_masks, deepstack_visual_embeds = _aggregate_visual_masks_and_embeds(
            image_mask,
            video_mask,
            deepstack_image_embeds,
            deepstack_video_embeds,
            original_image_mask,
            original_video_mask,
            sp_size=get_ulysses_sequence_parallel_world_size(),
        )
    elif image_mask is not None:
        visual_pos_masks, deepstack_visual_embeds = _process_single_visual_modality(
            image_mask,
            deepstack_image_embeds,
            original_image_mask,
            sp_size=get_ulysses_sequence_parallel_world_size(),
        )
    elif video_mask is not None:
        visual_pos_masks, deepstack_visual_embeds = _process_single_visual_modality(
            video_mask,
            deepstack_video_embeds,
            original_video_mask,
            sp_size=get_ulysses_sequence_parallel_world_size(),
        )

    if get_ulysses_sequence_parallel_world_size() > 1 and visual_pos_masks is not None:
        visual_pos_masks = slice_input_tensor(visual_pos_masks, dim=0)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        indices=indices,
        cu_seq_lens=cu_seq_lens,
        output_router_logits=output_router_logits,
        **kwargs,
    )

    return Qwen3VLMoeModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
        seq_lens=cu_seq_lens,
        word_idx=indices,
        router_logits=getattr(outputs, "router_logits", None),
    )


def text_model_forward(
    self: Qwen3VLMoeTextModel,
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
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
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

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cu_seq_lens is not None and indices is not None:
            seq_len_for_cache = inputs_embeds.shape[0]
        else:
            seq_len_for_cache = inputs_embeds.shape[1]
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seq_len_for_cache,
            device=inputs_embeds.device,
        )

    if position_ids is None:
        if cu_seq_lens is not None and indices is not None:
            position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        else:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_router_logits = () if output_router_logits else None

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

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

        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

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
    self: Qwen3VLMoeTextDecoderLayer,
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
) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
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

    hidden_states = hidden_states.squeeze(0)
    hidden_states = residual + hidden_states

    if output_router_logits and router_logits is not None:
        return hidden_states, router_logits
    return hidden_states


def attn_forward(
    self: Qwen3VLMoeTextAttention,
    hidden_states: torch.Tensor,
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

    head_dim = self.head_dim
    config = self.config

    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

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


def moe_sparse_layer_forward(
    self: Qwen3VLMoeTextSparseMoeBlock, hidden_states: torch.Tensor, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modified from the original code to support parallelization, similar to the MoE in torchtitan
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    selected_experts = selected_experts.to(torch.float32)

    # Always normalize the routing weights
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    # Calculate the number of tokens per expert
    # [num_tokens on expert_0, num_tokens on expert_1, ...]
    num_tokens_per_expert = torch.histc(selected_experts, bins=self.num_experts, min=0, max=self.num_experts)
    # Histc does not support half tensor or int64, so we cast to float32 and cast back to int64
    selected_experts = selected_experts.to(torch.int64)
    num_tokens_per_expert = num_tokens_per_expert.to(torch.int64)

    # Will need to compute num_tokens * top_k num tokens, sorted by the token index and match the expert order
    token_indices_experts_sorted = torch.argsort(selected_experts.view(-1), stable=True)
    # Get scores for each token
    top_scores_experts_sorted = routing_weights.view(-1)[token_indices_experts_sorted]
    # Divide by top_k to get the token index that matches the expert order
    # [token_index_on_expert_0, token_index_on_expert_1, ...]
    token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

    token_indices_experts_sorted = token_indices_experts_sorted.reshape(-1, 1).expand(-1, hidden_dim)
    routed_input = torch.gather(hidden_states, dim=0, index=token_indices_experts_sorted)

    out_experts_split = self.experts(routed_input, num_tokens_per_expert)

    # Gather the output from the experts
    routed_output = out_experts_split * top_scores_experts_sorted.reshape(-1, 1)
    final_hidden_states = torch.zeros_like(hidden_states)
    final_hidden_states = final_hidden_states.scatter_add(dim=0, index=token_indices_experts_sorted, src=routed_output)

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def experts_forward(self: Qwen3VLMoeTextExperts, *routed_input):
    out_experts_split = []
    if isinstance(self.down_proj, DTensor):
        down_proj = self.down_proj.to_local()
        gate_up_proj = self.gate_up_proj.to_local()
    else:
        down_proj = self.down_proj
        gate_up_proj = self.gate_up_proj

    for idx, x in enumerate(routed_input):
        gate_up = torch.matmul(x, gate_up_proj[idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = up * self.act_fn(gate)
        hidden = torch.matmul(hidden, down_proj[idx])
        out_experts_split.append(hidden)

    return torch.cat(out_experts_split, dim=0)
