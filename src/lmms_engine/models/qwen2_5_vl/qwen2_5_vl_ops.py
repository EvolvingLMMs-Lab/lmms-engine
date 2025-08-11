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
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLModel,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModelOutputWithPast as HFQwen2_5_VLModelOutputWithPast,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLTextModel,
    apply_multimodal_rotary_pos_emb,
    rotate_half,
)
from transformers.utils import is_flash_attn_2_available, logging

from lmms_engine.parallel.sequence_parallel.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad,
)
from lmms_engine.utils import Logging

from ..sequence_packing_utils import (
    BaseModelOutputWithPastAndRmpad,
    _get_unpad_data,
    _unpad_input,
)

logger = logging.get_logger(__name__)


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


@dataclass
class Qwen2_5_VLModelOutputWithPast(HFQwen2_5_VLModelOutputWithPast):
    """
    Base class for the output of the Qwen2.5-VL model with past key values.
    It extends the HFQwen2_5_VLModelOutputWithPast to include rope_deltas.
    """

    seq_lens: Optional[torch.IntTensor] = None
    word_idx: Optional[torch.IntTensor] = None


# VL Model forward
def vl_model_forward(
    self: Qwen2_5_VLModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    audio_values: Optional[torch.FloatTensor] = None,
    audio_attention_mask: Optional[torch.Tensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
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
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    batch_size, seq_length = input_ids.shape
    original_input_ids = input_ids

    # Unpad the input ids here
    input_ids, indices, cu_seq_lens, _ = _unpad_input(
        input_ids, attention_mask=attention_mask
    )

    if attention_mask is not None:
        attention_mask = attention_mask.to(input_ids.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            cache_position is not None and cache_position[0] == 0
        ) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                original_input_ids,  # Here we use the padded input ids
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            delta = (
                (cache_position[0] + self.rope_deltas).to(input_ids.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    position_ids = (
        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
        .transpose(0, 1)
        .unsqueeze(1)
    )

    if get_ulysses_sequence_parallel_world_size() > 1:
        # Pad the input ids and position ids if the sequence parallelism is used
        input_ids, position_ids, pad_size = ulysses_pad(
            input_ids.unsqueeze(0),
            position_ids,
            sp_size=get_ulysses_sequence_parallel_world_size(),
        )
        input_ids = input_ids.squeeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == self.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == self.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # Embed audio features
    if audio_values is not None:
        audio_features, audio_output_lengths = self.prepare_audio_values(
            audio_values, audio_attention_mask
        )
        n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
        n_audio_features = audio_output_lengths.sum()
        if n_audio_tokens != n_audio_features:
            raise ValueError(
                f"Audio features and image tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
            )
        audio_mask = (
            (input_ids == self.config.audio_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        # Audio feature is in (bs, max_seq_len, hidden_size)
        # If directly masked scatter, the embed will be place one by one (order is incorret)
        # We remove the padded values first
        unpadded_audio_features = [
            audio_feat[:audio_output_length]
            for audio_feat, audio_output_length in zip(
                audio_features, audio_output_lengths
            )
        ]
        # Concat the audio features
        # Should exactly have audio_mask.sum() values
        unpadded_audio_features = torch.concatenate(unpadded_audio_features, dim=0)
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_mask, unpadded_audio_features
        )

    kwargs = {"cache_position": cache_position}
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        indices=indices,
        cu_seq_lens=cu_seq_lens,
        **kwargs,
    )
    seq_lens = outputs.get("seq_lens", None)
    word_idx = outputs.get("word_idx", None)
    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
        seq_lens=seq_lens,
        word_idx=word_idx,
    )
    return output if return_dict else output.to_tuple()


# Text Model forward
def text_model_forward(
    self: Qwen2_5_VLTextModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    indices: Optional[torch.LongTensor] = None,
    cu_seq_lens: Optional[torch.LongTensor] = None,
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
    # use_cache = use_cache if use_cache is not None else self.config.use_cache
    # use_rmpad = use_rmpad if use_rmpad is not None else self.config.use_rmpad

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        seq_length, hidden_size = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    # Hardcode from https://github.com/huggingface/transformers/blob/c9d1e5238a752813ba91a8751a638a09b5efbb73/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1164-L1174
    # Because in forward we don't need to handle cache
    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )
    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(
            3, inputs_embeds.shape[0], -1
        )
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
    batch_size, seq_length = attention_mask.shape
    causal_mask = attention_mask
    # inputs_embeds, indices, cu_seq_lens, _ = _unpad_input(
    #     inputs_embeds.unsqueeze(-1), attention_mask
    # )
    # inputs_embeds = inputs_embeds.squeeze(-1)

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                None,
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
                causal_mask,
                position_ids,
                None,
                output_attentions,
                indices=indices,
                cu_seq_lens=cu_seq_lens,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPastAndRmpad(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        seq_lens=cu_seq_lens,
        word_idx=indices,
    )


# The decoder forward func for the LM
def decoder_layer_forward(
    self: Qwen2_5_VLDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    position_embeddings: Optional[
        List[Tuple[torch.Tensor, torch.Tensor]]
    ] = None,  # necessary, but kept here for BC
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cu_seq_lens=cu_seq_lens,
        indices=indices,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(
        slen, num_key_value_heads, n_rep, head_dim
    )
    return hidden_states.reshape(slen, num_key_value_heads * n_rep, head_dim)


# The attn forward func for the LM
def attn_forward(
    self: Qwen2_5_VLAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    position_embeddings: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    **kwargs,
):
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    bsz = hidden_states.shape[0]
    q_len = torch.max(position_ids).item() + 1
    kv_seq_len = q_len
    query_states = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(
        -1, self.num_key_value_heads, self.head_dim
    )
    value_states = self.v_proj(hidden_states).view(
        -1, self.num_key_value_heads, self.head_dim
    )
    # Because the input can be padded, the absolute sequence length depends on the max position id.
    cos, sin = position_embeddings
    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        assert (
            position_ids is not None
        ), "position_ids is required for Ulysses sequence parallelism"

        # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
        # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
        # For example:
        # - nheads_k=4, sp=8, repeats=2
        # - nheads_k=8, sp=8, repeats=1
        # - nheads_k=16, sp=8, repeats=1
        repeats = max(ulysses_sp_size // key_states.size(1), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (seq_len/n, n_head, head_dim) -> (seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=0, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=0, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=0, head_dim=1)
        # Cat the cu_seq_lens to the max seq len if padding is used
        if cu_seq_lens.max().item() < query_states.shape[0]:
            cu_seq_lens = torch.cat(
                [
                    cu_seq_lens,
                    torch.tensor(
                        [query_states.shape[0]],
                        device=cu_seq_lens.device,
                        dtype=cu_seq_lens.dtype,
                    ),
                ]
            )

    # Unsqueeze the first dim to apply pos embeds
    query_states = query_states.unsqueeze(0).transpose(1, 2)
    key_states = key_states.unsqueeze(0).transpose(1, 2)
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
    )

    max_seqlen = (
        torch.diff(cu_seq_lens).max().item() if cu_seq_lens is not None else None
    )

    # Reshape to the expected shape for Flash Attention
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
        softmax_scale=self.head_dim**-0.5,
        dropout_p=0.0,
    )

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1)

    attn_output = attn_output.reshape(-1, self.hidden_size).contiguous()

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
