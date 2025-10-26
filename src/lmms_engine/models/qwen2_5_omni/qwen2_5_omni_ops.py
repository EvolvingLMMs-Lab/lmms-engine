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
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAttention,
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniAudioEncoderLayer,
    Qwen2_5OmniDecoderLayer,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast as HFQwen2_5OmniModelOutputWithPast,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
    Qwen2_5OmniVisionEncoder,
    apply_multimodal_rotary_pos_emb,
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


@dataclass
class Qwen2_5OmniModelOutputWithPast(HFQwen2_5OmniModelOutputWithPast):
    seq_lens: Optional[torch.IntTensor] = None
    word_idx: Optional[torch.IntTensor] = None


def text_model_forward(
    self: Qwen2_5OmniThinkerTextModel,
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
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
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
        return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_attentions] if v is not None)

    return BaseModelOutputWithPastAndRmpad(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        seq_lens=cu_seq_lens,
        word_idx=indices,
    )


def decoder_layer_forward(
    self: Qwen2_5OmniDecoderLayer,
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


def attn_forward(
    self: Qwen2_5OmniAttention,
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
    query_states = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"
        repeats = max(ulysses_sp_size // key_states.size(1), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)
        # Testing
        # before all to all Q: torch.Size([22541, 28, 128]), K: torch.Size([22541, 4, 128]), V: torch.Size([22541, 4, 128])
        # after all to all Q: torch.Size([45082, 14, 128]), K: torch.Size([45082, 2, 128]), V: torch.Size([45082, 2, 128])
        query_states = gather_seq_scatter_heads(query_states, seq_dim=0, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=0, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=0, head_dim=1)

    query_states = query_states.unsqueeze(0).transpose(1, 2)
    key_states = key_states.unsqueeze(0).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
    )

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
        softmax_scale=self.head_dim**-0.5,
        dropout_p=0.0,
    )

    if ulysses_sp_size > 1:
        # [45082, 14, 128] -> [22541, 28, 128]
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1)

    attn_output = attn_output.reshape(-1, self.hidden_size).contiguous()

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
