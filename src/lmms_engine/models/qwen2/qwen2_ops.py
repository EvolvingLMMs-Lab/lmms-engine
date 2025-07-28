import inspect
import warnings
from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.utils import is_flash_attn_2_available, logging

from lmms_engine.utils import Logging

from ..sequence_packing_utils import (
    BaseModelOutputWithPastAndRmpad,
    _get_unpad_data,
    _unpad_input,
    apply_rotary_pos_emb_unpad,
)

logger = logging.get_logger(__name__)


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
except:
    apply_rotary_emb_func = None
    logger.warning_once(
        "fail to load faster rotary ops, use PyTorch version by default. Please check image version"
    )


# The forward func for the base model of a LM
def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
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
        if inputs_embeds.dim() == 3:
            batch_size, seq_length, _ = inputs_embeds.shape
        elif inputs_embeds.dim() == 2:
            batch_size, seq_length = inputs_embeds.shape
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

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        # 1 * 5695(seq_len) [0,1,2,3,4,5,...... 5000, 5001,5002]
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
    if position_ids.shape[0] != inputs_embeds.shape[0]:
        position_ids, _, _, _ = _unpad_input(
            position_ids.view(1, position_ids.shape[-1], 1).repeat(
                inputs_embeds.shape[0], 1, 1
            ),
            attention_mask,
        )
    else:
        position_ids, _, _, _ = _unpad_input(position_ids.unsqueeze(-1), attention_mask)

    # If already handled in the outside to optimize scattered performance
    # Then we do not unpad here
    if cu_seq_lens is None:
        inputs_embeds, indices, cu_seq_lens, _ = _unpad_input(
            inputs_embeds.unsqueeze(-1), attention_mask
        )
        inputs_embeds, position_ids = inputs_embeds.squeeze(-1), position_ids.squeeze(
            -1
        )

    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )

    causal_mask = self._update_causal_mask(
        attention_mask,
        inputs_embeds,
        cache_position,
        past_key_values,
        output_attentions,
    )
    attention_mask = causal_mask

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
                attention_mask,
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
                attention_mask,
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

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache and next_decoder_cache is not None
            else next_decoder_cache
        )

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
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
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


# The attn forward func for the LM
def attn_forward(
    self,
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
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    bsz = hidden_states.shape[0]
    q_len = torch.max(position_ids).item() + 1
    kv_seq_len = q_len
    query_states = self.q_proj(hidden_states).view(
        -1, self.config.num_attention_heads, self.head_dim
    )
    key_states = self.k_proj(hidden_states).view(
        -1, self.config.num_key_value_heads, self.head_dim
    )
    value_states = self.v_proj(hidden_states).view(
        -1, self.config.num_key_value_heads, self.head_dim
    )
    cos, sin = position_embeddings

    if apply_rotary_emb_func is not None:
        cos = cos.squeeze().index_select(
            dim=0, index=position_ids.squeeze()
        )  # [total_bs_seq, head_dim]
        sin = sin.squeeze().index_select(dim=0, index=position_ids.squeeze())
        # cos = cos.squeeze().index_select(dim=0, index=position_ids.squeeze()).unsqueeze(1) # [total_bs_seq, 1, head_dim]
        # sin = sin.squeeze().index_select(dim=0, index=position_ids.squeeze()).unsqueeze(1)
        query_states = apply_rotary_emb_func(
            query_states.unsqueeze(0),
            cos[:, : self.head_dim // 2],
            sin[:, : self.head_dim // 2],
            inplace=True,
        ).squeeze(0)
        key_states = apply_rotary_emb_func(
            key_states.unsqueeze(0),
            cos[:, : self.head_dim // 2],
            sin[:, : self.head_dim // 2],
            inplace=True,
        ).squeeze(0)
        # print(query_states.shape, key_states.shape, value_states.shape)
        # assert 1 > 2
    else:
        query_states, key_states = apply_rotary_pos_emb_unpad(
            query_states, key_states, cos, sin, position_ids
        )

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(attention_mask[:, -1:])],
                    dim=-1,
                )

        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    max_seqlen = (
        torch.diff(cu_seq_lens).max().item() if cu_seq_lens is not None else None
    )
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

    attn_output = attn_output.reshape(-1, self.config.hidden_size).contiguous()

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
