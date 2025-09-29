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
try:
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        apply_multimodal_rotary_pos_emb,
        rotate_half,
    )
    # Try to import model components - some may not be directly accessible
    try:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniAttention,
            Qwen2_5OmniAudioEncoder,
            Qwen2_5OmniAudioEncoderLayer,
            Qwen2_5OmniDecoderLayer,
            Qwen2_5OmniVisionEncoder,
        )
    except ImportError:
        # Define dummy classes as placeholders
        class Qwen2_5OmniAttention: pass
        class Qwen2_5OmniAudioEncoder: pass
        class Qwen2_5OmniAudioEncoderLayer: pass
        class Qwen2_5OmniDecoderLayer: pass
        class Qwen2_5OmniVisionEncoder: pass

    # Import the correct text model class
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerTextModel

except ImportError as e:
    print(f"Warning: Could not import Qwen2.5-Omni components: {e}")
    # Create dummy classes
    class Qwen2_5OmniAttention: pass
    class Qwen2_5OmniAudioEncoder: pass
    class Qwen2_5OmniAudioEncoderLayer: pass
    class Qwen2_5OmniDecoderLayer: pass
    class Qwen2_5OmniVisionEncoder: pass
    class Qwen2_5OmniThinkerTextModel: pass
    def apply_multimodal_rotary_pos_emb(*args, **kwargs): pass
    def rotate_half(*args, **kwargs): pass

# Try to import the main model
try:
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
    # Get the core model class
    Qwen2_5OmniThinkerModel = type(Qwen2_5OmniThinkerForConditionalGeneration({}).model)
except:
    class Qwen2_5OmniThinkerModel: pass

try:
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniModelOutputWithPast as HFQwen2_5OmniModelOutputWithPast,
    )
except ImportError:
    # Create a dummy output class
    from dataclasses import dataclass
    from transformers.utils import ModelOutput

    @dataclass
    class HFQwen2_5OmniModelOutputWithPast(ModelOutput):
        last_hidden_state: torch.FloatTensor = None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None
from transformers.utils import is_flash_attn_2_available, logging

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
class Qwen2_5OmniModelOutputWithPast(HFQwen2_5OmniModelOutputWithPast):
    """
    Base class for the output of the Qwen2.5-Omni model with past key values.
    It extends the HFQwen2_5OmniModelOutputWithPast to include rope_deltas.
    """

    seq_lens: Optional[torch.IntTensor] = None
    word_idx: Optional[torch.IntTensor] = None


# Note: Qwen2_5OmniThinkerForConditionalGeneration uses lce_forward from qwen2_5_omni_liger.py
# The unpacking is handled at the text model level when rmpad is enabled

# Unused function - kept for reference
# The main model forward is handled by lce_forward which properly manages multimodal inputs
def _omni_main_model_forward_reference(
    self,  # Qwen2_5OmniThinkerForConditionalGeneration
    input_ids: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
    audio_feature_lengths: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_audio_in_video: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    video_second_per_grid: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[tuple, Qwen2_5OmniModelOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    # Unpad the input ids here for rmpad
    original_input_ids = input_ids
    if input_ids is not None and attention_mask is not None:
        input_ids, indices, cu_seq_lens, _ = _unpad_input(
            input_ids, attention_mask=attention_mask
        )
    else:
        indices = None
        cu_seq_lens = None

    # Calculate position ids and rope deltas for multimodal inputs
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                original_input_ids if original_input_ids is not None else input_ids,
                image_grid_thw,
                video_grid_thw,
                video_second_per_grid,
                attention_mask,
                input_features,
                feature_attention_mask,
                audio_feature_lengths,
                use_audio_in_video,
            )
            self.rope_deltas = rope_deltas
        else:
            # Use pre-calculated rope-deltas to get the correct position ids
            delta = (
                (cache_position[0] + self.rope_deltas).to(input_ids.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # Process vision if provided
    if pixel_values is not None or pixel_values_videos is not None:
        image_features = self.get_image_features(pixel_values, image_grid_thw) if pixel_values is not None else None
        video_features = self.get_video_features(pixel_values_videos, video_grid_thw) if pixel_values_videos is not None else None
    else:
        image_features = None
        video_features = None

    # Process audio if provided
    if input_features is not None:
        audio_features = self.get_audio_features(
            input_features, feature_attention_mask, audio_feature_lengths, use_audio_in_video
        )
    else:
        audio_features = None

    # Call the text model with proper parameters
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        image_features=image_features,
        video_features=video_features,
        audio_features=audio_features,
        rope_deltas=rope_deltas,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        cu_seq_lens=cu_seq_lens,
        indices=indices,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Handle loss calculation with rmpad
        if indices is not None:
            # Unpad labels
            labels = labels.view(-1)[indices.long()]
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # Return the proper output format
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerCausalLMOutputWithPast
    return Qwen2_5OmniThinkerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )


# Text Model forward
def text_model_forward(
    self: Qwen2_5OmniThinkerTextModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_embeds: Optional[torch.FloatTensor] = None,
    audio_embeds: Optional[torch.FloatTensor] = None,
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

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # Merge multimodal embeddings
    if vision_embeds is not None:
        inputs_embeds = self.merge_vision_embeddings(
            inputs_embeds, vision_embeds, input_ids
        )
    if audio_embeds is not None:
        inputs_embeds = self.merge_audio_embeddings(
            inputs_embeds, audio_embeds, input_ids
        )

    # Apply rmpad if cu_seq_lens and indices are provided
    if cu_seq_lens is not None and indices is not None:
        # inputs_embeds are already unpaded from merge functions
        bs, seqlen = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]

        # Prepare position ids
        if position_ids is not None:
            position_ids = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
            ).transpose(0, 1)
            original_position_ids = position_ids

            # Pad the position ids according to the original input ids for Ulysses
            if get_ulysses_sequence_parallel_world_size() > 1:
                input_ids_rmpad = input_ids.unsqueeze(0) if input_ids is not None else inputs_embeds.unsqueeze(0)
                _, position_ids, pad_size = ulysses_pad(
                    input_ids_rmpad,
                    original_position_ids,
                    sp_size=get_ulysses_sequence_parallel_world_size(),
                )

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    # Initialize position_ids if None
    if position_ids is None:
        seq_length = inputs_embeds.shape[1]
        batch_size = inputs_embeds.shape[0]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=inputs_embeds.device
        ).view(1, -1).expand(batch_size, -1)
        # Expand to 3D for multimodal rope (3 dimensions for Qwen2.5-Omni)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    elif position_ids.dim() == 2:
        # If 2D, expand to 3D
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    # Create position embeddings for RoPE
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

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

        if use_cache:
            # Cache is at index 2 if output_attentions, else 1
            cache_idx = 2 if output_attentions else 1
            if len(layer_outputs) > cache_idx:
                past_key_values = layer_outputs[cache_idx]

        if output_attentions:
            all_attentions += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # Add last hidden state
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


# Decoder layer forward
def decoder_layer_forward(
    self: Qwen2_5OmniDecoderLayer,
    hidden_states: torch.Tensor,
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

    # Self Attention
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


# Attention forward
def attn_forward(
    self: Qwen2_5OmniAttention,
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
        attention_mask = kwargs.pop("padding_mask")

    bsz = hidden_states.shape[0]
    q_len = torch.max(position_ids).item() + 1 if position_ids is not None else hidden_states.shape[0]
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

    # AlltoAll for Ulysses
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    if ulysses_sp_size > 1:
        assert (
            position_ids is not None
        ), "position_ids is required for Ulysses sequence parallelism"

        # Repeat kv heads to be divided by sequence parallel
        repeats = max(ulysses_sp_size // key_states.size(1), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (seq_len/n, n_head, head_dim) -> (seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=0, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=0, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=0, head_dim=1)

        # Cat the cu_seq_lens to the max seq len if padding is used
        if cu_seq_lens is not None and cu_seq_lens.max().item() < query_states.shape[0]:
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

    query_states = query_states.unsqueeze(0).transpose(1, 2)
    key_states = key_states.unsqueeze(0).transpose(1, 2)

    # Get mrope_section from config if available
    mrope_section = None
    if hasattr(self, 'rope_scaling') and isinstance(self.rope_scaling, dict):
        mrope_section = self.rope_scaling.get("mrope_section", None)

    if mrope_section is not None:
        # Use liger kernel version with mrope_section
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, mrope_section
        )
    else:
        # Fallback to standard rotary embedding
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_rotary_pos_emb
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

    query_states = query_states.transpose(1, 2).squeeze(0)
    key_states = key_states.transpose(1, 2).squeeze(0)

    if cu_seq_lens is not None:
        # Use varlen flash attention when cu_seq_lens is provided
        max_seqlen = torch.diff(cu_seq_lens).max().item()
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
    else:
        # Use regular flash attention when cu_seq_lens is None
        # Reshape for regular flash attention (batch, seq_len, num_heads, head_dim)
        batch_size = 1  # Assuming single batch for now
        seq_len = query_states.shape[0]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        attn_output = flash_attn_func(
            q=query_states,
            k=key_states,
            v=value_states,
            causal=True,
            softmax_scale=self.head_dim**-0.5,
            dropout_p=0.0,
        )

        # Reshape back to (seq_len, hidden_size)
        attn_output = attn_output.view(seq_len, -1)

    # AlltoAll for Ulysses
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1)

    attn_output = attn_output.reshape(-1, self.config.hidden_size).contiguous()

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Audio Encoder forward
def audio_encoder_forward(
    self: Qwen2_5OmniAudioEncoder,
    audio_values: torch.FloatTensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
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
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # Unpad audio if attention mask provided
    if attention_mask is not None:
        audio_values, indices, cu_seq_lens, _ = _unpad_input(
            audio_values, attention_mask=attention_mask
        )
    else:
        indices = None
        cu_seq_lens = None

    # Conv layers
    hidden_states = self.conv1(audio_values)
    hidden_states = torch.nn.functional.gelu(hidden_states)
    hidden_states = self.conv2(hidden_states)
    hidden_states = torch.nn.functional.gelu(hidden_states)

    hidden_states = hidden_states.permute(0, 2, 1)
    hidden_states = self.embed_positions.weight + hidden_states
    hidden_states = self.dropout(hidden_states)

    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    for encoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = encoder_layer(
            hidden_states,
            attention_mask=None,  # Flash attention doesn't need mask for audio
            output_attentions=output_attentions,
            cu_seq_lens=cu_seq_lens,
            indices=indices,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions += (layer_outputs[1],)

    hidden_states = self.layer_norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # Project to output dimension
    hidden_states = self.proj_out(hidden_states)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndRmpad(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        seq_lens=cu_seq_lens,
        word_idx=indices,
    )


# Audio Encoder Layer forward
def audio_encoder_layer_forward(
    self: Qwen2_5OmniAudioEncoderLayer,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor = None,
    output_attentions: bool = False,
    cu_seq_lens: Optional[torch.IntTensor] = None,
    indices: Optional[torch.IntTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    residual = hidden_states

    hidden_states = self.self_attn_layer_norm(hidden_states)

    # Self-attention with flash attention if available
    if cu_seq_lens is not None:
        # Use flash attention varlen
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cu_seq_lens=cu_seq_lens,
            indices=indices,
        )
    else:
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

    hidden_states = self.dropout(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.fc2(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


# Vision Encoder forward
def vision_encoder_forward(
    self: Qwen2_5OmniVisionEncoder,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
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
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # Process images
    if pixel_values is not None:
        batch_size = pixel_values.shape[0]
        image_embeds = self.patch_embed(pixel_values)
        image_embeds = image_embeds.flatten(2).transpose(1, 2)

        # Unpad if needed
        if image_grid_thw is not None:
            # Calculate attention mask from grid
            attention_mask = self._create_attention_mask_from_grid(
                image_grid_thw, image_embeds.shape[1]
            )
            image_embeds, indices, cu_seq_lens, _ = _unpad_input(
                image_embeds, attention_mask=attention_mask
            )
        else:
            indices = None
            cu_seq_lens = None
    else:
        image_embeds = None
        indices = None
        cu_seq_lens = None

    # Process videos
    if pixel_values_videos is not None:
        batch_size = pixel_values_videos.shape[0]
        video_embeds = self.patch_embed(pixel_values_videos)
        video_embeds = video_embeds.flatten(2).transpose(1, 2)

        # Unpad if needed
        if video_grid_thw is not None:
            # Calculate attention mask from grid
            attention_mask = self._create_attention_mask_from_grid(
                video_grid_thw, video_embeds.shape[1]
            )
            video_embeds, v_indices, v_cu_seq_lens, _ = _unpad_input(
                video_embeds, attention_mask=attention_mask
            )

            # Merge with image if both present
            if image_embeds is not None:
                hidden_states = torch.cat([image_embeds, video_embeds], dim=0)
                # Merge cu_seq_lens
                cu_seq_lens = torch.cat([cu_seq_lens, v_cu_seq_lens + cu_seq_lens[-1]], dim=0)
                indices = torch.cat([indices, v_indices + indices.max() + 1], dim=0)
            else:
                hidden_states = video_embeds
                cu_seq_lens = v_cu_seq_lens
                indices = v_indices
        else:
            if image_embeds is not None:
                hidden_states = torch.cat([image_embeds, video_embeds], dim=0)
            else:
                hidden_states = video_embeds
    else:
        hidden_states = image_embeds

    # Add positional embeddings
    hidden_states = hidden_states + self.pos_embed

    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # Pass through vision transformer blocks
    for i, block in enumerate(self.blocks):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Check if this is a full attention block or window attention
        use_window = i not in self.config.fullatt_block_indexes if hasattr(self.config, 'fullatt_block_indexes') else False

        layer_outputs = block(
            hidden_states,
            output_attentions=output_attentions,
            cu_seq_lens=cu_seq_lens if use_window else None,
            indices=indices if use_window else None,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # Project to output hidden size
    hidden_states = self.merger(hidden_states)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndRmpad(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        seq_lens=cu_seq_lens,
        word_idx=indices,
    )