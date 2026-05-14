"""Forward overrides for LlavaOnevision2 model instances.

Because the OV2 modeling code is loaded via ``auto_map`` (trust_remote_code),
we cannot patch class objects in a shared module. These functions are bound
onto the OV2 model **instances** at load time by ``monkey_patch.py`` using
``types.MethodType``.

Provides:
- ``model_forward``: replacement for ``LlavaOnevision2Model.forward`` that
  performs sequence unpadding (rmpad) before invoking the inner ``Qwen3Model``
  (whose forward is already patched class-level by the qwen3 monkey patch).
- ``causal_lm_forward``: replacement for
  ``LlavaOnevision2ForConditionalGeneration.forward`` that adds Liger fused
  linear cross-entropy support, mirroring ``qwen3_vl_lce_forward``.
"""

from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache

from ..common_ops.loss import compute_lm_loss
from ..sequence_packing_utils import _unpad_input

# Filled in by monkey_patch.apply_liger_kernel_to_llava_onevision2 when it
# binds these forwards to a model instance. We cannot look up the OV2 output
# dataclasses from ``type(self).__module__`` at call time because FSDP wraps
# the module and replaces its class with an internal one.
_OV2_MODULES = {}


def _register_ov2_module(model):
    """Cache the OV2 modeling module so forwards can locate ModelOutput classes
    even after FSDP wraps the model."""
    import sys

    inner = getattr(model, "model", model)
    cls = type(inner)
    mod = sys.modules.get(cls.__module__)
    if mod is None:
        return
    _OV2_MODULES["modeling"] = mod
    _OV2_MODULES["ModelOutputWithPast"] = getattr(mod, "LlavaOnevision2ModelOutputWithPast", None)
    _OV2_MODULES["CausalLMOutputWithPast"] = getattr(mod, "LlavaOnevision2CausalLMOutputWithPast", None)


def model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
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
    patch_positions: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Drop-in replacement for ``LlavaOnevision2Model.forward`` with rmpad.

    Steps:
      1. Build padded ``inputs_embeds`` and inject image / video features just
         like the original OV2 forward (multi-image path; video aliased into
         the image path is handled by the data processor + image entry below).
      2. Unpad to ``(total_tokens,)`` and pass ``cu_seq_lens`` / ``indices``
         down to the inner Qwen3 language model (its forward is already
         patched to consume those kwargs).
    """
    return_dict = True if return_dict is None else return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Vision injection (still on padded tensors) ---------------------------
    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw, patch_positions=patch_positions)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, patch_positions=patch_positions)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # --- Unpad ----------------------------------------------------------------
    # NB: the patched ``Qwen3Model.forward`` (qwen3_ops.model_forward) ALSO
    # unpads internally if ``cu_seq_lens`` is None. We just let it handle that:
    # forward ``inputs_embeds`` + ``attention_mask`` straight through. The
    # qwen3 model_forward will do _unpad_input itself and return
    # ``BaseModelOutputWithPastAndRmpad`` carrying seq_lens / word_idx, which
    # we propagate upward so the OV2 LCE forward can slice labels.

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    # Reuse OV2 output dataclass to stay drop-in compatible, but stash rmpad
    # info on it. The OV2 ModelOutput dataclass accepts arbitrary kwargs
    # because it's a ``ModelOutput`` (just attribute assignment after init).
    ModelOutputCls = _OV2_MODULES.get("ModelOutputWithPast")

    out = ModelOutputCls(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
    )
    # Stash rmpad metadata (set as attributes; ModelOutput supports __setattr__)
    out["seq_lens"] = getattr(outputs, "seq_lens", None)
    out["word_idx"] = getattr(outputs, "word_idx", None)
    return out if return_dict else out.to_tuple()


def _compute_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    word_idx: Optional[torch.Tensor],
    lm_head_weight: torch.Tensor,
    loss_fn: str,
    use_rmpad: bool,
    text_config,
    kwargs: dict,
) -> torch.Tensor:
    """Thin wrapper over :func:`compute_lm_loss` that pulls ``hidden_size``
    from OV2's ``text_config``."""
    return compute_lm_loss(
        hidden_states=hidden_states,
        labels=labels,
        lm_head_weight=lm_head_weight,
        hidden_size=text_config.hidden_size,
        loss_fn=loss_fn,
        use_rmpad=use_rmpad,
        seq_lens=seq_lens,
        word_idx=word_idx,
        kwargs=kwargs,
    )


def causal_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    patch_positions: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    # ---- patch-time options (fixed via ``functools.partial`` in monkey patch) ----
    loss_fn: str = "lce",
    use_rmpad: bool = False,
    **kwargs,
):
    """Drop-in replacement for ``LlavaOnevision2ForConditionalGeneration.forward``.

    Behaviour depends on the patch-time options ``loss_fn`` and ``use_rmpad``:

      * ``loss_fn="lce"``: fused linear cross-entropy via Liger (no logits
        materialized). Falls back to ``loss_fn="ce"`` if liger is unavailable.
      * ``loss_fn="ce"``: standard cross-entropy on materialized logits.
      * ``use_rmpad=True``: assumes the inner LM ran with rmpad and the output
        ``hidden_states`` is a packed ``[total_tokens, H]`` tensor; shifts
        per-seq using ``seq_lens`` from the LM output.
      * Inference (``labels is None``) always materializes logits.
    """
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        patch_positions=patch_positions,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]

    seq_lens = outputs.get("seq_lens", None) if hasattr(outputs, "get") else None
    word_idx = outputs.get("word_idx", None) if hasattr(outputs, "get") else None

    loss = None
    logits = None
    text_config = getattr(self.config, "text_config", self.config)

    if labels is not None:
        loss = _compute_loss(
            hidden_states=hidden_states,
            labels=labels,
            seq_lens=seq_lens,
            word_idx=word_idx,
            lm_head_weight=self.lm_head.weight,
            loss_fn=loss_fn,
            use_rmpad=use_rmpad,
            text_config=text_config,
            kwargs=kwargs,
        )
    else:
        # Pure inference path.
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    OutputCls = _OV2_MODULES.get("CausalLMOutputWithPast")
    return OutputCls(
        loss=loss,
        logits=logits,
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
    )
