"""Fused-linear cross-entropy forward for Qwen3_5MoeForCausalLM."""

from typing import List, Optional, Union

import torch
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except ImportError:
    print("Liger Kernel is not installed, pip install liger-kernel to use this patch")


def lce_forward(
    self: Qwen3_5MoeForCausalLM,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    use_rmpad: bool = False,
    **kwargs,
) -> MoeCausalLMOutputWithPast:
    # Top-level config may be Qwen3_5MoeConfig (multimodal wrapper) or
    # Qwen3_5MoeTextConfig (text-only ForCausalLM). Pull text-side fields
    # from text_config when present.
    text_cfg = getattr(self.config, "text_config", self.config)
    output_attentions = (
        output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
    )
    output_router_logits = (
        output_router_logits if output_router_logits is not None else getattr(text_cfg, "output_router_logits", False)
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else getattr(self.config, "output_hidden_states", False)
    )

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        **kwargs,
    )
    seq_lens = outputs.get("seq_lens", None)
    word_idx = outputs.get("word_idx", None)

    hidden_states = outputs.last_hidden_state

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    if labels is not None and word_idx is not None:
        labels = labels.view(-1)[word_idx.long()]

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        if use_rmpad:
            shift_hidden_states = []
            shift_labels_list = []
            for i in range(len(seq_lens) - 1):
                cur_hidden_states = hidden_states[seq_lens[i] : seq_lens[i + 1], :]
                cur_shift_hidden_states = cur_hidden_states[:-1, :].contiguous()
                cur_labels = labels[seq_lens[i] : seq_lens[i + 1]]
                cur_shift_labels = cur_labels[1:].contiguous()
                shift_hidden_states.append(cur_shift_hidden_states)
                shift_labels_list.append(cur_shift_labels)
            shift_hidden_states = torch.cat(shift_hidden_states, dim=0)
            shift_labels = torch.cat(shift_labels_list, dim=0)
        else:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        shift_hidden_states = shift_hidden_states.view(-1, text_cfg.hidden_size)
        shift_labels = shift_labels.view(-1)

        reduction = "sum" if "num_items_in_batch" in kwargs else "mean"
        lce = LigerFusedLinearCrossEntropyLoss(reduction=reduction)

        loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
        if reduction == "sum":
            loss /= kwargs["num_items_in_batch"]
    else:
        logits = self.lm_head(hidden_states)
        if labels is not None:
            loss = self.loss_function(logits, labels, text_cfg.vocab_size, **kwargs)

    aux_loss = None
    router_logits = getattr(outputs, "router_logits", None)
    if output_router_logits and router_logits is not None:
        aux_loss_mask = None if use_rmpad else attention_mask
        aux_loss = load_balancing_loss_func(
            router_logits,
            text_cfg.num_experts,
            text_cfg.num_experts_per_tok,
            aux_loss_mask,
        )
        if labels is not None:
            loss += text_cfg.router_aux_loss_coef * aux_loss.to(loss.device)

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=router_logits,
    )
