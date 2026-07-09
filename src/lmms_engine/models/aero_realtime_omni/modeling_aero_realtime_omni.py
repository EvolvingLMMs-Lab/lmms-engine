# coding=utf-8
# Copyright 2025 LMMs-Lab team. All rights reserved.
"""AeroRealtime Omni wrapper: thinker (existing model) + talker (TTS).

Milestone 1: training forward only. The wrapper runs the (frozen) thinker with
``output_last_hidden_state=True``, extracts the audio-position hiddens on the packed
rmpad layout, and trains the talker with parallel teacher forcing. Assumes
``sp_ulysses_degree == 1`` (no sequence-parallel pad span).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.initialization import normal_ as _hf_normal_
from transformers.initialization import zeros_ as _hf_zeros_
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from lmms_engine.models.aero_realtime.modeling_aero_realtime import (
    AeroRealtimeForConditionalGeneration,
)

from .configuration_aero_realtime_omni import AeroRealtimeOmniConfig
from .modeling_aero_realtime_talker import AeroRealtimeTalkerForConditionalGeneration


@dataclass
class AeroRealtimeOmniCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    codec_loss: Optional[torch.FloatTensor] = None


class AeroRealtimeOmniForConditionalGeneration(PreTrainedModel):
    config_class = AeroRealtimeOmniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            _hf_normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                _hf_zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            _hf_normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data[module.padding_idx].zero_()

    def __init__(self, config: AeroRealtimeOmniConfig):
        super().__init__(config)
        self.thinker = AeroRealtimeForConditionalGeneration(config.thinker_config)
        self.talker = AeroRealtimeTalkerForConditionalGeneration(config.talker_config)
        self._default_speaker_id = next(iter(config.talker_config.speaker_id.values()))
        self.post_init()

    def forward(
        self,
        input_ids=None,
        text_stream_ids=None,
        attention_mask=None,
        codec_labels=None,
        codec_input_ids=None,
        labels=None,
        **thinker_inputs,
    ) -> AeroRealtimeOmniCausalLMOutputWithPast:
        talker_cfg = self.config.talker_config
        thinker_out = self.thinker(
            input_ids=input_ids,
            text_stream_ids=text_stream_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_last_hidden_state=True,
            **thinker_inputs,
        )

        text_loss = thinker_out.loss
        cu_seq_lens = thinker_out.cu_seq_lens
        indices = thinker_out.indices

        if indices is not None:
            packed_hidden = thinker_out.last_hidden_state.detach()
            packed_input_ids = input_ids.reshape(-1)[indices]
            codec_labels_flat = codec_labels.reshape(-1, codec_labels.shape[-1])[indices]
            codec_input_ids_flat = codec_input_ids.reshape(-1, codec_input_ids.shape[-1])[indices]
        else:
            B, L = input_ids.shape
            if attention_mask is None:
                attention_mask = input_ids.new_ones((B, L))
            valid = attention_mask.bool()
            packed_hidden = thinker_out.last_hidden_state[valid].detach()
            packed_input_ids = input_ids[valid]
            codec_labels_flat = codec_labels[valid]
            codec_input_ids_flat = codec_input_ids[valid]
            lengths = valid.sum(dim=1)
            cu_seq_lens = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, dim=0)]).to(torch.long)

        codec_loss = self.compute_talker_loss(
            packed_last_hidden_state=packed_hidden,
            packed_input_ids=packed_input_ids,
            codec_labels_flat=codec_labels_flat,
            codec_input_ids_flat=codec_input_ids_flat,
            cu_seq_lens=cu_seq_lens,
        )

        text_loss_val = text_loss if text_loss is not None else codec_loss.new_zeros(())
        loss = text_loss_val + self.config.codec_loss_weight * codec_loss

        return AeroRealtimeOmniCausalLMOutputWithPast(
            loss=loss,
            text_loss=text_loss,
            codec_loss=codec_loss,
        )

    def compute_talker_loss(
        self,
        packed_last_hidden_state: torch.Tensor,
        packed_input_ids: torch.Tensor,
        codec_labels_flat: torch.Tensor,
        codec_input_ids_flat: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Padded-batch teacher-forced talker training loss (group-0 + residual).

        This is the fallback path used when the rmpad monkey patch is not applied.
        The rmpad monkey patch replaces this method with a packed + Ulysses version
        (see ``aero_realtime_omni_ops.compute_talker_loss``).

        ``codec_input_ids_flat`` carries the gold codec (used for teacher-forced
        input embeddings). ``codec_labels_flat`` may contain ``-100`` at silence
        frames (used only as CE targets).
        """
        talker = self.talker
        talker_cfg = self.config.talker_config
        audio_token_id = self.config.thinker_config.audio_token_index
        speaker_id = self._default_speaker_id
        codec_bos_id = talker_cfg.codec_bos_id
        codec_nothink_id = talker_cfg.codec_nothink_id

        device = packed_last_hidden_state.device
        d_talker = talker.config.hidden_size
        codec_emb = talker.get_input_embeddings()

        audio_mask = packed_input_ids == audio_token_id
        S = cu_seq_lens.numel() - 1

        cond_ids = torch.tensor(
            [codec_bos_id, codec_nothink_id, speaker_id],
            device=device,
            dtype=torch.long,
        )

        seq_embs = []
        body_lens = []
        audio_label_segs = []
        audio_input_segs = []
        for s in range(S):
            start = int(cu_seq_lens[s].item())
            end = int(cu_seq_lens[s + 1].item())
            seg_mask = audio_mask[start:end]
            n_i = int(seg_mask.sum().item())
            if n_i == 0:
                continue
            seg_hidden = packed_last_hidden_state[start:end][seg_mask]
            seg_labels = codec_labels_flat[start:end][seg_mask]
            seg_inputs = codec_input_ids_flat[start:end][seg_mask]

            cond_emb = codec_emb(cond_ids)
            text_h = talker.text_projection(seg_hidden)
            prev_ids = torch.empty(n_i, device=device, dtype=torch.long)
            prev_ids[0] = codec_bos_id
            if n_i > 1:
                prev_ids[1:] = seg_inputs[:-1, 0]
            prev_emb = codec_emb(prev_ids)
            body_emb = text_h + prev_emb
            seq_emb = torch.cat([cond_emb, body_emb], dim=0)

            seq_embs.append(seq_emb)
            body_lens.append(n_i)
            audio_label_segs.append(seg_labels)
            audio_input_segs.append(seg_inputs)

        if len(seq_embs) == 0:
            return packed_last_hidden_state.sum() * 0.0

        max_len = max(e.shape[0] for e in seq_embs)
        Sp = len(seq_embs)
        batch = packed_last_hidden_state.new_zeros(Sp, max_len, d_talker)
        attn_mask = torch.zeros(Sp, max_len, dtype=torch.long, device=device)
        for i, e in enumerate(seq_embs):
            batch[i, : e.shape[0]] = e
            attn_mask[i, : e.shape[0]] = 1

        trunk_out = talker.model(inputs_embeds=batch, attention_mask=attn_mask)
        trunk_hidden_padded = trunk_out.last_hidden_state

        trunk_hidden = torch.cat([trunk_hidden_padded[i, 3 : 3 + body_lens[i]] for i in range(Sp)], dim=0)
        audio_label_codes = torch.cat(audio_label_segs, dim=0)
        audio_input_codes = torch.cat(audio_input_segs, dim=0)

        group0_logits = talker.codec_head(trunk_hidden)
        group0_loss = F.cross_entropy(group0_logits, audio_label_codes[:, 0], ignore_index=-100)

        _, residual_loss = talker.forward_sub_talker_finetune(audio_input_codes, audio_label_codes, trunk_hidden)

        return group0_loss + residual_loss
