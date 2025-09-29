from typing import List, Tuple

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from lmms_engine.models.nsa.naive import naive_nsa_with_compression
from lmms_engine.models.nsa.triton_fa import triton_fa_nsa


def forward_train(
    self,
    packed_sequence: torch.Tensor,
    sample_lens: List[int],
    attention_mask,
    packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    packed_und_token_indexes: torch.LongTensor,
    packed_gen_token_indexes: torch.LongTensor,
):
    packed_query_states = packed_sequence.new_zeros(
        (packed_sequence.shape[0], self.num_heads * self.head_dim)
    )
    packed_key_states = packed_sequence.new_zeros(
        (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
    )
    packed_value_states = packed_sequence.new_zeros(
        (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
    )

    packed_sequence_und = packed_sequence[packed_und_token_indexes]
    packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

    packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
    packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(
        packed_sequence_gen
    )

    packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
    packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(
        packed_sequence_gen
    )

    packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
    packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(
        packed_sequence_gen
    )

    g = self.g_proj(packed_sequence)
    g = g.view(1, packed_sequence.shape[0], self.num_heads, 3)
    g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)

    packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
    packed_key_states = packed_key_states.view(
        -1, self.num_key_value_heads, self.head_dim
    )
    packed_value_states = packed_value_states.view(
        -1, self.num_key_value_heads, self.head_dim
    )
    if self.config.freeze_und:
        packed_value_states[packed_und_token_indexes] = packed_value_states[
            packed_und_token_indexes
        ].detach()

    packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
    packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

    packed_query_states_[packed_und_token_indexes] = self.q_norm(
        packed_query_states[packed_und_token_indexes]
    )
    if self.config.freeze_und:
        packed_query_states_[packed_und_token_indexes] = packed_query_states_[
            packed_und_token_indexes
        ].detach()
    packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(
        packed_query_states[packed_gen_token_indexes]
    )

    packed_key_states_[packed_und_token_indexes] = self.k_norm(
        packed_key_states[packed_und_token_indexes]
    )
    if self.config.freeze_und:
        packed_key_states_[packed_und_token_indexes] = packed_key_states_[
            packed_und_token_indexes
        ].detach()
    packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(
        packed_key_states[packed_gen_token_indexes]
    )

    packed_cos, packed_sin = packed_position_embeddings
    packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
        packed_query_states_,
        packed_key_states_,
        packed_cos,
        packed_sin,
        unsqueeze_dim=1,
    )
    cu_seqlens = torch.tensor(
        [0] + sample_lens, dtype=torch.int32, device=packed_query_states_.device
    )

    packed_attn_output, block_indices = triton_fa_nsa(
        packed_query_states_.unsqueeze(0),
        packed_key_states_.unsqueeze(0),
        packed_value_states.unsqueeze(0),
        g_slc=g_slc,
        g_swa=g_swa,
        g_cmp=g_cmp,
        block_counts=self.config.block_counts,
        block_size=self.config.block_size,
        window_size=self.config.window_size,
        cu_seqlens=cu_seqlens,
    )

    packed_attn_output = packed_attn_output.squeeze(0)

    packed_attn_output = packed_attn_output.transpose(0, 1).reshape(
        -1, self.num_heads * self.head_dim
    )
    packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
    packed_attn_output_[packed_und_token_indexes] = self.o_proj(
        packed_attn_output[packed_und_token_indexes]
    )
    packed_attn_output_[packed_gen_token_indexes] = self.o_proj_moe_gen(
        packed_attn_output[packed_gen_token_indexes]
    )

    return packed_attn_output_
