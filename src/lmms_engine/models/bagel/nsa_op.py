from typing import List, Tuple

import torch
from loguru import logger
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

try:
    from native_sparse_attention.ops import (
        compressed_attention,
        linear_compress,
        topk_sparse_attention,
    )
except ImportError:
    logger.warning(
        "native_sparse_attention is not installed, please install with"
        " `pip install git+https://github.com/XunhaoLai/native-sparse-attention-triton.git`"
    )

from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func


def forward_train(
    self,
    packed_sequence: torch.Tensor,
    sample_lens: List[int],
    attention_mask,
    packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    packed_und_token_indexes: torch.LongTensor,
    packed_gen_token_indexes: torch.LongTensor,
):
    packed_query_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_heads * self.head_dim))
    packed_key_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))
    packed_value_states = packed_sequence.new_zeros(
        (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
    )

    packed_sequence_und = packed_sequence[packed_und_token_indexes]
    packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

    packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
    packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(packed_sequence_gen)

    packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
    packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(packed_sequence_gen)

    packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
    packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(packed_sequence_gen)

    g = self.g_proj(packed_sequence)
    g = g.view(1, packed_sequence.shape[0], self.num_heads, 3)
    g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)

    packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
    packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
    packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
    if self.config.freeze_und:
        packed_value_states[packed_und_token_indexes] = packed_value_states[packed_und_token_indexes].detach()

    packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
    packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

    packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
    if self.config.freeze_und:
        packed_query_states_[packed_und_token_indexes] = packed_query_states_[packed_und_token_indexes].detach()
    packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_gen_token_indexes])

    packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
    if self.config.freeze_und:
        packed_key_states_[packed_und_token_indexes] = packed_key_states_[packed_und_token_indexes].detach()
    packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes])
    cu_seqlens = torch.tensor([0] + sample_lens, dtype=torch.int32, device=packed_query_states_.device)

    # 1. key value compression
    compressed_key, compressed_cu_seqlens = self.compress_func(
        packed_key_states_,
        self.compress_key,
        cu_seqlens,
        self.kernel_size,
        self.kernel_stride,
        self.intra_block_pe,
    )
    compressed_value, _ = self.compress_func(
        packed_value_states,
        self.compress_value,
        cu_seqlens,
        self.kernel_size,
        self.kernel_stride,
        None,
    )

    packed_cos, packed_sin = packed_position_embeddings
    packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
        packed_query_states_,
        packed_key_states_,
        packed_cos,
        packed_sin,
        unsqueeze_dim=1,
    )
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

    compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
    compressed_attn_output, topk_idx = compressed_attention(
        packed_query_states_,
        compressed_key,
        compressed_value,
        self.kernel_size,
        self.kernel_stride,
        self.block_size,
        self.topk,
        cu_seqlens,
        compressed_cu_seqlens,
        seqlens.max().item(),
        compressed_seqlens.max().item(),
        None,
        self.init_blocks,
        self.local_blocks,
    )

    # topk sparse attention
    sparse_attn_output = topk_sparse_attention(
        packed_query_states_,
        packed_key_states_,
        packed_value_states,
        topk_idx,
        self.block_size,
        cu_seqlens,
        None,
    )

    # sliding window attention
    sliding_attn_output = flash_attn_varlen_func(
        packed_query_states_,
        packed_key_states_,
        packed_value_states,
        cu_seqlens,
        cu_seqlens,
        seqlens.max().item(),
        seqlens.max().item(),
        causal=True,
        window_size=(self.window_size, -1),
    )

    attn_output = (
        compressed_attn_output * g_cmp.unsqueeze(-1)
        + sparse_attn_output * g_swa.unsqueeze(-1)
        + sliding_attn_output * g_slc.unsqueeze(-1)
    )

    packed_attn_output = attn_output.squeeze(0)

    packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
    packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
    packed_attn_output_[packed_und_token_indexes] = self.o_proj(packed_attn_output[packed_und_token_indexes])
    packed_attn_output_[packed_gen_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_gen_token_indexes])

    return packed_attn_output_
