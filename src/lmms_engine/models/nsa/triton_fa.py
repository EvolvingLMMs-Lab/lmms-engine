import math

import torch
import torch.nn.functional as F
from einops import repeat
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from transformers.utils import is_flash_attn_2_available

from .flash_dmattn_triton import triton_dmattn_func
from .naive import compression

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
else:
    flash_attn_varlen_func = None


def create_block_causal_mask(query, key, block_size):
    """
    Create a block-based causal mask for attention computation with different q and k lengths.

    The mask allows queries to attend to key blocks in a causal manner. Each query token
    can attend to all key tokens up to its corresponding block position.

    Args:
        query: Tensor with shape (bs, seq_len_q, num_heads, num_dim)
        key: Tensor with shape (bs, seq_len_k, num_kv_heads, num_dim)

    Returns:
        block_causal_mask: Tensor with shape (bs, num_heads, seq_len_q, seq_len_k)
                          where True indicates positions that can be attended to

    Example:
        For seq_len_q=4, seq_len_k=2, block_size=2:
        [[1, 0],
         [1, 0],
         [1, 1],
         [1, 1]]
    """
    bs, seq_len_q, num_heads, _ = query.shape
    _, seq_len_k, num_kv_heads, _ = key.shape

    # Create query and key indices tensors
    q_indices = torch.arange(
        seq_len_q, device=query.device
    )  # [0, 1, 2, ..., seq_len_q-1]
    k_indices = torch.arange(
        seq_len_k, device=query.device
    )  # [0, 1, 2, ..., seq_len_k-1]

    # Determine which block each query token belongs to
    q_blocks = q_indices // block_size  # [0, 0, 1, 1, ...] for block_size=2

    # Create mask: each query can attend to keys up to and including its block
    # q_blocks[:, None] creates column vector, k_indices[None, :] creates row vector
    # Broadcasting creates (seq_len_q, seq_len_k) mask
    mask = k_indices[None, :] <= q_blocks[:, None]

    # Also ensure we don't exceed seq_len_k
    mask = mask & (k_indices[None, :] < seq_len_k)

    # Expand to batch and head dimensions
    # Shape: (bs, num_heads, seq_len_q, seq_len_k)
    block_causal_mask = (
        mask.unsqueeze(0).unsqueeze(0).expand(bs, num_heads, seq_len_q, seq_len_k)
    )

    return block_causal_mask


def create_block_causal_mask_varlen(
    query, key, cu_seq_lens_q, cu_seq_lens_k, block_size
):
    """
    Create a block-based causal mask for variable length sequences with different q and k lengths.

    The mask allows queries to attend to key blocks in a causal manner within each sequence.
    Each query token can attend to all key tokens up to its corresponding block position
    within the same sequence.

    Args:
        query: Tensor with shape (1, total_seq_len_q, num_heads, num_dim)
        key: Tensor with shape (1, total_seq_len_k, num_kv_heads, num_dim)
        cu_seq_lens_q: Cumulative sequence lengths for queries, where the last element is total_seq_len_q
        cu_seq_lens_k: Cumulative sequence lengths for keys, where the last element is total_seq_len_k

    Returns:
        block_causal_mask: Tensor with shape (1, num_heads, total_seq_len_q, total_seq_len_k)
                          Block-diagonal causal mask where each sequence has its own block causal structure

    Example:
        For 2 sequences:
        - Seq 1: q_len=4, k_len=2 -> block_size=2
        - Seq 2: q_len=2, k_len=4 -> block_size=0.5 (each q attends to 2 k tokens)
    """
    bs, total_seq_len_q, num_heads, _ = query.shape
    _, total_seq_len_k, num_kv_heads, _ = key.shape

    assert bs == 1, "Batch size must be 1 for variable length sequences"
    assert (
        total_seq_len_q == cu_seq_lens_q[-1]
    ), "Total query sequence length must match last element of cu_seq_lens_q"
    assert (
        total_seq_len_k == cu_seq_lens_k[-1]
    ), "Total key sequence length must match last element of cu_seq_lens_k"
    assert len(cu_seq_lens_q) == len(
        cu_seq_lens_k
    ), "Query and key must have same number of sequences"

    # Initialize mask as all False (all positions masked out)
    mask = torch.zeros(
        total_seq_len_q, total_seq_len_k, device=query.device, dtype=torch.bool
    )

    # Create global position indices
    q_indices = torch.arange(total_seq_len_q, device=query.device)
    k_indices = torch.arange(total_seq_len_k, device=query.device)

    # Process each sequence using vectorized operations
    q_start_idx = 0
    k_start_idx = 0

    for i in range(len(cu_seq_lens_q)):
        q_end_idx = cu_seq_lens_q[i].item()
        k_end_idx = cu_seq_lens_k[i].item()

        seq_len_q = q_end_idx - q_start_idx
        seq_len_k = k_end_idx - k_start_idx

        if seq_len_q > 0 and seq_len_k > 0:
            # Create local indices for this sequence
            local_q_indices = torch.arange(seq_len_q, device=query.device)
            local_k_indices = torch.arange(seq_len_k, device=query.device)

            if seq_len_q >= seq_len_k:
                # Each query token belongs to a block
                q_blocks = local_q_indices // block_size
                # Create mask: each query can attend to keys up to and including its block
                local_mask = local_k_indices[None, :] <= q_blocks[:, None]
                # Ensure we don't exceed seq_len_k
                local_mask = local_mask & (local_k_indices[None, :] < seq_len_k)
            else:
                # Each query can attend to keys in blocks 0 through q_idx
                max_k_indices = torch.clamp(
                    (local_q_indices + 1) * block_size, max=seq_len_k
                )
                local_mask = local_k_indices[None, :] < max_k_indices[:, None]

            # Apply the local mask to the global mask
            mask[q_start_idx:q_end_idx, k_start_idx:k_end_idx] = local_mask

        q_start_idx = q_end_idx
        k_start_idx = k_end_idx

    # Expand to batch and head dimensions
    # Shape: (1, num_heads, total_seq_len_q, total_seq_len_k)
    block_causal_mask = (
        mask.unsqueeze(0)
        .unsqueeze(0)
        .expand(1, num_heads, total_seq_len_q, total_seq_len_k)
    )

    return block_causal_mask


def triton_fa_nsa(
    query,
    key,
    value,
    block_counts,
    block_size,
    window_size,
    g_cmp,
    g_slc,
    g_swa,
    cu_seqlens=None,
    is_causal=True,
):
    scale = key.shape[-1] ** -0.5
    batch_size, seq_len_q, num_heads, _ = query.shape
    _, seq_len_k, num_kv_heads, _ = key.shape
    group_size = num_heads // num_kv_heads
    assert seq_len_q == seq_len_k, "Query and key must have the same length"

    if cu_seqlens is not None:
        assert batch_size == 1, "Batch size must be 1 when cu_seqlens are provided"

    if cu_seqlens is None:
        cu_seqlens = torch.cat([torch.tensor([0]), torch.tensor([seq_len_q])])
        cu_seqlens = cu_seqlens.to(torch.int32)

    k_cmp_list = []
    v_cmp_list = []
    cu_seq_lens_k = [0]
    # Compression
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        k_cmp, v_cmp = compression(key[:, start:end], value[:, start:end], block_size)
        k_cmp_list.append(k_cmp)
        v_cmp_list.append(v_cmp)
        cu_seq_lens_k.append(k_cmp.shape[1] + cu_seq_lens_k[-1])

    k_cmp = torch.cat(k_cmp_list, dim=1)
    v_cmp = torch.cat(v_cmp_list, dim=1)
    cu_seq_lens_k = torch.tensor(cu_seq_lens_k)
    cu_seq_lens_k = cu_seq_lens_k.to(torch.int32)

    # Calculate o_cmp
    cmp_mask = create_block_causal_mask_varlen(
        query, k_cmp, cu_seqlens, cu_seq_lens_k, block_size
    )
    k_cmp = repeat_kv(k_cmp.transpose(1, 2), group_size).transpose(1, 2)
    v_cmp = repeat_kv(v_cmp.transpose(1, 2), group_size).transpose(1, 2)
    o_cmp = triton_dmattn_func(
        query, k_cmp, v_cmp, cmp_mask, is_causal=is_causal, scale=scale
    )

    # Calculate attention score for selection
    # We have to materialize the softmax score again
    block_counts = min(block_counts, math.ceil(seq_len_q / block_size))
    block_indices = torch.zeros(
        batch_size,
        num_kv_heads,
        seq_len_q,
        seq_len_k,
        device=query.device,
        dtype=torch.bool,
    )
    for i in range(len(cu_seqlens) - 1):
        T_b = cu_seqlens[i + 1] - cu_seqlens[i]
        C_b = math.ceil(T_b / block_size)
        S_b = min(block_counts, C_b)
        k_cmp_curr = k_cmp[:, cu_seq_lens_k[i] : cu_seq_lens_k[i + 1]]
        q_b = query[:, cu_seqlens[i] : cu_seqlens[i + 1]]

        casual_mask = (
            (torch.arange(T_b) - block_size + 1)[:, None] // block_size
            < torch.arange(C_b)[None, :]
        ).to(q_b.device)
        local_mask = (
            torch.arange(T_b)[:, None] // block_size == torch.arange(C_b)[None, :]
        ).to(q_b.device)

        attn_cmp = torch.einsum("bqhd,bkhd->bhqk", q_b * scale, k_cmp_curr)
        attn_cmp = attn_cmp.masked_fill(casual_mask, float("-inf"))
        attn_cmp = attn_cmp.softmax(-1)
        attn_select = attn_cmp.masked_fill(local_mask, float(1.0))
        attn_select = attn_select.view(
            batch_size, num_kv_heads, group_size, T_b, C_b
        ).sum(2)
        attn_select = attn_select.nan_to_num()
        block_indices_b = attn_select.topk(S_b, -1)[1]
        mask = torch.zeros_like(attn_select, dtype=torch.bool)
        mask = mask.scatter_(-1, block_indices_b, True)
        mask = mask.repeat_interleave(block_size, -1)[:, :, :, :T_b]
        mask = torch.tril(mask, diagonal=0)
        block_indices[
            :, :, cu_seqlens[i] : cu_seqlens[i + 1], cu_seqlens[i] : cu_seqlens[i + 1]
        ] = mask

    block_indices = block_indices.repeat_interleave(group_size, 1)
    key = repeat_kv(key.transpose(1, 2), group_size).transpose(1, 2)
    value = repeat_kv(value.transpose(1, 2), group_size).transpose(1, 2)
    o_slc = triton_dmattn_func(
        query, key, value, block_indices, is_causal=is_causal, scale=scale
    )

    if batch_size == 1:
        max_seqlen = cu_seqlens.diff().max().item()
        o_swa = flash_attn_varlen_func(
            query.squeeze(0),
            key.squeeze(0),
            value.squeeze(0),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            window_size=(window_size - 1, 0),
        ).unsqueeze(0)
    else:
        o_swa = flash_attn_func(
            query, key, value, causal=True, window_size=(window_size - 1, 0)
        )
    torch.distributed.breakpoint()
    o = (
        o_slc * g_slc.unsqueeze(-1)
        + o_swa * g_swa.unsqueeze(-1)
        + o_cmp * g_cmp.unsqueeze(-1)
    )
    return o, block_indices
