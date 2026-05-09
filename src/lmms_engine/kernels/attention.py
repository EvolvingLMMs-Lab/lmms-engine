"""Attention backends.

Varlen attention implementations with identical signatures, dispatched via
``varlen_attn(..., backend=...)``.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from lmms_engine.utils.import_utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None


def fa2_varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: Tuple[int, int] = (-1, -1),
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """FlashAttention-2 varlen forward.

    Inputs ``q``, ``k``, ``v`` have shape ``(total_tokens, num_heads, head_dim)``.
    Returns a tensor of the same shape as ``q``.
    """
    assert flash_attn_varlen_func is not None, "flash_attn is not installed; use backend='sdpa' or install flash-attn."
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        softmax_scale=softmax_scale,
        window_size=window_size,
        dropout_p=dropout_p,
    )


def sdpa_varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: Tuple[int, int] = (-1, -1),
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """SDPA-based varlen attention with the same signature as ``fa2_varlen_attn``.

    Builds a block-diagonal additive mask from ``cu_seqlens`` and dispatches to
    ``torch.nn.functional.scaled_dot_product_attention``. Intended as a portable
    fallback (e.g. ROCm without flash-attn). Memory cost is O((sum_q)*(sum_k));
    for very long packed sequences prefer the FA2 backend.

    Inputs ``q``, ``k``, ``v`` have shape ``(total_tokens, num_heads, head_dim)``.
    """
    total_q, num_heads_q, head_dim = q.shape
    total_k, num_heads_k, _ = k.shape

    # GQA: broadcast kv heads to query heads
    if num_heads_k != num_heads_q:
        assert (
            num_heads_q % num_heads_k == 0
        ), f"num_heads_q ({num_heads_q}) must be divisible by num_heads_k ({num_heads_k})"
        repeat = num_heads_q // num_heads_k
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    # (T, H, D) -> (1, H, T, D) for SDPA
    q_ = q.transpose(0, 1).unsqueeze(0)
    k_ = k.transpose(0, 1).unsqueeze(0)
    v_ = v.transpose(0, 1).unsqueeze(0)

    # Build additive block-diagonal mask of shape (total_q, total_k).
    # Position-based construction avoids Python loops over sequences.
    device = q.device
    q_seq_id = torch.zeros(total_q, dtype=torch.long, device=device)
    q_seq_id[cu_seqlens_q[1:-1].long()] = 1
    q_seq_id = q_seq_id.cumsum(0)  # which sequence each query token belongs to

    k_seq_id = torch.zeros(total_k, dtype=torch.long, device=device)
    k_seq_id[cu_seqlens_k[1:-1].long()] = 1
    k_seq_id = k_seq_id.cumsum(0)

    # Same-sequence mask: True where attention is allowed (before causal/window).
    allow = q_seq_id.unsqueeze(1) == k_seq_id.unsqueeze(0)  # (Tq, Tk)

    if causal or window_size != (-1, -1):
        # Per-sequence local positions
        q_starts = cu_seqlens_q[q_seq_id]
        k_starts = cu_seqlens_k[k_seq_id]
        q_pos = (torch.arange(total_q, device=device) - q_starts).unsqueeze(1)  # (Tq, 1)
        k_pos = (torch.arange(total_k, device=device) - k_starts).unsqueeze(0)  # (1, Tk)

        if causal:
            # FA2 convention: align q to the END of k when lengths differ.
            offset = max_seqlen_k - max_seqlen_q
            allow &= k_pos <= (q_pos + offset)

        left, right = window_size
        if left >= 0:
            allow &= k_pos >= (q_pos - left)
        if right >= 0:
            allow &= k_pos <= (q_pos + right)

    attn_mask = torch.zeros(total_q, total_k, dtype=q.dtype, device=device)
    attn_mask.masked_fill_(~allow, float("-inf"))
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, Tq, Tk)

    out = F.scaled_dot_product_attention(
        q_,
        k_,
        v_,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        scale=softmax_scale,
        is_causal=False,  # causal already encoded in mask
    )
    # (1, H, Tq, D) -> (Tq, H, D)
    return out.squeeze(0).transpose(0, 1).contiguous()


_BACKENDS = {
    "flash_attention_2": fa2_varlen_attn,
    "sdpa": sdpa_varlen_attn,
}


def varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: Tuple[int, int] = (-1, -1),
    dropout_p: float = 0.0,
    backend: str = "flash_attention_2",
) -> torch.Tensor:
    """Dispatch varlen attention to the requested backend.

    Args:
        backend: ``"flash_attention_2"`` or ``"sdpa"`` (matches HF
            ``config._attn_implementation`` values).
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown attention backend: {backend!r}. Available: {list(_BACKENDS)}")
    return _BACKENDS[backend](
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
        softmax_scale=softmax_scale,
        window_size=window_size,
        dropout_p=dropout_p,
    )


__all__ = ["fa2_varlen_attn", "sdpa_varlen_attn", "varlen_attn"]
