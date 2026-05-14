"""Common LM loss helpers shared across model wrappers.

Used by custom CausalLM forwards (e.g. LlavaOnevision2) to compute the next-
token loss with the right combination of fused linear CE (liger), packed
``rmpad`` shifting, and Ulysses sequence parallelism.
"""

from typing import Optional

import torch
import torch.distributed as dist

from lmms_engine.parallel.sequence_parallel.ulysses import (
    calculate_seq_len_per_rank,
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    pad_to_max_across_ranks,
    slice_input_tensor,
)

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None


def compute_lm_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head_weight: torch.Tensor,
    hidden_size: int,
    *,
    loss_fn: str = "lce",
    use_rmpad: bool = False,
    seq_lens: Optional[torch.Tensor] = None,
    word_idx: Optional[torch.Tensor] = None,
    kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """Next-token LM loss with optional fused LCE / rmpad / Ulysses SP.

    Args:
        hidden_states: ``[B, L, H]`` (when ``use_rmpad=False``) or
            ``[total_tokens, H]`` (when ``use_rmpad=True``).
        labels: padded ``[B, L]`` token labels. When ``use_rmpad=True`` and
            ``word_idx`` is provided, labels are gathered via ``word_idx``.
        lm_head_weight: the LM head weight tensor used either by the fused
            LCE kernel or by a plain ``F.linear`` for ``loss_fn="ce"``.
        hidden_size: text-model hidden size, for reshaping shifted hidden states.
        loss_fn: ``"lce"`` (Liger fused linear CE) or ``"ce"`` (materialized
            logits + ``F.cross_entropy``).
        use_rmpad: when True, shift inside each packed seq using ``seq_lens``.
        seq_lens: cumulative seq lens of packed sequences (rmpad metadata).
        word_idx: indices into the flattened padded ``labels`` for unpadding.
        kwargs: forwarded model kwargs; we look for ``num_items_in_batch`` to
            decide reduction mode.

    Returns:
        Scalar loss tensor.
    """
    if kwargs is None:
        kwargs = {}
    sp_size = get_ulysses_sequence_parallel_world_size()

    # Align labels with hidden_states layout.
    if use_rmpad and word_idx is not None:
        labels_use = labels.view(-1)[word_idx.long()]
    else:
        labels_use = labels

    if sp_size > 1:
        if seq_lens is not None:
            seq_lens = calculate_seq_len_per_rank(seq_lens.tolist())
        labels_use = slice_input_tensor(labels_use, dim=0, padding=True)

    # Shift hidden states / labels for next-token prediction.
    if use_rmpad and seq_lens is not None:
        shift_h, shift_l = [], []
        for i in range(len(seq_lens) - 1):
            ch = hidden_states[seq_lens[i] : seq_lens[i + 1], :]
            cl = labels_use[seq_lens[i] : seq_lens[i + 1]]
            shift_h.append(ch[:-1, :].contiguous())
            shift_l.append(cl[1:].contiguous())
        shift_hidden = torch.cat(shift_h, dim=0)
        shift_labels = torch.cat(shift_l, dim=0)
    else:
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels_use[..., 1:].contiguous()

    shift_hidden = shift_hidden.view(-1, hidden_size)
    shift_labels = shift_labels.view(-1)

    reduction = "sum" if "num_items_in_batch" in kwargs else "mean"
    if sp_size > 1:
        reduction = "none"

    if loss_fn == "lce":
        if LigerFusedLinearCrossEntropyLoss is None:
            raise RuntimeError("loss_fn='lce' requires liger-kernel; install it or use loss_fn='ce'.")
        lce = LigerFusedLinearCrossEntropyLoss(reduction=reduction)
        loss = lce(lm_head_weight, shift_hidden, shift_labels)
    elif loss_fn == "ce":
        logits = torch.nn.functional.linear(shift_hidden, lm_head_weight)
        loss = torch.nn.functional.cross_entropy(logits.float(), shift_labels, reduction=reduction, ignore_index=-100)
    else:
        raise ValueError(f"Unknown loss_fn={loss_fn!r}; expected 'lce' or 'ce'.")

    # Ulysses SP gather.
    if sp_size > 1:
        loss, total_padding = pad_to_max_across_ranks(loss, dim=0)
        loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=total_padding)
        num_valid_tokens = (shift_labels != -100).sum().float()
        sp_group = get_ulysses_sequence_parallel_group()
        if sp_group is not None:
            dist.all_reduce(num_valid_tokens, op=dist.ReduceOp.SUM, group=sp_group)
        loss = torch.sum(loss) / (num_valid_tokens + 1e-8)

    if reduction == "sum":
        loss = loss / kwargs["num_items_in_batch"]

    return loss
