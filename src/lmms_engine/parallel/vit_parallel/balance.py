"""LPT balancing for ViT frame/image parallelism.

Distributes a global set of frames (or images) across DP ranks so that the
total ViT compute load (e.g. patch count) per rank is as balanced as possible.
The minimum movable unit is one frame: a single grid_thw row cannot be split.

The planner is a pure function: given the same inputs on every rank it returns
the same assignment, so no communication is needed to agree on the plan once
loads have been gathered.

Algorithm: Longest Processing Time (LPT) greedy. Worst-case ratio 4/3, in
practice usually within ~1% of optimal on typical multimodal batches.
"""

from typing import List, Optional, Sequence, Tuple


def lpt_balance(
    loads: Sequence[int],
    num_ranks: Optional[int] = None,
    rank_idx: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Compute an LPT assignment of frames to ranks.

    Args:
        loads: per-frame load (e.g. ``T * H * W`` patches). Frames are
            identified by their position in this list.
        num_ranks: total number of ranks. Required when ``rank_idx`` is not
            provided; ignored otherwise (defaults to ``rank_idx + 1`` if both
            are None — which is the degenerate "one rank, takes everything"
            case).
        rank_idx: if provided, also return the list of frame indices assigned
            to this rank as a convenience. If ``None``, the caller can derive
            it from the returned ``assignment``.

    Returns:
        ``(assignment, load_per_rank)`` where
            * ``assignment[k]`` is the destination rank for frame ``k``
            * ``load_per_rank[r]`` is the total load assigned to rank ``r``

    Notes:
        * Deterministic: the same ``loads`` and ``num_ranks`` give the same
          assignment on every rank, so calling this on each rank produces a
          consistent global plan with no extra communication.
        * Ties in load are broken by frame index (stable sort), and ties in
          remaining rank load are broken by smallest rank index. Both
          tie-breakers are stable, which is what we rely on for determinism.
        * If ``num_ranks > len(loads)`` some ranks will be assigned zero
          frames. This is allowed; downstream code must handle empty
          ViT input shapes.
    """
    if num_ranks is None:
        if rank_idx is None:
            num_ranks = 1
        else:
            num_ranks = rank_idx + 1
    if num_ranks <= 0:
        raise ValueError(f"num_ranks must be positive, got {num_ranks}")

    n_frames = len(loads)
    assignment: List[int] = [-1] * n_frames
    load_per_rank: List[int] = [0] * num_ranks

    # Frames in descending load order; stable tie-break by original index.
    order = sorted(range(n_frames), key=lambda i: (-loads[i], i))

    for k in order:
        # argmin with stable tie-break (smallest rank id wins).
        target = 0
        best = load_per_rank[0]
        for r in range(1, num_ranks):
            if load_per_rank[r] < best:
                best = load_per_rank[r]
                target = r
        assignment[k] = target
        load_per_rank[target] += loads[k]

    return assignment, load_per_rank


def frames_for_rank(assignment: Sequence[int], rank_idx: int) -> List[int]:
    """Return the global frame indices assigned to ``rank_idx``.

    Convenience wrapper; ``assignment`` is what ``lpt_balance`` returns.
    """
    return [k for k, r in enumerate(assignment) if r == rank_idx]
