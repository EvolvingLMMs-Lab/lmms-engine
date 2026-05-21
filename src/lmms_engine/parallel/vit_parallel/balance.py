"""Balancing for ViT frame/image parallelism.

Distributes a global set of frames (or images) across DP ranks so that the
total ViT compute load (e.g. patch count) per rank is as balanced as possible.
The minimum movable unit is one frame: a single grid_thw row cannot be split.

The planner is a pure function: given the same inputs on every rank it returns
the same assignment, so no communication is needed to agree on the plan once
loads have been gathered.

When each frame's source rank is known, the planner is locality-aware: frames
start on their source rank and only overloaded ranks spill frames to
underloaded ranks. This avoids the communication-heavy behavior of global LPT,
where a rank that is already near average can still have all local frames
replaced by frames from other ranks.
"""

import math
from typing import List, Optional, Sequence, Tuple


def lpt_balance(
    loads: Sequence[int],
    num_ranks: Optional[int] = None,
    rank_idx: Optional[int] = None,
    frames_per_rank: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int]]:
    """Compute a deterministic assignment of frames to ranks.

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
        frames_per_rank: source frame counts for each rank in the flattened
            ``loads`` list. When provided, enables locality-aware balancing:
            frames stay on their source rank unless that rank is overloaded.

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

    if frames_per_rank is not None:
        return _locality_aware_balance(loads, frames_per_rank, num_ranks)

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


def _source_ranks(frames_per_rank: Sequence[int], num_frames: int, num_ranks: int) -> List[int]:
    if len(frames_per_rank) != num_ranks:
        raise ValueError(f"frames_per_rank must have {num_ranks} entries, got {len(frames_per_rank)}")
    if sum(frames_per_rank) != num_frames:
        raise ValueError(f"sum(frames_per_rank) must equal {num_frames}, got {sum(frames_per_rank)}")

    source_ranks: List[int] = []
    for rank, num_rank_frames in enumerate(frames_per_rank):
        if num_rank_frames < 0:
            raise ValueError(f"frames_per_rank entries must be non-negative, got {num_rank_frames}")
        source_ranks.extend([rank] * num_rank_frames)
    return source_ranks


def _pick_spill_frame(
    frame_indices: Sequence[int],
    loads: Sequence[int],
    donor_load: int,
    receiver_load: int,
    target_load: int,
) -> Optional[int]:
    receiver_deficit = target_load - receiver_load

    fitting = [idx for idx in frame_indices if loads[idx] <= receiver_deficit]
    if fitting:
        return max(fitting, key=lambda idx: (loads[idx], -idx))

    improving = [idx for idx in frame_indices if max(donor_load - loads[idx], receiver_load + loads[idx]) < donor_load]
    if improving:
        return min(improving, key=lambda idx: (loads[idx], idx))

    return None


def _locality_aware_balance(
    loads: Sequence[int],
    frames_per_rank: Sequence[int],
    num_ranks: int,
) -> Tuple[List[int], List[int]]:
    """Balance by spilling only overloaded ranks' local frames."""
    n_frames = len(loads)
    source_ranks = _source_ranks(frames_per_rank, n_frames, num_ranks)
    assignment = source_ranks[:]
    load_per_rank = [0] * num_ranks
    frames_by_rank: List[List[int]] = [[] for _ in range(num_ranks)]

    for idx, (load, rank) in enumerate(zip(loads, source_ranks)):
        load_per_rank[rank] += load
        frames_by_rank[rank].append(idx)

    total_load = sum(loads)
    if total_load == 0:
        return assignment, load_per_rank

    target_load = math.ceil(total_load / num_ranks)

    while True:
        donors = [rank for rank, load in enumerate(load_per_rank) if load > target_load]
        receivers = [rank for rank, load in enumerate(load_per_rank) if load < target_load]
        if not donors or not receivers:
            break

        donors.sort(key=lambda rank: (-(load_per_rank[rank] - target_load), rank))
        receivers.sort(key=lambda rank: (-(target_load - load_per_rank[rank]), rank))

        moved = False
        for donor in donors:
            donor_frames = [idx for idx in frames_by_rank[donor] if assignment[idx] == donor]
            donor_frames.sort(key=lambda idx: (-loads[idx], idx))
            if not donor_frames:
                continue

            for receiver in receivers:
                frame_idx = _pick_spill_frame(
                    donor_frames,
                    loads,
                    load_per_rank[donor],
                    load_per_rank[receiver],
                    target_load,
                )
                if frame_idx is None:
                    continue

                assignment[frame_idx] = receiver
                load_per_rank[donor] -= loads[frame_idx]
                load_per_rank[receiver] += loads[frame_idx]
                moved = True
                break

            if moved:
                break

        if not moved:
            break

    return assignment, load_per_rank


def frames_for_rank(assignment: Sequence[int], rank_idx: int) -> List[int]:
    """Return the global frame indices assigned to ``rank_idx``.

    Convenience wrapper; ``assignment`` is what ``lpt_balance`` returns.
    """
    return [k for k, r in enumerate(assignment) if r == rank_idx]
