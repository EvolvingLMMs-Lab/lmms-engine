"""Frame-parallel dispatch for Qwen3.5 ``Qwen3_5VisionModel.forward``.

The upstream forward signature is::

    Qwen3_5VisionModel.forward(self, hidden_states, grid_thw, **kwargs)

where
    hidden_states : (total_patches, C*T*P*P)   — packed
    grid_thw      : (num_segments, 3)          — each row contributes T*H*W patches

``input_dispatch`` redistributes frames across the DP group via LPT so each
rank handles a balanced number of ViT patches, runs the original forward
locally on the received slice, and the matching ``output_dispatch`` (TODO)
gathers features back so each rank ends up with the features for its own
original frames.

Communication:
    * Metadata (per-rank token / frame counts) goes through ``all_gather_object``.
    * ``hidden_states`` uses ``all_to_all_single_autograd`` so gradients route
      back to the originating rank.
    * ``grid_thw`` uses plain ``all_to_all_single`` (no grad needed).
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)

from lmms_engine.parallel.vit_parallel.balance import lpt_balance


def _patches_per_row(grid_thw: torch.Tensor) -> torch.Tensor:
    """Patches contributed by each grid_thw row: T * H * W."""
    return grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]


def input_dispatch(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    **kwargs,
) -> Tuple[Tuple, Dict[str, Any], Dict[str, Any]]:
    """Dispatch frames across the DP group ahead of the ViT forward.

    Returns ``(new_args, new_kwargs, ctx)`` for ``wrap_vit_forward``:
        new_args / new_kwargs : passed to the original ``Qwen3_5VisionModel.forward``
        ctx                   : minimal state needed by ``output_dispatch`` to
                                run the reverse all_to_all (just the group and
                                the swapped splits).

    Steps:
        1. Gather each rank's per-frame token counts (``num_tokens``) and
           frame count via two ``all_gather_object`` calls. Flatten into a
           global ``loads`` list.
        2. Run LPT on ``loads`` to get a global ``assignment`` (which rank
           each global frame belongs to). Deterministic, same on every rank.
        3. Slice ``assignment`` into the segment owned by this rank to compute
           src-view ``input_splits`` / ``input_frames`` (how many tokens /
           frames I send to each dst).
        4. Walk ``assignment`` to compute src-view ``output_splits`` /
           ``output_frames`` (how many I receive from each src).
        5. ``all_to_all_single_autograd`` on ``hidden_states`` (carries grads
           via the reverse permutation) and plain ``all_to_all_single`` on
           ``grid_thw`` (metadata only).
    """
    world_size = dist.get_world_size(group=group)
    my_rank = dist.get_rank(group=group)
    device = hidden_states.device

    # ---- 1) gather per-rank token/frame counts ----
    num_tokens = grid_thw.prod(-1).tolist()
    num_frames = grid_thw.shape[0]
    total_tokens = [None for _ in range(world_size)]
    total_frames = [None for _ in range(world_size)]
    dist.all_gather_object(total_tokens, num_tokens, group=group)
    dist.all_gather_object(total_frames, num_frames, group=group)
    loads = [token for tokens in total_tokens for token in tokens]

    # ---- 2) LPT ----
    assignment_list, _ = lpt_balance(loads, num_ranks=world_size)

    # ---- 3) src-view input splits (what I send to each dst) ----
    # Slice out the segment of `assignment_list` corresponding to my local frames.
    my_start = sum(total_frames[:my_rank])
    my_end = my_start + num_frames
    my_assignment = assignment_list[my_start:my_end]

    input_splits = [0] * world_size  # tokens I send to each dst
    input_frames = [0] * world_size  # frames I send to each dst
    for tokens, dst in zip(num_tokens, my_assignment):
        input_splits[dst] += tokens
        input_frames[dst] += 1

    # ---- 4) src-view output splits (what I receive from each src) ----
    output_splits = [0] * world_size  # tokens I receive from each src
    output_frames = [0] * world_size  # frames I receive from each src
    cursor = 0
    for src in range(world_size):
        n = total_frames[src]
        for k in range(cursor, cursor + n):
            if assignment_list[k] == my_rank:
                output_splits[src] += loads[k]
                output_frames[src] += 1
        cursor += n

    # ---- 5) permute local tensors so frames are grouped by destination ----
    # all_to_all_single splits the input row-wise in tensor order, so we must
    # rearrange local frames into [dst=0 block, dst=1 block, ...] first.
    send_order = torch.argsort(
        torch.tensor(my_assignment, dtype=torch.long, device=device),
        stable=True,
    )
    # patch-level permutation: each frame contributes T*H*W consecutive rows.
    patches_per_local = grid_thw.prod(-1)
    local_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), patches_per_local.cumsum(0)])
    patch_perm = (
        torch.cat([torch.arange(local_starts[i], local_starts[i + 1], device=device) for i in send_order.tolist()])
        if num_frames > 0
        else torch.empty(0, dtype=torch.long, device=device)
    )
    hidden_states = hidden_states[patch_perm].contiguous()
    grid_thw = grid_thw[send_order].contiguous()

    # ---- 6) all_to_all dispatch ----
    hidden_states = all_to_all_single_autograd(
        hidden_states,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )

    grid_thw = all_to_all_single(
        grid_thw,
        output_split_sizes=output_frames,
        input_split_sizes=input_frames,
        group=group,
    )

    ctx = {
        "group": group,
        # Swap on the way back.
        "input_splits": output_splits,
        "output_splits": input_splits,
        # Inverse permutation for un-shuffling features back to local-original frame order.
        "send_order": send_order,
        "patches_per_local": patches_per_local,
    }
    return (self, hidden_states), {"grid_thw": grid_thw, **kwargs}, ctx


def output_dispatch(out, ctx):
    """Send ViT features back to the rank that originally owned each frame.

    Mirrors ``input_dispatch``: splits are swapped, ``all_to_all_single_autograd``
    routes gradients back along the reverse permutation. Both
    ``last_hidden_state`` (patch-level) and ``pooler_output`` (merger-reduced)
    are shipped — the latter uses splits rescaled by the merger factor.

    After the reverse all_to_all, features are laid out in the same dst-sorted
    order we used on the way out. Undo that permutation so the LLM's
    ``masked_scatter`` sees frames in the original local order.
    """
    in_splits = ctx["input_splits"]
    out_splits = ctx["output_splits"]
    group = ctx["group"]
    send_order: torch.Tensor = ctx["send_order"]
    patches_per_local: torch.Tensor = ctx["patches_per_local"]
    device = patches_per_local.device

    # last_hidden_state: same scale as patches, use splits as-is.
    last_hidden = all_to_all_single_autograd(
        out.last_hidden_state,
        output_split_sizes=out_splits,
        input_split_sizes=in_splits,
        group=group,
    )

    # pooler_output: patches // spatial_merge_size**2, infer scale from tensor.
    n_tokens = out.pooler_output.shape[0]
    n_patches = sum(in_splits)
    assert n_patches % n_tokens == 0, f"pooler_output tokens ({n_tokens}) doesn't divide patch total ({n_patches})"
    scale = n_patches // n_tokens
    pooler_in = [s // scale for s in in_splits]
    pooler_out = [s // scale for s in out_splits]

    pooler = all_to_all_single_autograd(
        out.pooler_output,
        output_split_sizes=pooler_out,
        input_split_sizes=pooler_in,
        group=group,
    )

    # ---- unpermute back to local-original frame order ----
    n_local = send_order.numel()
    if n_local > 0:
        # Inverse permutation on frame index.
        inv_order = torch.empty_like(send_order)
        inv_order[send_order] = torch.arange(n_local, device=device)

        # last_hidden patches: T*H*W per frame
        starts_full = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), patches_per_local[send_order].cumsum(0)]
        )
        full_perm = torch.cat(
            [torch.arange(starts_full[i], starts_full[i + 1], device=device) for i in inv_order.tolist()]
        )
        out.last_hidden_state = last_hidden[full_perm]

        # pooler tokens: (T*H*W // scale) per frame
        per_pooler = patches_per_local[send_order] // scale
        starts_pool = torch.cat([torch.zeros(1, dtype=torch.long, device=device), per_pooler.cumsum(0)])
        pool_perm = torch.cat(
            [torch.arange(starts_pool[i], starts_pool[i + 1], device=device) for i in inv_order.tolist()]
        )
        out.pooler_output = pooler[pool_perm]
    else:
        out.last_hidden_state = last_hidden
        out.pooler_output = pooler

    return out
