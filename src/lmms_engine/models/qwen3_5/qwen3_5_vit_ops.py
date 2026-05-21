"""Frame-parallel dispatch for Qwen3.5 ``Qwen3_5VisionModel.forward``.

The upstream forward signature is::

    Qwen3_5VisionModel.forward(self, hidden_states, grid_thw, **kwargs)

where
    hidden_states : (total_patches, C*T*P*P)   — packed
    grid_thw      : (num_segments, 3)          — each row contributes T*H*W patches

``input_dispatch`` redistributes frames across the DP (or DP×CP) group via
LPT so each rank handles a balanced number of ViT patches, runs the original
forward locally on the received slice, and the matching ``output_dispatch``
gathers features back so each rank ends up with the features for its own
original frames.

Sequence-parallel (CP) integration
----------------------------------
When SP is on, the dataloader still shards by ``dp_rank`` only, so the
``cp_rank`` axis sees the *same* frames duplicated. To actually cut ViT
memory under SP while keeping the autograd graph symmetric across CP ranks,
each CP rank first takes a deterministic shard of its duplicated local frames
(roughly ``num_frames / cp_size``). We then run the usual LPT balancing over
the flat ``dp_cp_group`` (size = dp_size × cp_size). After the ViT forward,
features flow back to the CP rank that owned each local frame shard; a
CP-group autograd-aware all-gather reconstructs the full local-dp feature set
on every CP rank so each rank can do its ``masked_scatter`` before the SP layer
slices the seq.

Communication:
    * Metadata (per-rank token / frame counts) goes through ``all_gather_object``.
    * ``hidden_states`` uses ``all_to_all_single_autograd`` so gradients route
      back to the originating rank.
    * ``grid_thw`` uses plain ``all_to_all_single`` (no grad needed).
    * Optional CP gather uses autograd-aware ``all_gather_tensor_autograd``;
      gradients route back to the CP rank that owned each frame shard.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_gather_tensor_autograd,
    all_to_all_single,
    all_to_all_single_autograd,
)

from lmms_engine.parallel.vit_parallel.balance import lpt_balance


def _patches_per_row(grid_thw: torch.Tensor) -> torch.Tensor:
    """Patches contributed by each grid_thw row: T * H * W."""
    return grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]


def _cp_frame_range(num_frames: int, cp_rank: int, cp_size: int) -> Tuple[int, int]:
    """Contiguous frame shard for this CP rank.

    CP ranks hold duplicated dataloader frames. Sharding those frames before
    the dp×cp LPT removes the old source/receiver asymmetry where cp_rank==0
    sent real frames and cp_rank>0 sent zeros.
    """
    per_rank = (num_frames + cp_size - 1) // cp_size
    start = min(cp_rank * per_rank, num_frames)
    end = min(start + per_rank, num_frames)
    return start, end


def _all_gather_variable_dim0(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Autograd-aware all-gather for variable dim-0 tensors.

    Pads local tensors to the CP-group max length, gathers with autograd, then
    removes padding and concatenates ranks in group order.
    """
    world_size = dist.get_world_size(group=group)
    local_len = x.shape[0]
    lengths = [None for _ in range(world_size)]
    dist.all_gather_object(lengths, local_len, group=group)
    max_len = max(lengths)
    if local_len < max_len:
        pad_shape = list(x.shape)
        pad_shape[0] = max_len - local_len
        x = torch.cat([x, x.new_zeros(pad_shape)], dim=0)
    gathered = all_gather_tensor_autograd(x, gather_dim=0, group=group)
    chunks = gathered.split(max_len, dim=0)
    return torch.cat([chunk[:length] for chunk, length in zip(chunks, lengths)], dim=0)


def input_dispatch(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    cp_group: Optional[dist.ProcessGroup] = None,
    **kwargs,
) -> Tuple[Tuple, Dict[str, Any], Dict[str, Any]]:
    """Dispatch frames across ``group`` ahead of the ViT forward.

    When ``cp_group`` is provided (SP enabled), cp ranks hold duplicated
    dataloader frames. Each cp rank contributes a deterministic local shard of
    those frames before the dp×cp LPT, so all cp ranks have a real source path.

    Returns ``(new_args, new_kwargs, ctx)`` for ``wrap_vit_forward``.
    """
    world_size = dist.get_world_size(group=group)
    my_rank = dist.get_rank(group=group)
    device = hidden_states.device

    cp_rank = dist.get_rank(group=cp_group) if cp_group is not None else 0
    cp_size = dist.get_world_size(group=cp_group) if cp_group is not None else 1

    frame_start, frame_end = _cp_frame_range(grid_thw.shape[0], cp_rank, cp_size)
    local_grid_thw = grid_thw[frame_start:frame_end].contiguous()
    patch_start = 0 if frame_start == 0 else int(_patches_per_row(grid_thw[:frame_start]).sum().item())
    local_num_patches = _patches_per_row(local_grid_thw).sum().item() if local_grid_thw.numel() > 0 else 0
    local_hidden_states = hidden_states[patch_start : patch_start + local_num_patches].contiguous()

    # ---- 1) gather per-rank token/frame counts ----
    num_tokens = local_grid_thw.prod(-1).tolist()
    num_frames = local_grid_thw.shape[0]
    total_tokens = [None for _ in range(world_size)]
    total_frames = [None for _ in range(world_size)]
    dist.all_gather_object(total_tokens, num_tokens, group=group)
    dist.all_gather_object(total_frames, num_frames, group=group)
    loads = [token for tokens in total_tokens for token in tokens]

    # ---- 2) LPT ----
    assignment_list, _ = lpt_balance(loads, num_ranks=world_size, frames_per_rank=total_frames)

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
    if num_frames > 0:
        send_order = torch.argsort(
            torch.tensor(my_assignment, dtype=torch.long, device=device),
            stable=True,
        )
        patches_per_local = local_grid_thw.prod(-1)
        local_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), patches_per_local.cumsum(0)])
        patch_perm = torch.cat(
            [torch.arange(local_starts[i], local_starts[i + 1], device=device) for i in send_order.tolist()]
        )
        send_hidden = local_hidden_states[patch_perm].contiguous()
        send_grid = local_grid_thw[send_order].contiguous()
    else:
        send_order = torch.empty(0, dtype=torch.long, device=device)
        patches_per_local = torch.empty(0, dtype=torch.long, device=device)
        send_hidden = hidden_states.new_zeros((0, hidden_states.shape[1]))
        send_grid = grid_thw.new_zeros((0, grid_thw.shape[1]))

    # ---- 6) all_to_all dispatch ----
    recv_hidden = all_to_all_single_autograd(
        send_hidden,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )

    recv_grid = all_to_all_single(
        send_grid,
        output_split_sizes=output_frames,
        input_split_sizes=input_frames,
        group=group,
    )

    ctx = {
        "group": group,
        "cp_group": cp_group,
        # Swap on the way back.
        "input_splits": output_splits,
        "output_splits": input_splits,
        # Inverse permutation for un-shuffling features back to local-original
        # frame-shard order.
        "send_order": send_order,
        "patches_per_local": patches_per_local,
    }
    return (self, recv_hidden), {"grid_thw": recv_grid, **kwargs}, ctx


def output_dispatch(out, ctx):
    """Send ViT features back to the rank that originally owned each frame.

    Mirrors ``input_dispatch``: splits are swapped, ``all_to_all_single_autograd``
    routes gradients back along the reverse permutation. Both
    ``last_hidden_state`` (patch-level) and ``pooler_output`` (merger-reduced)
    are shipped — the latter uses splits rescaled by the merger factor.

    After the reverse all_to_all, each CP rank holds the features for the
    local frame shard it contributed. We then broadcast/sum within the cp group
    so every cp rank reconstructs the full local-dp feature set and can run
    ``masked_scatter`` before the SP layer slices the seq.

    Each rank first undoes the dst-sorted permutation for its local frame
    shard so CP all_reduce(SUM) reconstructs the original per-dp frame order.
    """
    in_splits = ctx["input_splits"]
    out_splits = ctx["output_splits"]
    group = ctx["group"]
    cp_group: Optional[dist.ProcessGroup] = ctx["cp_group"]
    send_order: torch.Tensor = ctx["send_order"]
    patches_per_local: torch.Tensor = ctx["patches_per_local"]
    device = out.last_hidden_state.device

    # last_hidden_state: same scale as patches, use splits as-is.
    last_hidden = all_to_all_single_autograd(
        out.last_hidden_state,
        output_split_sizes=out_splits,
        input_split_sizes=in_splits,
        group=group,
    )

    # pooler_output: patches // spatial_merge_size**2, infer scale from tensor.
    # Ranks with an empty local shard have zero-sized inputs and outputs, so
    # any ``scale`` works (all the all_to_all splits below are 0). The cp
    # broadcast at the bottom restores the full local-dp shape.
    n_tokens = out.pooler_output.shape[0]
    n_patches = sum(in_splits)
    if n_tokens > 0 and n_patches > 0:
        assert n_patches % n_tokens == 0, f"pooler_output tokens ({n_tokens}) doesn't divide patch total ({n_patches})"
        scale = n_patches // n_tokens
    else:
        scale = 1
    pooler_in = [s // scale for s in in_splits]
    pooler_out = [s // scale for s in out_splits]

    pooler = all_to_all_single_autograd(
        out.pooler_output,
        output_split_sizes=pooler_out,
        input_split_sizes=pooler_in,
        group=group,
    )

    # ---- unpermute back to local frame-shard order ----
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
        last_hidden = last_hidden[full_perm]

        # pooler tokens: (T*H*W // scale) per frame
        per_pooler = patches_per_local[send_order] // max(scale, 1)
        starts_pool = torch.cat([torch.zeros(1, dtype=torch.long, device=device), per_pooler.cumsum(0)])
        pool_perm = torch.cat(
            [torch.arange(starts_pool[i], starts_pool[i + 1], device=device) for i in inv_order.tolist()]
        )
        pooler = pooler[pool_perm]

    # ---- CP gather+broadcast: each cp rank holds a disjoint contiguous frame
    # shard from the same dp sample. Gather shards in cp-rank order so every cp
    # rank reconstructs the same full local-dp feature sequence before SP
    # slicing. This keeps all cp ranks as real sources in autograd (no
    # cp_rank==0-only source path).
    if cp_group is not None and dist.get_world_size(group=cp_group) > 1:
        last_hidden = _all_gather_variable_dim0(last_hidden, cp_group)
        pooler = _all_gather_variable_dim0(pooler, cp_group)

    out.last_hidden_state = last_hidden
    out.pooler_output = pooler
    return out
