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
memory under SP we run frame-balancing on the flat ``dp_cp_group`` (size
= dp_size × cp_size), but only let ``cp_rank == 0`` contribute frames so
the LPT load equals the real (de-duplicated) frame count. After the ViT
forward, features destined for a given dp_rank flow back to its ``cp_rank
== 0`` worker via the reverse all_to_all, then a CP-group all_reduce
(autograd-aware) broadcasts them to ``cp_rank > 0`` so every rank can do
its ``masked_scatter`` *before* the SP layer slices the seq.

Communication:
    * Metadata (per-rank token / frame counts) goes through ``all_gather_object``.
    * ``hidden_states`` uses ``all_to_all_single_autograd`` so gradients route
      back to the originating rank.
    * ``grid_thw`` uses plain ``all_to_all_single`` (no grad needed).
    * Optional CP broadcast uses autograd-aware ``all_reduce`` (sum), which
      back-props as another sum — equivalent to broadcasting forward and
      summing gradients on the source rank.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_reduce,
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
    cp_group: Optional[dist.ProcessGroup] = None,
    **kwargs,
) -> Tuple[Tuple, Dict[str, Any], Dict[str, Any]]:
    """Dispatch frames across ``group`` ahead of the ViT forward.

    When ``cp_group`` is provided (SP enabled), only ``cp_rank == 0`` workers
    contribute their frames to the LPT pool; ``cp_rank > 0`` workers report
    zero frames so they are pure receivers. This matches the dataloader's
    dp-only sharding (cp ranks within the same dp_rank hold duplicated input).

    Returns ``(new_args, new_kwargs, ctx)`` for ``wrap_vit_forward``.
    """
    world_size = dist.get_world_size(group=group)
    my_rank = dist.get_rank(group=group)
    device = hidden_states.device

    # Determine whether this rank is the "source" (contributes frames) within
    # its cp group. cp_rank > 0 holds duplicated input, so it should not push
    # frames into the LPT pool.
    cp_rank = dist.get_rank(group=cp_group) if cp_group is not None else 0
    is_source = cp_rank == 0

    # ---- 1) gather per-rank token/frame counts ----
    if is_source:
        num_tokens = grid_thw.prod(-1).tolist()
        num_frames = grid_thw.shape[0]
    else:
        num_tokens = []
        num_frames = 0
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
    if is_source:
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
    # rearrange local frames into [dst=0 block, dst=1 block, ...] first. Only
    # source ranks have real input to permute; cp_rank>0 sends an empty tensor.
    if is_source and num_frames > 0:
        send_order = torch.argsort(
            torch.tensor(my_assignment, dtype=torch.long, device=device),
            stable=True,
        )
        patches_per_local = grid_thw.prod(-1)
        local_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), patches_per_local.cumsum(0)])
        patch_perm = torch.cat(
            [torch.arange(local_starts[i], local_starts[i + 1], device=device) for i in send_order.tolist()]
        )
        send_hidden = hidden_states[patch_perm].contiguous()
        send_grid = grid_thw[send_order].contiguous()
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
        # frame order (only meaningful on source ranks).
        "send_order": send_order,
        "patches_per_local": patches_per_local,
        "is_source": is_source,
    }
    return (self, recv_hidden), {"grid_thw": recv_grid, **kwargs}, ctx


def output_dispatch(out, ctx):
    """Send ViT features back to the rank that originally owned each frame.

    Mirrors ``input_dispatch``: splits are swapped, ``all_to_all_single_autograd``
    routes gradients back along the reverse permutation. Both
    ``last_hidden_state`` (patch-level) and ``pooler_output`` (merger-reduced)
    are shipped — the latter uses splits rescaled by the merger factor.

    After the reverse all_to_all, features are laid out on the originating
    ``cp_rank == 0`` worker only (since cp_rank > 0 was a pure receiver in the
    forward dispatch). We then broadcast within the cp group via an
    autograd-aware all_reduce so every cp rank can run ``masked_scatter``
    before the SP layer slices the seq.

    Source ranks (cp_rank == 0) finally undo the dst-sorted permutation so the
    LLM's ``masked_scatter`` sees frames in the original local order.
    """
    in_splits = ctx["input_splits"]
    out_splits = ctx["output_splits"]
    group = ctx["group"]
    cp_group: Optional[dist.ProcessGroup] = ctx["cp_group"]
    send_order: torch.Tensor = ctx["send_order"]
    patches_per_local: torch.Tensor = ctx["patches_per_local"]
    is_source: bool = ctx["is_source"]
    device = out.last_hidden_state.device

    # last_hidden_state: same scale as patches, use splits as-is.
    last_hidden = all_to_all_single_autograd(
        out.last_hidden_state,
        output_split_sizes=out_splits,
        input_split_sizes=in_splits,
        group=group,
    )

    # pooler_output: patches // spatial_merge_size**2, infer scale from tensor.
    # Source ranks (cp_rank == 0) always have real patches; non-source ranks
    # have zero-sized inputs and outputs everywhere, so any ``scale`` works
    # (all the all_to_all splits below are 0). The cp broadcast at the bottom
    # restores the real shape.
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

    # ---- unpermute back to local-original frame order (source ranks only) ----
    n_local = send_order.numel()
    if is_source and n_local > 0:
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

    # ---- CP broadcast: source rank holds the real features, others hold
    # zero-sized tensors. We need every cp rank to end up with an identical
    # full-shape copy so ``masked_scatter`` works before SP slicing.
    if cp_group is not None and dist.get_world_size(group=cp_group) > 1:
        cp_size = dist.get_world_size(group=cp_group)

        # Agree on the canonical shape across the cp group (source rank knows
        # it; others have shape[0]==0). We pick the max along dim 0.
        local_shape = torch.tensor([last_hidden.shape[0], pooler.shape[0]], dtype=torch.long, device=device)
        max_shape = local_shape.clone()
        dist.all_reduce(max_shape, op=dist.ReduceOp.MAX, group=cp_group)
        n_last, n_pool = int(max_shape[0].item()), int(max_shape[1].item())

        hidden_dim = last_hidden.shape[1] if last_hidden.shape[0] > 0 else None
        pooler_dim = pooler.shape[1] if pooler.shape[0] > 0 else None
        # Hidden_dim must agree across cp ranks: gather it via max as well.
        dim_tensor = torch.tensor([hidden_dim or 0, pooler_dim or 0], dtype=torch.long, device=device)
        dist.all_reduce(dim_tensor, op=dist.ReduceOp.MAX, group=cp_group)
        hidden_dim = int(dim_tensor[0].item())
        pooler_dim = int(dim_tensor[1].item())

        # Pad non-source ranks up to (n_last, hidden_dim) / (n_pool, pooler_dim)
        # with zeros so all_reduce(SUM) reproduces the source values.
        if last_hidden.shape[0] != n_last:
            last_hidden = last_hidden.new_zeros((n_last, hidden_dim))
        if pooler.shape[0] != n_pool:
            pooler = pooler.new_zeros((n_pool, pooler_dim))

        # autograd-aware: backward through all_reduce(SUM) is itself a SUM,
        # which on the source rank receives the summed grads from all cp ranks.
        last_hidden = all_reduce(last_hidden, reduceOp="sum", group=cp_group)
        pooler = all_reduce(pooler, reduceOp="sum", group=cp_group)

    out.last_hidden_state = last_hidden
    out.pooler_output = pooler
    return out
