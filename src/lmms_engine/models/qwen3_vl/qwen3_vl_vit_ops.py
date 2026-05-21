"""Frame-parallel dispatch for Qwen3-VL ``Qwen3VLVisionModel.forward``.

Qwen3-VL's vision output includes patch-level ``last_hidden_state``, merged
``pooler_output``, and merged ``deepstack_features``. The dispatch mirrors the
Qwen3.5 DPxCP frame-parallel path and additionally routes deepstack tensors
with the same split scale as ``pooler_output``.
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
    return grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]


def _cp_frame_range(num_frames: int, cp_rank: int, cp_size: int) -> Tuple[int, int]:
    per_rank = (num_frames + cp_size - 1) // cp_size
    start = min(cp_rank * per_rank, num_frames)
    end = min(start + per_rank, num_frames)
    return start, end


def _all_gather_variable_dim0(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
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
    world_size = dist.get_world_size(group=group)
    my_rank = dist.get_rank(group=group)
    device = hidden_states.device

    cp_rank = dist.get_rank(group=cp_group) if cp_group is not None else 0
    cp_size = dist.get_world_size(group=cp_group) if cp_group is not None else 1

    frame_start, frame_end = _cp_frame_range(grid_thw.shape[0], cp_rank, cp_size)
    local_grid_thw = grid_thw[frame_start:frame_end].contiguous()
    patch_start = 0 if frame_start == 0 else int(_patches_per_row(grid_thw[:frame_start]).sum().item())
    local_num_patches = int(_patches_per_row(local_grid_thw).sum().item()) if local_grid_thw.numel() > 0 else 0
    local_hidden_states = hidden_states[patch_start : patch_start + local_num_patches].contiguous()

    num_tokens = local_grid_thw.prod(-1).tolist()
    num_frames = local_grid_thw.shape[0]
    total_tokens = [None for _ in range(world_size)]
    total_frames = [None for _ in range(world_size)]
    dist.all_gather_object(total_tokens, num_tokens, group=group)
    dist.all_gather_object(total_frames, num_frames, group=group)
    loads = [token for tokens in total_tokens for token in tokens]

    assignment_list, _ = lpt_balance(loads, num_ranks=world_size, frames_per_rank=total_frames)

    my_start = sum(total_frames[:my_rank])
    my_end = my_start + num_frames
    my_assignment = assignment_list[my_start:my_end]

    input_splits = [0] * world_size
    input_frames = [0] * world_size
    for tokens, dst in zip(num_tokens, my_assignment):
        input_splits[dst] += tokens
        input_frames[dst] += 1

    output_splits = [0] * world_size
    output_frames = [0] * world_size
    cursor = 0
    for src in range(world_size):
        n = total_frames[src]
        for k in range(cursor, cursor + n):
            if assignment_list[k] == my_rank:
                output_splits[src] += loads[k]
                output_frames[src] += 1
        cursor += n

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
        "input_splits": output_splits,
        "output_splits": input_splits,
        "send_order": send_order,
        "patches_per_local": patches_per_local,
    }
    return (self, recv_hidden), {"grid_thw": recv_grid, **kwargs}, ctx


def _dispatch_merged_tensor(
    x: torch.Tensor,
    *,
    in_splits: list[int],
    out_splits: list[int],
    scale: int,
    send_order: torch.Tensor,
    patches_per_local: torch.Tensor,
    group: dist.ProcessGroup,
    cp_group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    merged_in = [s // scale for s in in_splits]
    merged_out = [s // scale for s in out_splits]
    x = all_to_all_single_autograd(
        x,
        output_split_sizes=merged_out,
        input_split_sizes=merged_in,
        group=group,
    )

    n_local = send_order.numel()
    if n_local > 0:
        device = x.device
        inv_order = torch.empty_like(send_order)
        inv_order[send_order] = torch.arange(n_local, device=device)
        per_frame = patches_per_local[send_order] // scale
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), per_frame.cumsum(0)])
        perm = torch.cat([torch.arange(starts[i], starts[i + 1], device=device) for i in inv_order.tolist()])
        x = x[perm]

    if cp_group is not None and dist.get_world_size(group=cp_group) > 1:
        x = _all_gather_variable_dim0(x, cp_group)
    return x


def output_dispatch(out, ctx):
    in_splits = ctx["input_splits"]
    out_splits = ctx["output_splits"]
    group = ctx["group"]
    cp_group: Optional[dist.ProcessGroup] = ctx["cp_group"]
    send_order: torch.Tensor = ctx["send_order"]
    patches_per_local: torch.Tensor = ctx["patches_per_local"]
    device = out.last_hidden_state.device

    last_hidden = all_to_all_single_autograd(
        out.last_hidden_state,
        output_split_sizes=out_splits,
        input_split_sizes=in_splits,
        group=group,
    )

    n_tokens = out.pooler_output.shape[0]
    n_patches = sum(in_splits)
    if n_tokens > 0 and n_patches > 0:
        assert n_patches % n_tokens == 0, f"pooler_output tokens ({n_tokens}) doesn't divide patch total ({n_patches})"
        scale = n_patches // n_tokens
    else:
        scale = 1

    n_local = send_order.numel()
    if n_local > 0:
        inv_order = torch.empty_like(send_order)
        inv_order[send_order] = torch.arange(n_local, device=device)
        starts_full = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), patches_per_local[send_order].cumsum(0)]
        )
        full_perm = torch.cat(
            [torch.arange(starts_full[i], starts_full[i + 1], device=device) for i in inv_order.tolist()]
        )
        last_hidden = last_hidden[full_perm]

    if cp_group is not None and dist.get_world_size(group=cp_group) > 1:
        last_hidden = _all_gather_variable_dim0(last_hidden, cp_group)

    out.last_hidden_state = last_hidden
    out.pooler_output = _dispatch_merged_tensor(
        out.pooler_output,
        in_splits=in_splits,
        out_splits=out_splits,
        scale=scale,
        send_order=send_order,
        patches_per_local=patches_per_local,
        group=group,
        cp_group=cp_group,
    )
    if out.deepstack_features is not None:
        out.deepstack_features = [
            _dispatch_merged_tensor(
                feature,
                in_splits=in_splits,
                out_splits=out_splits,
                scale=scale,
                send_order=send_order,
                patches_per_local=patches_per_local,
                group=group,
                cp_group=cp_group,
            )
            for feature in out.deepstack_features
        ]
    return out
