import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)

import lmms_engine.parallel.process_group_manager as pgm


def _token_dispatch(routed_input: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
    ep_size = pgm.process_group_manager.ep_world_size
    ep_group = pgm.process_group_manager.ep_group
    with torch.no_grad():
        num_tokens_per_expert_group = all_to_all_single(
            num_tokens_per_expert,
            None,
            None,
            group=ep_group,
        )

        # Need to wait explicitly because it is used by a triton kernel later
        # which doesn't realize that AsyncCollectiveTensor needs unwrapping
        num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(num_tokens_per_expert_group)
        input_splits = num_tokens_per_expert.view(ep_size, -1).sum(dim=1).to(torch.device("cpu"), non_blocking=True)
        # NOTE: this would incur a device-to-host sync
        output_splits = (
            num_tokens_per_expert_group.view(ep_size, -1).sum(dim=1).to(torch.device("cpu"), non_blocking=False)
        )
        input_splits = input_splits.tolist()
        output_splits = output_splits.tolist()
        num_tokens_per_expert_group = num_tokens_per_expert_group.tolist()
    # perform all-to-all
    routed_input = all_to_all_single_autograd(
        routed_input,
        output_splits,
        input_splits,
        ep_group,
    )
    return routed_input, input_splits, output_splits, num_tokens_per_expert_group


def _token_combine(routed_output, input_splits, output_splits):
    ep_group = pgm.process_group_manager.ep_group
    routed_output = all_to_all_single_autograd(
        routed_output,
        input_splits,
        output_splits,
        ep_group,
    )
    return routed_output


def _compute_permute_indices(
    num_tokens_per_expert: torch.Tensor,
    num_ranks: int,
    num_experts: int,
) -> torch.Tensor:
    device = num_tokens_per_expert.device
    total_tokens = num_tokens_per_expert.sum().item()

    source_counts_2d = num_tokens_per_expert.view(num_ranks, num_experts)
    source_offsets_flat = torch.cumsum(num_tokens_per_expert, dim=0) - num_tokens_per_expert
    source_offsets_2d = source_offsets_flat.view(num_ranks, num_experts)

    counts_t_flat = source_counts_2d.transpose(0, 1).flatten()
    offsets_t_flat = source_offsets_2d.transpose(0, 1).flatten()

    repeated_offsets = torch.repeat_interleave(offsets_t_flat, counts_t_flat)

    group_ends = torch.cumsum(counts_t_flat, dim=0)
    group_starts = group_ends - counts_t_flat
    all_indices = torch.arange(total_tokens, device=device)
    local_indices = all_indices - torch.repeat_interleave(group_starts, counts_t_flat)

    permute_indices_alt = repeated_offsets + local_indices
    return permute_indices_alt, source_counts_2d.sum(dim=0).tolist()
