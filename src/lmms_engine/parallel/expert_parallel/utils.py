import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)

import lmms_engine.parallel.process_group_manager as pgm


def _token_dispatch(
    routed_input: torch.Tensor, num_tokens_per_expert: torch.Tensor
) -> torch.Tensor:
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
        num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
            num_tokens_per_expert_group
        )
        input_splits = (
            num_tokens_per_expert.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        # NOTE: this would incur a device-to-host sync
        output_splits = (
            num_tokens_per_expert_group.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=False)
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


def sync_gradients(model):
    shared_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and "expert" in name:
            shared_params.append(param)
    world_size = dist.get_world_size()
    buffer_size = sum(p.numel() for p in shared_params)
    buffer = torch.zeros(buffer_size, device=shared_params[0].device)
    with torch.no_grad():
        offset = 0
        for param in shared_params:
            if param.grad is not None:
                numel = param.grad.numel()
                buffer[offset : offset + numel].copy_(param.grad.view(-1))
                offset += numel
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        buffer /= world_size
        offset = 0
        for param in shared_params:
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(buffer[offset : offset + numel].view_as(param))
                offset += numel
