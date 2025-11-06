from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.models.qwen3_omni_moe.qwen3_omni_moe_experts import Qwen3OmniMoeExperts
from lmms_engine.parallel.expert_parallel.utils import (
    _compute_permute_indices,
    _token_combine,
    _token_dispatch,
)


class Qwen3OmniMoeParallelStyle(ParallelStyle):
    """
    Parallel style for Qwen3-Omni MoE models with expert parallelism support.

    This class implements the PyTorch DTensor ParallelStyle interface to enable
    expert parallelism for Qwen3-Omni MoE models. It handles:
    - Sharding expert parameters across EP ranks (dimension 0)
    - Token dispatch via all-to-all communication
    - Token combine to restore original token ordering

    The implementation is modality-agnostic - it works with any token type
    (text, audio, vision) since the routing logic operates on the abstract
    token representation after modality encoding.

    Args:
        input_layouts: DTensor placement for inputs (default: Shard(0))
        output_layouts: DTensor placement for outputs (default: Shard(0))
        use_local_output: Whether to convert output DTensor to local tensor (default: True)

    Attributes:
        num_experts: Total number of experts in the MoE layer
        input_splits: Split sizes for token dispatch (set during forward pass)
        output_splits: Split sizes for token combine (set during forward pass)
        permute_indices: Indices for reordering tokens after expert computation

    Example:
        >>> # Model with 60 experts, EP world size = 4
        >>> # Each GPU holds 15 experts and processes tokens routed to those experts
        >>> style = Qwen3OmniMoeParallelStyle()
        >>> # Apply style to MoE layer
        >>> distributed_module = style._apply(moe_layer, device_mesh)
    """

    def __init__(
        self,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ) -> None:
        super().__init__()
        # Default to Shard(0) placement for both input and output
        self.input_layouts = (input_layouts or Shard(0),)
        self.output_layouts = (output_layouts or Shard(0),)
        self.use_local_output = use_local_output
        self.desired_input_layouts = (Shard(0),)

        # These will be set during the forward pass
        self.input_splits = None
        self.output_splits = None
        self.permute_indices = None
        self.num_experts = None

    def _input_fn(self, inputs, mesh: DeviceMesh):
        """
        Token dispatch function with all-to-all communication across EP ranks.

        This function handles the input preparation for expert processing:
        1. Receives tokens already routed to experts locally
        2. If EP > 1: Performs all-to-all to send tokens to the correct EP rank
        3. Permutes and splits tokens by local expert assignment
        4. Returns tuple of tensors, one per local expert

        The dispatch is agnostic to token modality (text/audio/vision) - all tokens
        are treated uniformly based on their expert assignments.

        Args:
            inputs: Tuple of (routed_input, num_tokens_per_expert)
                - routed_input: Tensor of shape [total_tokens, hidden_dim]
                  containing all tokens routed to experts in expert order
                - num_tokens_per_expert: Tensor of shape [num_experts] with
                  token counts for each expert

            mesh: DeviceMesh for distributed tensor operations

        Returns:
            Tuple of tensors, one per local expert. Each tensor has shape
            [num_tokens_for_expert_i, hidden_dim].

        Communication Pattern (EP > 1):
            - All-to-all sends tokens to the rank that owns their assigned expert
            - Example with 4 experts, EP=2:
              * Rank 0: Owns experts [0, 1], receives tokens for experts 0, 1
              * Rank 1: Owns experts [2, 3], receives tokens for experts 2, 3
        """
        routed_input, num_tokens_per_expert = inputs

        if pgm.process_group_manager.ep_world_size > 1:
            # Perform all-to-all token dispatch across EP ranks
            # Returns:
            # - routed_input: tokens for local experts after all-to-all
            # - input_splits: how many tokens this rank sent to each rank
            # - output_splits: how many tokens this rank received from each rank
            # - num_tokens_per_expert_group: token counts per local expert
            (
                routed_input,
                input_splits,
                output_splits,
                num_tokens_per_expert_group,
            ) = _token_dispatch(routed_input, num_tokens_per_expert)

            # Compute permutation indices to reorder tokens by local expert
            # This ensures tokens are grouped by expert for efficient computation
            permute_indices, split_sizes = _compute_permute_indices(
                torch.tensor(num_tokens_per_expert_group, device=routed_input.device),
                pgm.process_group_manager.ep_world_size,
                self.num_experts // pgm.process_group_manager.ep_world_size,
            )

            # Apply permutation to group tokens by expert
            routed_input = routed_input[permute_indices]

            # Split into per-expert tensors for local experts
            # Only use tokens up to sum(output_splits) as that's what we received
            routed_input = torch.split(
                routed_input[: sum(output_splits)],
                split_size_or_sections=split_sizes,
                dim=0,
            )

            # Cache these for use in output_fn
            self.input_splits = input_splits
            self.output_splits = output_splits
            self.permute_indices = permute_indices

        else:
            # No EP: simply split tokens by expert assignment
            # Each expert gets its assigned tokens
            routed_input = torch.split(
                routed_input,
                split_size_or_sections=num_tokens_per_expert.tolist(),
                dim=0,
            )

        return routed_input

    def _output_fn(self, output, mesh: DeviceMesh):
        """
        Token combine function with all-to-all communication across EP ranks.

        This function handles the output gathering after expert processing:
        1. Reverse the permutation applied in _input_fn
        2. If EP > 1: Performs all-to-all to return tokens to original ranks
        3. Returns concatenated output in the original token order

        Args:
            output: Concatenated expert outputs with shape [total_local_tokens, hidden_dim]
            mesh: DeviceMesh for distributed tensor operations

        Returns:
            Tensor with shape [total_tokens, hidden_dim] containing all expert outputs
            in the original token order.

        Communication Pattern (EP > 1):
            - All-to-all sends expert outputs back to the rank that originally owned them
            - Restores the original token ordering before expert routing
        """
        if pgm.process_group_manager.ep_world_size > 1:
            # Reverse the permutation to restore token order before all-to-all
            output[self.permute_indices] = output.clone()

            # Perform all-to-all token combine to return tokens to original ranks
            # Uses the input_splits and output_splits from _input_fn to determine
            # how to scatter tokens back to their original locations
            output = _token_combine(output, self.input_splits, self.output_splits)

        return output

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        """
        Partition function to shard expert parameters across EP ranks.

        This function distributes the stacked expert parameters across the expert
        parallel mesh by sharding along dimension 0 (the expert dimension). Each
        EP rank will own a subset of experts.

        For example, with 60 experts and EP=4:
        - Rank 0: Experts 0-14 (15 experts)
        - Rank 1: Experts 15-29 (15 experts)
        - Rank 2: Experts 30-44 (15 experts)
        - Rank 3: Experts 45-59 (15 experts)

        The expert parameters are already stacked into single tensors by the
        stack_expert_params_qwen3_omni_moe function before this is called.

        Args:
            name: Name of the module being partitioned
            mod: Module instance to partition (should be Qwen3OmniMoeExperts)
            device_mesh: DeviceMesh defining the expert parallel topology

        Note:
            This function modifies the module in-place by replacing its parameters
            with DTensor versions that are sharded across the EP mesh.
        """
        if isinstance(mod, Qwen3OmniMoeExperts):
            # Distribute the expert parameters across the expert parallel mesh
            # Shard along dimension 0 (expert dimension)
            expert_parallel_dim = 0

            # Shard gate projection weights
            # Original shape: [num_experts, intermediate_size, hidden_dim]
            # After sharding: [num_experts/ep_size, intermediate_size, hidden_dim] per rank
            mod.register_parameter(
                "gate_proj",
                nn.Parameter(
                    distribute_tensor(
                        mod.gate_proj,
                        device_mesh,
                        [Shard(expert_parallel_dim)],
                    )
                ),
            )

            # Shard up projection weights
            # Original shape: [num_experts, intermediate_size, hidden_dim]
            # After sharding: [num_experts/ep_size, intermediate_size, hidden_dim] per rank
            mod.register_parameter(
                "up_proj",
                nn.Parameter(
                    distribute_tensor(
                        mod.up_proj,
                        device_mesh,
                        [Shard(expert_parallel_dim)],
                    )
                ),
            )

            # Shard down projection weights
            # Original shape: [num_experts, hidden_dim, intermediate_size]
            # After sharding: [num_experts/ep_size, hidden_dim, intermediate_size] per rank
            mod.register_parameter(
                "down_proj",
                nn.Parameter(
                    distribute_tensor(
                        mod.down_proj,
                        device_mesh,
                        [Shard(expert_parallel_dim)],
                    )
                ),
            )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """
        Apply the parallel style to a module.

        This method wraps the PyTorch DTensor distribute_module function with
        our custom partition, input, and output functions to enable expert
        parallelism.

        Args:
            module: The module to apply parallelism to (typically Qwen3OmniMoeExperts)
            device_mesh: DeviceMesh defining the expert parallel topology

        Returns:
            The module with expert parallelism applied (parameters converted to DTensors,
            input/output functions registered for token dispatch/combine)

        Note:
            This is called automatically by the DTensor framework when applying
            parallelism to a model.
        """
        # Store num_experts for use in _input_fn
        if isinstance(module, Qwen3OmniMoeExperts):
            self.num_experts = module.num_experts

        return distribute_module(
            module,
            device_mesh,
            partition_fn=Qwen3OmniMoeParallelStyle._partition_fn,
            input_fn=self._input_fn,
            output_fn=self._output_fn,
        )
