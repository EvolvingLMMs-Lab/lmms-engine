from functools import partial
from typing import Optional

import torch
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
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock


class Qwen3MoeParallelStyle(ParallelStyle):
    def __init__(
        self,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ) -> None:
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.use_local_output = use_local_output
        self.desired_input_layouts = (Replicate(),)

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        """
        Partition experts across devices based on expert parallelism (EP) size.

        Args:
            name: Module name
            mod: Module containing experts (typically a Qwen3MoeSparseMoeBlock)
            device_mesh: DeviceMesh representing the EP dimension

        Returns:
            Dictionary specifying how to partition the module
        """
        if isinstance(mod, Qwen3MoeSparseMoeBlock):
            # Get EP size and current device rank from the device mesh
            ep_size = device_mesh.size(0)  # Assuming EP is the first dimension
            current_rank = device_mesh.get_local_rank(0)  # Get rank in EP dimension

            # Calculate expert distribution
            total_experts = len(mod.experts)
            assert (
                total_experts % ep_size == 0
            ), f"Number of experts ({total_experts}) must be divisible by EP size ({ep_size})"

            experts_per_device = total_experts // ep_size

            # Calculate start and end indices for current device
            start_idx = current_rank * experts_per_device
            end_idx = start_idx + experts_per_device

            # Create new experts list with only the experts for this device
            new_experts = nn.ModuleList()
            for expert_idx in range(start_idx, end_idx):
                new_experts.append(mod.experts[expert_idx])

            # Update the module list with the new experts
            setattr(mod, "experts", new_experts)

    # The token all to all dispatch will be handled in the ops
    # Check the lmms_engine/models/qwen3_moe/qwen3_moe_ops.py
    # Since the expert module is a module list, we don't define the token
    # dispatch here
    @staticmethod
    def _input_fn(input_layouts, desired_input_layouts, mod, input, device_mesh):
        return input

    #  if not isinstance(input, tuple):
    #      input = (input,)

    #  new_input = []
    #  for input_tensor in input:
    #      if isinstance(input_tensor, torch.Tensor):
    #          input_tensor = DTensor.from_local(
    #              input_tensor, device_mesh, input_layouts, run_check=False
    #          )

    #      # transform the input layouts to the desired layouts of ColwiseParallel
    #      if input_layouts != desired_input_layouts:
    #          input_tensor = input_tensor.redistribute(
    #              placements=desired_input_layouts, async_op=True
    #          )
    #      new_input.append(input_tensor)
    #  return tuple(new_input)

    @staticmethod
    def _output_fn(output_layouts, use_local_output, mod, output, device_mesh):
        if isinstance(output, DTensor):
            output = output.redistribute(placements=output_layouts, async_op=True)
            output = output.wait()
            if use_local_output:
                output = output.to_local()
        return output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=Qwen3MoeParallelStyle._partition_fn,
            input_fn=partial(
                Qwen3MoeParallelStyle._input_fn,
                self.input_layouts,
                self.desired_input_layouts,
            ),
            output_fn=partial(
                Qwen3MoeParallelStyle._output_fn,
                self.output_layouts,
                self.use_local_output,
            ),
        )
