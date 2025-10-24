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
        self.input_layouts = (input_layouts or Shard(0),)
        self.output_layouts = (output_layouts or Shard(0),)
        self.use_local_output = use_local_output
        self.desired_input_layouts = (Shard(0),)

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        if isinstance(mod, Qwen3MoeSparseMoeBlock):
            # Distribute the expert parameters across the expert parallel mesh
            expert_parallel_dim = 0  # Assuming experts are sharded along the first dimension

            mod.register_parameter("up_proj", nn.Parameter(distribute_tensor(
                mod.up_proj,
                device_mesh,
                [Shard(expert_parallel_dim)],
            )))
            mod.register_parameter("down_proj", nn.Parameter(distribute_tensor(
                mod.down_proj,
                device_mesh,
                [Shard(expert_parallel_dim)],
            )))
            mod.register_parameter("gate_proj", nn.Parameter(distribute_tensor(
                mod.gate_proj,
                device_mesh,
                [Shard(expert_parallel_dim)],
            )))

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
