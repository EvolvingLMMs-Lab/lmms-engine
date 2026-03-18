from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

from lmms_engine.utils.import_utils import is_transformers_version_greater_or_equal_to

_IS_TRANSFORMERS_5 = is_transformers_version_greater_or_equal_to("5.0")

if _IS_TRANSFORMERS_5:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
else:
    from lmms_engine.models.qwen3_moe.qwen3_moe_experts import Qwen3MoeExperts

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.parallel.expert_parallel.utils import (
    _compute_permute_indices,
    _token_combine,
    _token_dispatch,
)


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
        self.input_splits = None
        self.output_splits = None
        self.permute_indices = None
        self.num_experts = None

    def _input_fn(self, inputs, mesh: DeviceMesh):
        routed_input, num_tokens_per_expert = inputs
        if pgm.process_group_manager.ep_world_size > 1:
            (
                routed_input,
                input_splits,
                output_splits,
                num_tokens_per_expert_group,
            ) = _token_dispatch(routed_input, num_tokens_per_expert)
            permute_indices, split_sizes = _compute_permute_indices(
                torch.tensor(num_tokens_per_expert_group, device=routed_input.device),
                pgm.process_group_manager.ep_world_size,
                self.num_experts // pgm.process_group_manager.ep_world_size,
            )
            routed_input = routed_input[permute_indices]
            routed_input = torch.split(
                routed_input[: sum(output_splits)],
                split_size_or_sections=split_sizes,
                dim=0,
            )
            self.input_splits = input_splits
            self.output_splits = output_splits
            self.permute_indices = permute_indices

        else:
            routed_input = torch.split(
                routed_input,
                split_size_or_sections=num_tokens_per_expert.tolist(),
                dim=0,
            )
        return routed_input

    def _output_fn(self, output, mesh: DeviceMesh):
        if pgm.process_group_manager.ep_world_size > 1:
            output[self.permute_indices] = output.clone()
            output = _token_combine(output, self.input_splits, self.output_splits)
        return output

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        if isinstance(mod, Qwen3MoeExperts):
            # Distribute the expert parameters across the expert parallel mesh
            expert_parallel_dim = 0  # Assuming experts are sharded along the first dimension

            if _IS_TRANSFORMERS_5:
                # transformers >= 5.0: fused gate_up_proj + down_proj
                mod.register_parameter(
                    "gate_up_proj",
                    nn.Parameter(
                        distribute_tensor(
                            mod.gate_up_proj,
                            device_mesh,
                            [Shard(expert_parallel_dim)],
                        )
                    ),
                )
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
            else:
                # transformers < 5.0: separate gate_proj + up_proj + down_proj
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

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, Qwen3MoeExperts):
            self.num_experts = module.num_experts

        return distribute_module(
            module,
            device_mesh,
            partition_fn=Qwen3MoeParallelStyle._partition_fn,
            input_fn=self._input_fn,
            output_fn=self._output_fn,
        )
