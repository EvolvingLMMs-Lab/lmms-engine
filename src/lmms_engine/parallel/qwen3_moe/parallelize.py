import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.utils
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    parallelize_module,
)
from tqdm import tqdm
from transformers import Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeMLP,
    Qwen3MoeRMSNorm,
    Qwen3MoeSparseMoeBlock,
)

from .style import Qwen3MoeParallelStyle


def stack_expert_params(model: Qwen3MoeForCausalLM) -> None:
    logger.info("Stacking expert parameters for Qwen3Moe model")
    with torch.no_grad():
        for decoder_layer in tqdm(
            model.model.layers, desc="Stacking expert parameters", disable=not dist.get_rank() == 0
        ):
            up_proj_weights = [expert.up_proj.weight for expert in decoder_layer.mlp.experts]
            stacked_up_proj = torch.stack(up_proj_weights, dim=0)
            decoder_layer.mlp.register_parameter("up_proj", nn.Parameter(stacked_up_proj))

            down_proj_weights = [expert.down_proj.weight for expert in decoder_layer.mlp.experts]
            stacked_down_proj = torch.stack(down_proj_weights, dim=0)
            decoder_layer.mlp.register_parameter("down_proj", nn.Parameter(stacked_down_proj))

            gate_proj_weights = [expert.gate_proj.weight for expert in decoder_layer.mlp.experts]
            stacked_gate_proj = torch.stack(gate_proj_weights, dim=0)
            decoder_layer.mlp.register_parameter("gate_proj", nn.Parameter(stacked_gate_proj))
            decoder_layer.mlp.act_fn = decoder_layer.mlp.experts[0].act_fn

            del decoder_layer.mlp.experts


def _unstack_expert_params_post_hook(module, destination, prefix, local_metadata):
    """Post-hook to unstack expert parameters in the state dict.

    This hook is called after state_dict() is generated and transforms stacked expert
    parameters (shape: [num_experts, ...]) into individual expert parameters
    (shape: [1, ...] for each expert).

    Args:
        module: The module being hooked
        destination: The state dictionary to modify
        prefix: The prefix for module names
        local_metadata: The local metadata dictionary
    """
    logger.info("Unstacking expert parameters in state_dict for Qwen3Moe model")

    # Process stacked expert parameters and unstack them
    keys_to_remove = []
    keys_to_add = {}

    for key in list(destination.keys()):
        # Check if this is an expert parameter that needs unstacking
        if "mlp." in key and any(proj in key for proj in ["up_proj.weight", "down_proj.weight", "gate_proj.weight"]):
            value = destination[key]

            # Check if this is a stacked parameter (3D tensor with num_experts as first dimension)
            if isinstance(value, torch.Tensor) and len(value.shape) == 3:
                num_experts = value.shape[0]

                # Extract the parameter type
                param_type = None
                for proj_type in ["up_proj", "down_proj", "gate_proj"]:
                    if proj_type in key:
                        param_type = proj_type
                        break

                if param_type is not None:
                    # Mark this key for removal
                    keys_to_remove.append(key)

                    # Create unstacked parameters for each expert
                    for i in range(num_experts):
                        expert_key = key.replace(f"mlp.{param_type}", f"mlp.experts.{i}.{param_type}")
                        keys_to_add[expert_key] = value[i]

    # Remove original stacked parameters
    for key in keys_to_remove:
        del destination[key]

    # Add unstacked parameters
    destination.update(keys_to_add)


def apply_qwen3_moe_parallel(
    model: Qwen3MoeForCausalLM,
    ep_mesh: DeviceMesh,
    tp_mesh: DeviceMesh = None,
    **kwargs,
):
    assert tp_mesh is None, "Tensor Parallelism is not supported yet for Qwen3Moe"

    stack_expert_params(model)

    for decoder_layer in model.model.layers:
        module = decoder_layer
        ep_plan = Qwen3MoeParallelStyle()
        parallelize_module(
            module,
            device_mesh=ep_mesh,
            parallelize_plan=ep_plan,
        )

    for name, module in model.model.named_modules():
        if isinstance(module, Qwen3MoeSparseMoeBlock):
            parallel_style = PrepareModuleInput(
                input_layouts=Shard(0),
                desired_input_layouts=Shard(0),
                use_local_output=True,
            )
            parallelize_module(
                module,
                device_mesh=ep_mesh,
                parallelize_plan=parallel_style,
            )
        # if isinstance(module, Qwen3MoeAttention):
        #     attention_parallel_style = PrepareModuleInputOutput(
        #         input_kwarg_layouts={
        #             "hidden_states": Replicate(),
        #             "position_embeddings": Replicate(),
        #         },
        #         desired_input_kwarg_layouts={
        #             "hidden_states": Replicate(),
        #             "position_embeddings": Replicate(),
        #         },
        #         output_layouts=(Replicate(), None),
        #         desired_output_layouts=(Replicate(), None),
        #         use_local_output=True,
        #     )
        #     parallelize_module(
        #         module,
        #         device_mesh=ep_mesh,
        #         parallelize_plan=attention_parallel_style,
        #     )
        # No need to prepare input for the norm layer in model
        if (
            (
                isinstance(module, nn.Linear)
                or isinstance(module, Qwen3MoeMLP)
                or isinstance(module, Qwen3MoeRMSNorm)
            )
            and name != "norm"
        ):
            linear_parallel_style = PrepareModuleInputOutput(
                input_layouts=Shard(0),
                desired_input_layouts=Shard(0),
                output_layouts=Shard(0),
                desired_output_layouts=Shard(0),
                use_local_output=True,
            )
            parallelize_module(
                module,
                device_mesh=ep_mesh,
                parallelize_plan=linear_parallel_style,
            )
    model.register_state_dict_post_hook(_unstack_expert_params_post_hook)
    logger.info(f"Applied Qwen3MoeParallelStyle to {len(model.model.layers)} layers")
    logger.info(f"Model: {model}")
