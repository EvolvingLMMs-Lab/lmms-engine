import torch
import torch.nn as nn
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import (
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    parallelize_module,
)
from transformers import Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeMLP,
    Qwen3MoeRMSNorm,
    Qwen3MoeSparseMoeBlock,
)

from lmms_engine.parallel.expert_parallel import Qwen3MoeParallelStyle


def apply_qwen3_moe_parallel(
    model: Qwen3MoeForCausalLM,
    ep_mesh: DeviceMesh,
    tp_mesh: DeviceMesh = None,
    **kwargs,
):
    assert tp_mesh is None, "Tensor Parallelism is not supported yet for Qwen3Moe"

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
                input_layouts=Replicate(),
                desired_input_layouts=Replicate(),
                use_local_output=True,
            )
            parallelize_module(
                module,
                device_mesh=ep_mesh,
                parallelize_plan=parallel_style,
            )
        if isinstance(module, Qwen3MoeAttention):
            attention_parallel_style = PrepareModuleInputOutput(
                input_kwarg_layouts={
                    "hidden_states": Replicate(),
                    "position_embeddings": Replicate(),
                },
                desired_input_kwarg_layouts={
                    "hidden_states": Replicate(),
                    "position_embeddings": Replicate(),
                },
                output_layouts=(Replicate(), None),
                desired_output_layouts=(Replicate(), None),
                use_local_output=True,
            )
            parallelize_module(
                module,
                device_mesh=ep_mesh,
                parallelize_plan=attention_parallel_style,
            )
        # No need to prepare input for the norm layer in model
        if (
            isinstance(module, nn.Linear)
            or isinstance(module, Qwen3MoeMLP)
            or isinstance(module, Qwen3MoeRMSNorm)
            and name != "norm"
        ):
            linear_parallel_style = PrepareModuleInputOutput(
                input_layouts=Replicate(),
                desired_input_layouts=Replicate(),
                output_layouts=Replicate(),
                desired_output_layouts=Replicate(),
                use_local_output=True,
            )
            parallelize_module(
                module,
                device_mesh=ep_mesh,
                parallelize_plan=linear_parallel_style,
            )
    logger.info(f"Applied Qwen3MoeParallelStyle to {len(model.model.layers)} layers")
    logger.info(f"Model: {model}")
