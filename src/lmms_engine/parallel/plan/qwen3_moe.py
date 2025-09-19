import torch.nn as nn
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module
from transformers import Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
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
    logger.info(f"Applied Qwen3MoeParallelStyle to {len(model.model.layers)} layers")
    logger.info(f"Model: {model}")
