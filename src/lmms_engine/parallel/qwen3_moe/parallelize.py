from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import parallelize_module
from transformers import Qwen3MoeForCausalLM

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.utils.fsdp2_utils import fsdp2_load_full_state_dict

from .style import Qwen3MoeParallelStyle

if TYPE_CHECKING:
    from lmms_engine.train.config import TrainingArguments


def apply_qwen3_moe_parallel(
    model: Qwen3MoeForCausalLM,
    ep_mesh: DeviceMesh,
    tp_mesh: DeviceMesh = None,
    **kwargs,
):
    assert tp_mesh is None, "Tensor Parallelism is not supported yet for Qwen3Moe"

    for decoder_layer in model.model.layers:
        module = decoder_layer.mlp
        ep_plan = Qwen3MoeParallelStyle()
        parallelize_module(
            module.experts,
            device_mesh=ep_mesh,
            parallelize_plan=ep_plan,
        )

    logger.info(f"Applied Qwen3MoeParallelStyle to {len(model.model.layers)} layers")
    logger.info(f"Model: {model}")


def apply_qwen3_moe_fsdp2(
    model: Qwen3MoeForCausalLM,
    train_args: "TrainingArguments",
    **kwargs,
):
    if not train_args.fsdp_config.get("transformer_layer_cls_to_wrap", None):
        logger.warning(
            "By default, we wrap the decoder layers for Qwen3Moe, the transformer_layer_cls_to_wrap will be ignored"
        )

    if train_args.bf16:
        param_dtype = torch.bfloat16
    else:
        param_dtype = torch.float16

    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    reduce_dtype = getattr(torch, train_args.reduce_dtype)
    output_dtype = getattr(torch, train_args.output_dtype)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
    )

    dp_mesh = pgm.process_group_manager.device_mesh["fsdp"]

    fsdp_kwargs = {
        "reshard_after_forward": getattr(train_args, "fsdp_config", {}).get("reshard_after_forward", True),
        "mp_policy": mp_policy,
        "mesh": dp_mesh,
    }

    ep_size = pgm.process_group_manager.ep_size
    if ep_size > 1:
        # Prefer dim-1 sharding for expert weights when composing with EP shard on dim-0
        def _experts_shard_placement_fn(param):
            return Shard(1)

        expert_fsdp_kwargs = dict(fsdp_kwargs)
        expert_fsdp_kwargs["mesh"] = pgm.process_group_manager.device_mesh["dp_shard_mod_ep"]
        expert_fsdp_kwargs["shard_placement_fn"] = _experts_shard_placement_fn

    for decoder_layer in model.model.layers:
        expert_mod = decoder_layer.mlp

        if ep_size > 1:
            fully_shard(expert_mod, **expert_fsdp_kwargs)

        fully_shard(decoder_layer.self_attn, **fsdp_kwargs)

    # Shard the embed tokens
    fully_shard(model.model.embed_tokens, **fsdp_kwargs)
    # Shard the root model
    fully_shard(model, **fsdp_kwargs)


def apply_qwen3_moe_parallelize_fn(
    model: Qwen3MoeForCausalLM,
    train_args: "TrainingArguments",
    **kwargs,
):
    ep_size = pgm.process_group_manager.ep_size
    full_state_dict = model.state_dict()
    if ep_size > 1:
        ep_mesh = pgm.process_group_manager.device_mesh["ep"]
        apply_qwen3_moe_parallel(model, ep_mesh=ep_mesh, **kwargs)

    apply_qwen3_moe_fsdp2(model, train_args, **kwargs)
    fsdp2_load_full_state_dict(model, full_state_dict)
