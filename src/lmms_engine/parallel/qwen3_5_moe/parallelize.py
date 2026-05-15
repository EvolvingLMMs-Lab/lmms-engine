"""FSDP2 + Expert Parallel wiring for qwen3_5_moe.

qwen3_5_moe is multimodal: the top-level model class is
``Qwen3_5MoeForConditionalGeneration``, whose ``.model`` is the multimodal
wrapper ``Qwen3_5MoeModel`` (containing ``visual`` + ``language_model``).
Decoder layers live at ``model.model.language_model.layers`` (same shape as
qwen3_vl_moe — there is no ``.layers`` attribute on the outer wrapper).
"""

from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import parallelize_module
from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.utils.fsdp2_utils import fsdp2_load_full_state_dict

from .style import Qwen3_5MoeParallelStyle

if TYPE_CHECKING:
    from lmms_engine.train.config import TrainingArguments


def apply_qwen3_5_moe_parallel(
    model: Qwen3_5MoeForConditionalGeneration,
    ep_mesh: DeviceMesh,
    tp_mesh: DeviceMesh = None,
    **kwargs,
):
    assert tp_mesh is None, "Tensor Parallelism is not supported yet for Qwen3_5Moe"

    for decoder_layer in model.model.language_model.layers:
        module = decoder_layer.mlp
        ep_plan = Qwen3_5MoeParallelStyle()
        parallelize_module(
            module.experts,
            device_mesh=ep_mesh,
            parallelize_plan=ep_plan,
        )

    logger.info(f"Applied Qwen3_5MoeParallelStyle to {len(model.model.language_model.layers)} layers")


def apply_qwen3_5_moe_fsdp2(
    model: Qwen3_5MoeForConditionalGeneration,
    train_args: "TrainingArguments",
    **kwargs,
):
    if not train_args.fsdp_config.get("transformer_layer_cls_to_wrap", None):
        logger.warning("transformer_layer_cls_to_wrap ignored; qwen3_5_moe wraps decoder layers explicitly.")

    param_dtype = torch.bfloat16 if train_args.bf16 else torch.float16

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
    expert_fsdp_kwargs = None
    if ep_size > 1:

        def _experts_shard_placement_fn(param):
            return Shard(1)

        expert_fsdp_kwargs = dict(fsdp_kwargs)
        expert_fsdp_kwargs["mesh"] = pgm.process_group_manager.device_mesh["dp_shard_mod_ep"]
        expert_fsdp_kwargs["shard_placement_fn"] = _experts_shard_placement_fn

    # Wrap vision tower (same pattern as qwen3_vl_moe)
    if hasattr(model.model, "visual") and model.model.visual is not None:
        fully_shard(model.model.visual, **fsdp_kwargs)

    for decoder_layer in model.model.language_model.layers:
        # MoE block
        if ep_size > 1:
            fully_shard(decoder_layer.mlp, **expert_fsdp_kwargs)

        # Attention block — branch on layer_type
        if decoder_layer.layer_type == "linear_attention":
            fully_shard(decoder_layer.linear_attn, **fsdp_kwargs)
        else:  # "full_attention"
            fully_shard(decoder_layer.self_attn, **fsdp_kwargs)

    fully_shard(model.model.language_model.embed_tokens, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)


def apply_qwen3_5_moe_parallelize_fn(
    model: Qwen3_5MoeForConditionalGeneration,
    train_args: "TrainingArguments",
    **kwargs,
):
    ep_size = pgm.process_group_manager.ep_size
    full_state_dict = model.state_dict()
    if ep_size > 1:
        ep_mesh = pgm.process_group_manager.device_mesh["ep"]
        apply_qwen3_5_moe_parallel(model, ep_mesh=ep_mesh, **kwargs)

    apply_qwen3_5_moe_fsdp2(model, train_args, **kwargs)
    fsdp2_load_full_state_dict(model, full_state_dict)
