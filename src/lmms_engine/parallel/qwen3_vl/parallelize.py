from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.utils.fsdp2_utils import fsdp2_load_full_state_dict

if TYPE_CHECKING:
    from lmms_engine.train.config import TrainingArguments


def _check_divisible(name: str, value: int, degree: int) -> None:
    if value % degree != 0:
        raise ValueError(f"{name} ({value}) must be divisible by tp_degree ({degree})")


def _validate_qwen3_vl_tp_config(model: Qwen3VLForConditionalGeneration, train_args: "TrainingArguments") -> None:
    tp_degree = pgm.process_group_manager.tp_world_size
    sp_degree = pgm.process_group_manager.cp_world_size

    if tp_degree < 1:
        raise ValueError(f"tp_degree must be >= 1, got {tp_degree}")
    if train_args.ep_degree > 1:
        raise ValueError("ep_degree > 1 is not supported for plain qwen3_vl")
    if tp_degree == 1:
        return

    text_config = model.config.text_config
    _check_divisible("hidden_size", text_config.hidden_size, tp_degree)
    _check_divisible("intermediate_size", text_config.intermediate_size, tp_degree)
    _check_divisible("num_attention_heads", text_config.num_attention_heads, tp_degree)
    _check_divisible("num_key_value_heads", text_config.num_key_value_heads, tp_degree)

    local_attention_heads = text_config.num_attention_heads // tp_degree
    if sp_degree > 1 and local_attention_heads % sp_degree != 0:
        raise ValueError(
            f"num_attention_heads / tp_degree ({local_attention_heads}) must be divisible by "
            f"sp_ulysses_degree ({sp_degree})"
        )


def apply_qwen3_vl_parallel(
    model: Qwen3VLForConditionalGeneration,
    tp_mesh: DeviceMesh,
    **kwargs,
) -> None:
    tp_plan = {
        "self_attn.q_proj": ColwiseParallel(use_local_output=True),
        "self_attn.k_proj": ColwiseParallel(use_local_output=True),
        "self_attn.v_proj": ColwiseParallel(use_local_output=True),
        "self_attn.o_proj": RowwiseParallel(use_local_output=True),
        "mlp.gate_proj": ColwiseParallel(use_local_output=True),
        "mlp.up_proj": ColwiseParallel(use_local_output=True),
        "mlp.down_proj": RowwiseParallel(use_local_output=True),
    }

    for decoder_layer in model.model.language_model.layers:
        parallelize_module(decoder_layer, device_mesh=tp_mesh, parallelize_plan=tp_plan)

    logger.info(f"Applied Qwen3-VL text TP to {len(model.model.language_model.layers)} decoder layers")


def apply_qwen3_vl_fsdp2(
    model: Qwen3VLForConditionalGeneration,
    train_args: "TrainingArguments",
    **kwargs,
) -> None:
    if not train_args.fsdp_config.get("transformer_layer_cls_to_wrap", None):
        logger.warning("transformer_layer_cls_to_wrap ignored; qwen3_vl wraps modules explicitly.")

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

    fsdp_kwargs = {
        "reshard_after_forward": getattr(train_args, "fsdp_config", {}).get("reshard_after_forward", True),
        "mp_policy": mp_policy,
        "mesh": pgm.process_group_manager.device_mesh["fsdp"],
    }

    if hasattr(model.model, "visual") and model.model.visual is not None:
        fully_shard(model.model.visual, **fsdp_kwargs)

    for decoder_layer in model.model.language_model.layers:
        fully_shard(decoder_layer.self_attn, **fsdp_kwargs)
        fully_shard(decoder_layer.mlp, **fsdp_kwargs)

    fully_shard(model.model.language_model.embed_tokens, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)


def apply_qwen3_vl_parallelize_fn(
    model: Qwen3VLForConditionalGeneration,
    train_args: "TrainingArguments",
    **kwargs,
) -> None:
    _validate_qwen3_vl_tp_config(model, train_args)

    full_state_dict = model.state_dict()
    if pgm.process_group_manager.tp_world_size > 1:
        tp_mesh = pgm.process_group_manager.device_mesh["tp"]
        apply_qwen3_vl_parallel(model, tp_mesh=tp_mesh, **kwargs)

    apply_qwen3_vl_fsdp2(model, train_args, **kwargs)
    fsdp2_load_full_state_dict(model, full_state_dict)
