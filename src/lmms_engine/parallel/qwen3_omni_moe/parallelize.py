from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.utils
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    parallelize_module,
)
from tqdm import tqdm
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
)

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.models.qwen3_omni_moe.qwen3_omni_moe_experts import Qwen3OmniMoeExperts
from lmms_engine.utils.fsdp2_utils import fsdp2_load_full_state_dict

from .style import Qwen3OmniMoeParallelStyle

if TYPE_CHECKING:
    from lmms_engine.train.config import TrainingArguments


def stack_expert_params_qwen3_omni_moe(model: Qwen3OmniMoeThinkerForConditionalGeneration) -> None:
    logger.info("Stacking expert parameters for Qwen3-Omni MoE model")

    with torch.no_grad():
        for decoder_layer in tqdm(
            model.model.layers, desc="Stacking expert parameters", disable=not dist.get_rank() == 0
        ):
            # Check if this layer has MoE structure
            if not isinstance(decoder_layer.mlp, Qwen3OmniMoeThinkerTextSparseMoeBlock):
                continue

            if not hasattr(decoder_layer.mlp, "experts"):
                continue

            # Get expert configuration from the first expert
            first_expert = decoder_layer.mlp.experts[0]
            num_experts = len(decoder_layer.mlp.experts)
            hidden_dim = first_expert.down_proj.weight.size(0)
            intermediate_size = first_expert.down_proj.weight.size(1)
            act_fn = first_expert.act_fn

            new_experts = Qwen3OmniMoeExperts(
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                intermediate_size=intermediate_size,
                act_fn=act_fn,
            )

            up_proj_weights = [expert.up_proj.weight for expert in decoder_layer.mlp.experts]
            stacked_up_proj = torch.stack(up_proj_weights, dim=0)
            new_experts.up_proj = nn.Parameter(stacked_up_proj)

            down_proj_weights = [expert.down_proj.weight for expert in decoder_layer.mlp.experts]
            stacked_down_proj = torch.stack(down_proj_weights, dim=0)
            new_experts.down_proj = nn.Parameter(stacked_down_proj)

            gate_proj_weights = [expert.gate_proj.weight for expert in decoder_layer.mlp.experts]
            stacked_gate_proj = torch.stack(gate_proj_weights, dim=0)
            new_experts.gate_proj = nn.Parameter(stacked_gate_proj)

            del decoder_layer.mlp.experts
            decoder_layer.mlp.add_module("experts", new_experts)


def apply_qwen3_omni_moe_parallel(
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    ep_mesh: DeviceMesh,
    tp_mesh: DeviceMesh = None,
    **kwargs,
):
    assert tp_mesh is None, "Tensor Parallelism is not supported yet for Qwen3-Omni MoE"

    num_moe_layers = 0
    for decoder_layer in model.model.layers:
        # Only apply EP to MoE layers i.e. SparseMoeBlock
        if not isinstance(decoder_layer.mlp, Qwen3OmniMoeThinkerTextSparseMoeBlock):
            continue

        if not hasattr(decoder_layer.mlp, "experts"):
            continue

        module = decoder_layer.mlp
        ep_plan = Qwen3OmniMoeParallelStyle()
        parallelize_module(
            module.experts,
            device_mesh=ep_mesh,
            parallelize_plan=ep_plan,
        )
        num_moe_layers += 1

    logger.info(f"Applied Qwen3OmniMoeParallelStyle to {num_moe_layers} MoE layers")
    logger.info(f"Model structure: {model}")


def apply_qwen3_omni_moe_fsdp2(
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    train_args: "TrainingArguments",
    **kwargs,
):
    if not train_args.fsdp_config.get("transformer_layer_cls_to_wrap", None):
        logger.warning(
            "By default, we wrap the decoder layers for Qwen3-Omni MoE, the transformer_layer_cls_to_wrap will be ignored"
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

        def _experts_shard_placement_fn(param):
            return Shard(1)

        expert_fsdp_kwargs = dict(fsdp_kwargs)
        expert_fsdp_kwargs["mesh"] = pgm.process_group_manager.device_mesh["dp_shard_mod_ep"]
        expert_fsdp_kwargs["shard_placement_fn"] = _experts_shard_placement_fn

    # Wrap multimodal encoders with standard FSDP
    if hasattr(model, "visual") and model.visual is not None:
        fully_shard(model.visual, **fsdp_kwargs)

    if hasattr(model, "audio_tower") and model.audio_tower is not None:
        fully_shard(model.audio_tower, **fsdp_kwargs)

    for decoder_layer in model.model.layers:
        # Check if this is a MoE layer
        is_moe_layer = isinstance(decoder_layer.mlp, Qwen3OmniMoeThinkerTextSparseMoeBlock) and hasattr(
            decoder_layer.mlp, "experts"
        )

        if is_moe_layer and ep_size > 1:
            fully_shard(decoder_layer.mlp, **expert_fsdp_kwargs)
        elif is_moe_layer:
            fully_shard(decoder_layer.mlp, **fsdp_kwargs)

        fully_shard(decoder_layer.self_attn, **fsdp_kwargs)

    fully_shard(model.model.embed_tokens, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)


def apply_qwen3_omni_moe_parallelize_fn(
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    train_args: "TrainingArguments",
    **kwargs,
):
    ep_size = pgm.process_group_manager.ep_size
    stack_expert_params_qwen3_omni_moe(model)
    full_state_dict = model.state_dict()
    if ep_size > 1:
        ep_mesh = pgm.process_group_manager.device_mesh["dp_shard_in_ep"]
        apply_qwen3_omni_moe_parallel(model, ep_mesh=ep_mesh, **kwargs)

    apply_qwen3_omni_moe_fsdp2(model, train_args, **kwargs)
    fsdp2_load_full_state_dict(model, full_state_dict)
