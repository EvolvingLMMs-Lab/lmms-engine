from typing import TYPE_CHECKING
import types

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
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeTextSparseMoeBlock,
    Qwen3VLMoeTextExperts,
)

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.utils.fsdp2_utils import fsdp2_load_full_state_dict

from .style import Qwen3VLMoeParallelStyle

if TYPE_CHECKING:
    from lmms_engine.train.config import TrainingArguments


def stack_expert_params_qwen3_vl_moe(model: Qwen3VLMoeForConditionalGeneration) -> None:
    logger.info("Stacking expert parameters for Qwen3-VL MoE model")

    with torch.no_grad():
        for decoder_layer in tqdm(
            model.model.language_model.layers, desc="Stacking expert parameters", disable=not dist.get_rank() == 0
        ):
            # Check if this layer has MoE structure
            if not isinstance(decoder_layer.mlp, Qwen3VLMoeTextSparseMoeBlock):
                continue

            if not hasattr(decoder_layer.mlp, "experts"):
                continue

            # No need to copy parameters
            def ep_forward(self, *routed_input):
                out_experts_split = []

                # Handle DTensor case
                if isinstance(self.down_proj, DTensor):
                    down_proj = self.down_proj.to_local()
                    gate_up_proj = self.gate_up_proj.to_local()
                else:
                    down_proj = self.down_proj
                    gate_up_proj = self.gate_up_proj

                # Process each local expert
                for idx, x in enumerate(routed_input):
                    gate_up = torch.matmul(x, gate_up_proj[idx])
                    gate, up = gate_up.chunk(2, dim=-1)
                    hidden = up * self.act_fn(gate)
                    hidden = torch.matmul(hidden, down_proj[idx])
                    out_experts_split.append(hidden)

                return torch.cat(out_experts_split, dim=0)

            decoder_layer.mlp.experts.forward = types.MethodType(ep_forward, decoder_layer.mlp.experts)


def apply_qwen3_vl_moe_parallel(
    model: Qwen3VLMoeForConditionalGeneration,
    ep_mesh: DeviceMesh,
    tp_mesh: DeviceMesh = None,
    **kwargs,
):
    assert tp_mesh is None, "Tensor Parallelism is not supported yet for Qwen3-VL MoE"

    num_moe_layers = 0
    for decoder_layer in model.model.language_model.layers:
        # Only apply EP to MoE layers i.e. SparseMoeBlock
        if not isinstance(decoder_layer.mlp, Qwen3VLMoeTextSparseMoeBlock):
            continue

        if not hasattr(decoder_layer.mlp, "experts"):
            continue

        module = decoder_layer.mlp
        ep_plan = Qwen3VLMoeParallelStyle()
        parallelize_module(
            module.experts,
            device_mesh=ep_mesh,
            parallelize_plan=ep_plan,
        )
        num_moe_layers += 1

    logger.info(f"Applied Qwen3VLMoeParallelStyle to {num_moe_layers} MoE layers")
    logger.info(f"Model structure: {model}")


def apply_qwen3_vl_moe_fsdp2(
    model: Qwen3VLMoeForConditionalGeneration,
    train_args: "TrainingArguments",
    **kwargs,
):
    if not train_args.fsdp_config.get("transformer_layer_cls_to_wrap", None):
        logger.warning(
            "By default, we wrap the decoder layers for Qwen3-VL MoE, the transformer_layer_cls_to_wrap will be ignored"
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

    # Wrap vision encoder with standard FSDP (uses dp_mesh only)
    if hasattr(model.model, "visual") and model.model.visual is not None:
        fully_shard(model.model.visual, **fsdp_kwargs)

    # Wrap text model decoder layers
    for decoder_layer in model.model.language_model.layers:
        # Check if this is a MoE layer
        is_moe_layer = isinstance(decoder_layer.mlp, Qwen3VLMoeTextSparseMoeBlock) and hasattr(
            decoder_layer.mlp, "experts"
        )

        if is_moe_layer and ep_size > 1:
            fully_shard(decoder_layer.mlp, **expert_fsdp_kwargs)
        elif is_moe_layer:
            fully_shard(decoder_layer.mlp, **fsdp_kwargs)

        fully_shard(decoder_layer.self_attn, **fsdp_kwargs)

    fully_shard(model.model.language_model.embed_tokens, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)


def apply_qwen3_vl_moe_parallelize_fn(
    model: Qwen3VLMoeForConditionalGeneration,
    train_args: "TrainingArguments",
    **kwargs,
):
    ep_size = pgm.process_group_manager.ep_size
    stack_expert_params_qwen3_vl_moe(model)
    full_state_dict = model.state_dict()
    if ep_size > 1:
        ep_mesh = pgm.process_group_manager.device_mesh["dp_shard_in_ep"]
        apply_qwen3_vl_moe_parallel(model, ep_mesh=ep_mesh, **kwargs)

    apply_qwen3_vl_moe_fsdp2(model, train_args, **kwargs)
    fsdp2_load_full_state_dict(model, full_state_dict)
