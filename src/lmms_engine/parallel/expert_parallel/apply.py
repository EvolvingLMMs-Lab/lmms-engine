import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInputOutput,
    RowwiseParallel,
    parallelize_module,
)

from lmms_engine.models import MOEPARALLELPATCHER
from lmms_engine.utils.deep_attr import deep_getattr, has_nested_attr

from .expert_parallel import (
    ExpertParallel,
    ExpertTensorParallel,
    ReordererSequenceParallel,
    TensorParallel,
)
from .no_parallel import NoParallel


def validate_attr(model, model_dict, transformer_block_dict):
    for v in model_dict.values():
        if not has_nested_attr(model, v):
            raise ValueError(f"Model attribute '{v}' not found in the model.")
    for transformer_block in model.layers.values():
        for v in transformer_block_dict.values():
            if not has_nested_attr(transformer_block, v):
                raise ValueError(
                    f"Transformer block attribute '{v}' not found in the transformer block."
                )


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    ep_tp_mesh: DeviceMesh | None,
    etp_enabled: bool = True,
    model_dict: dict | None = None,
    transformer_block_dict: dict | None = None,
):
    validate_attr(model, model_dict, transformer_block_dict)
    layers = deep_getattr(model, model_dict["layers"])
    for transformer_block in layers.values():
        if not deep_getattr(transformer_block, transformer_block_dict["moe_enabled"]):
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                # all-gather for input, reduce-scatter for output
                transformer_block_dict["moe"]: PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=True,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                # replicate computation for the router
                transformer_block_dict["moe.router.gate"]: NoParallel(),
            }
            if ep_mesh is not None and not etp_enabled:
                # If TP is borrowed for EP, then split the tokens across TP ranks so that
                # the reorderer, the all-to-all comms, and routed experts computation
                # are effectively running Sequence Parallel (split along the folded bs*slen dim)
                moe_layer_plan.update(
                    {
                        transformer_block_dict[
                            "moe.reorderer"
                        ]: ReordererSequenceParallel()
                    }
                )
            if transformer_block.moe.shared_experts is not None:
                # input Replicate, output Partial
                moe_layer_plan.update(
                    {
                        transformer_block_dict[
                            "moe.shared_experts.w1"
                        ]: ColwiseParallel(),
                        transformer_block_dict[
                            "moe.shared_experts.w2"
                        ]: RowwiseParallel(output_layouts=Partial()),
                        transformer_block_dict[
                            "moe.shared_experts.w3"
                        ]: ColwiseParallel(),
                    }
                )
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None:
            experts_mesh = ep_mesh
            # input / output sharding on the batch / tokens dim
            experts_plan = ExpertParallel()
        elif etp_enabled:
            experts_mesh = ep_tp_mesh
            experts_plan = ExpertTensorParallel(tp_mesh=tp_mesh, ep_mesh=ep_mesh)
        else:
            experts_mesh = ep_mesh
            experts_plan = ExpertParallel()
        parallelize_module(
            module=deep_getattr(
                transformer_block, transformer_block_dict["moe.experts"]
            ),
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )
