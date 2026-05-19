"""FSDP2 + Expert Parallel wiring for aero_realtime.

aero_realtime owns ``language_model`` / ``vision_tower`` / ``audio_tower``
/ ``multi_modal_projector`` / ``lm_head`` directly under the
``AeroRealtimeForConditionalGeneration`` instance (no extra ``model.``
wrapper, unlike qwen3_*_moe ForConditionalGeneration which nests
decoder layers under ``model.model.language_model.layers``).

For MoE backbone families (``qwen3_vl_moe``, ``qwen3_5_moe``) we apply
expert-parallel to each language_model decoder layer's ``mlp.experts``,
then FSDP2-shard per-submodule (attention block + MoE block) mirroring
the inner family's parallelize fn. Dense families (``qwen3_vl``,
``qwen3_5``) skip EP and fall through to FSDP2-only sharding.

Mesh keys (from process_group_manager):
- ``device_mesh["fsdp"]``           — flattened dp/cp mesh for FSDP
- ``device_mesh["ep"]``             — EP mesh (only when ep_size>1)
- ``device_mesh["dp_shard_mod_ep"]`` — expert-FSDP mesh (only when ep_size>1)
"""

from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import parallelize_module

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.models.aero_realtime.backbone_registry import family_is_moe
from lmms_engine.utils.fsdp2_utils import fsdp2_load_full_state_dict

if TYPE_CHECKING:
    from lmms_engine.train.config import TrainingArguments


def _ep_style_cls(family: str):
    """Return the family-appropriate ParallelStyle class for MoE experts."""
    if family == "qwen3_vl_moe":
        from lmms_engine.parallel.qwen3_vl_moe.style import Qwen3VLMoeParallelStyle

        return Qwen3VLMoeParallelStyle
    if family == "qwen3_5_moe":
        from lmms_engine.parallel.qwen3_5_moe.style import Qwen3_5MoeParallelStyle

        return Qwen3_5MoeParallelStyle
    raise ValueError(f"no EP ParallelStyle for backbone_family={family}")


def apply_aero_realtime_parallel(
    model,
    ep_mesh: DeviceMesh,
    tp_mesh: DeviceMesh = None,
    **kwargs,
):
    """Apply EP ParallelStyle to each language_model decoder layer's
    ``mlp.experts``. Only meaningful for MoE backbone families."""
    assert tp_mesh is None, "Tensor Parallelism is not supported yet for AeroRealtime"

    family = model.config.backbone_family
    if not family_is_moe(family):
        raise ValueError(f"ep_degree>1 requires an MoE backbone_family; got {family}")

    style_cls = _ep_style_cls(family)
    num_moe_layers = 0
    for decoder_layer in model.language_model.layers:
        module = decoder_layer.mlp
        parallelize_module(
            module.experts,
            device_mesh=ep_mesh,
            parallelize_plan=style_cls(),
        )
        num_moe_layers += 1

    logger.info(f"Applied {style_cls.__name__} to {num_moe_layers} aero_realtime MoE layers")


def apply_aero_realtime_fsdp2(
    model,
    train_args: "TrainingArguments",
    **kwargs,
):
    """FSDP2-shard aero_realtime per-submodule, mirroring qwen3_5_moe /
    qwen3_vl_moe granularity but walking ``model.language_model.layers``
    directly (no extra ``.model.`` wrapper)."""
    if not train_args.fsdp_config.get("transformer_layer_cls_to_wrap", None):
        logger.warning(
            "transformer_layer_cls_to_wrap ignored; aero_realtime wraps decoder " "submodules (attn + mlp) explicitly."
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

    family = model.config.backbone_family
    is_moe = family_is_moe(family)
    ep_size = pgm.process_group_manager.ep_size

    expert_fsdp_kwargs = None
    if ep_size > 1:

        def _experts_shard_placement_fn(param):
            return Shard(1)

        expert_fsdp_kwargs = dict(fsdp_kwargs)
        expert_fsdp_kwargs["mesh"] = pgm.process_group_manager.device_mesh["dp_shard_mod_ep"]
        expert_fsdp_kwargs["shard_placement_fn"] = _experts_shard_placement_fn

    # --- Towers / projector (aero-specific; qwen3_*_moe parallelize fns
    # only wrap a single ``visual`` tower) ---
    if getattr(model, "vision_tower", None) is not None:
        fully_shard(model.vision_tower, **fsdp_kwargs)
    if getattr(model, "audio_tower", None) is not None:
        fully_shard(model.audio_tower, **fsdp_kwargs)
    if getattr(model, "multi_modal_projector", None) is not None:
        fully_shard(model.multi_modal_projector, **fsdp_kwargs)

    # --- Decoder layers: per-submodule wrap (attn block + mlp / experts) ---
    for decoder_layer in model.language_model.layers:
        # MoE expert block — only for MoE families with ep_size>1
        if is_moe and ep_size > 1:
            fully_shard(decoder_layer.mlp, **expert_fsdp_kwargs)

        # Attention block — qwen3_5 / qwen3_5_moe families have layer_type
        # branching (linear_attention vs full_attention); qwen3_vl(_moe) do not.
        layer_type = getattr(decoder_layer, "layer_type", None)
        if layer_type == "linear_attention":
            fully_shard(decoder_layer.linear_attn, **fsdp_kwargs)
        else:
            fully_shard(decoder_layer.self_attn, **fsdp_kwargs)

    fully_shard(model.language_model.embed_tokens, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)


def apply_aero_realtime_parallelize_fn(
    model,
    train_args: "TrainingArguments",
    **kwargs,
):
    """Entry registered as ``MODEL_TO_PARALLEL_METHOD['aero_realtime']``.

    Mirrors the qwen3_5_moe / qwen3_vl_moe two-stage flow:
      1. capture ``full_state_dict`` BEFORE parallelization
      2. apply EP (if ep_size>1; requires MoE family)
      3. apply FSDP2
      4. reload full state dict into the now-sharded model
    """
    ep_size = pgm.process_group_manager.ep_size
    full_state_dict = model.state_dict()

    if ep_size > 1:
        ep_mesh = pgm.process_group_manager.device_mesh["ep"]
        apply_aero_realtime_parallel(model, ep_mesh=ep_mesh, **kwargs)

    apply_aero_realtime_fsdp2(model, train_args, **kwargs)
    fsdp2_load_full_state_dict(model, full_state_dict)
    return model
