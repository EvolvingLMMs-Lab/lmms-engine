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
from lmms_engine.train.config import TrainingArguments
from lmms_engine.utils.fsdp2_utils import fsdp2_load_full_state_dict

from .style import Qwen3OmniMoeParallelStyle


def stack_expert_params_qwen3_omni_moe(model: Qwen3OmniMoeThinkerForConditionalGeneration) -> None:
    """
    Stack expert parameters for Qwen3-Omni MoE model.

    Converts the ModuleList-based expert structure into stacked parameter tensors
    for efficient computation and expert parallelism. Only processes MoE layers in
    the text decoder; vision and audio encoders are unchanged.

    Process:
    1. Iterate through all text decoder layers in model.model.layers
    2. Check if layer.mlp is a SparseMoeBlock (has .experts attribute)
    3. For each SparseMoeBlock, stack gate_proj, up_proj, down_proj weights
    4. Replace the ModuleList experts with optimized Qwen3OmniMoeExperts module

    Args:
        model: Qwen3-Omni MoE model with multimodal encoders and text decoder
               Structure: model.model.layers[i].mlp.experts (for MoE layers)

    Note:
        - Only text decoder layers may have MoE; vision/audio encoders do not
        - Not all decoder layers have MoE (depends on decoder_sparse_step config)
        - Stacking happens in-place with torch.no_grad() to avoid gradient tracking
        - After stacking, expert parameters become 3D tensors: [num_experts, ...]
    """
    logger.info("Stacking expert parameters for Qwen3-Omni MoE model")

    with torch.no_grad():
        for decoder_layer in tqdm(
            model.model.layers,
            desc="Stacking expert parameters",
            disable=not dist.get_rank() == 0
        ):
            # Check if this layer has MoE structure
            # Only SparseMoeBlock layers have .experts attribute
            if not isinstance(decoder_layer.mlp, Qwen3OmniMoeThinkerTextSparseMoeBlock):
                continue

            if not hasattr(decoder_layer.mlp, "experts"):
                continue

            # Get expert configuration from the first expert
            # All experts in a layer have identical architecture
            first_expert = decoder_layer.mlp.experts[0]
            num_experts = len(decoder_layer.mlp.experts)
            hidden_dim = first_expert.down_proj.weight.size(0)
            intermediate_size = first_expert.down_proj.weight.size(1)
            act_fn = first_expert.act_fn

            # Create optimized stacked expert module
            new_experts = Qwen3OmniMoeExperts(
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                intermediate_size=intermediate_size,
                act_fn=act_fn,
            )

            # Stack up_proj weights: [num_experts, intermediate_size, hidden_dim]
            up_proj_weights = [expert.up_proj.weight for expert in decoder_layer.mlp.experts]
            stacked_up_proj = torch.stack(up_proj_weights, dim=0)
            new_experts.up_proj = nn.Parameter(stacked_up_proj)

            # Stack down_proj weights: [num_experts, hidden_dim, intermediate_size]
            down_proj_weights = [expert.down_proj.weight for expert in decoder_layer.mlp.experts]
            stacked_down_proj = torch.stack(down_proj_weights, dim=0)
            new_experts.down_proj = nn.Parameter(stacked_down_proj)

            # Stack gate_proj weights: [num_experts, intermediate_size, hidden_dim]
            gate_proj_weights = [expert.gate_proj.weight for expert in decoder_layer.mlp.experts]
            stacked_gate_proj = torch.stack(gate_proj_weights, dim=0)
            new_experts.gate_proj = nn.Parameter(stacked_gate_proj)

            # Replace the old ModuleList experts with the new stacked version
            del decoder_layer.mlp.experts
            decoder_layer.mlp.add_module("experts", new_experts)


def apply_qwen3_omni_moe_parallel(
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    ep_mesh: DeviceMesh,
    tp_mesh: DeviceMesh = None,
    **kwargs,
):
    """
    Apply expert parallelism to Qwen3-Omni MoE model.

    Distributes expert parameters across the expert parallel (EP) mesh using DTensor.
    Each GPU in the EP mesh will hold a subset of experts (num_experts / ep_size).

    Process:
    1. Iterate through text decoder layers in model.model.layers
    2. For layers with MoE (SparseMoeBlock), apply Qwen3OmniMoeParallelStyle
    3. Expert parameters become DTensors sharded along dimension 0 (expert dimension)
    4. Token routing and combining logic is handled by the parallel style

    Args:
        model: Qwen3-Omni MoE model with stacked experts
        ep_mesh: DeviceMesh for expert parallelism (typically "ep" mesh)
        tp_mesh: DeviceMesh for tensor parallelism (not supported yet)

    Note:
        - Vision/audio encoders are NOT parallelized here (no EP needed)
        - Only text decoder MoE layers receive expert parallelism
        - Tensor parallelism (TP) is not yet supported for Qwen3-Omni MoE
        - After this call, expert weights are DTensors distributed across EP ranks
    """
    assert tp_mesh is None, "Tensor Parallelism is not supported yet for Qwen3-Omni MoE"

    num_moe_layers = 0
    for decoder_layer in model.model.layers:
        # Only apply EP to MoE layers (SparseMoeBlock)
        if not isinstance(decoder_layer.mlp, Qwen3OmniMoeThinkerTextSparseMoeBlock):
            continue

        if not hasattr(decoder_layer.mlp, "experts"):
            continue

        # Apply expert parallel style to distribute experts across EP mesh
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
    train_args: TrainingArguments,
    **kwargs,
):
    """
    Apply FSDP2 with mixed sharding strategies to Qwen3-Omni MoE model.

    Wraps model components with FSDP2 using different strategies for different modules:
    - Expert modules: Use dp_shard_mod_ep mesh with dim-1 sharding (when ep_size > 1)
    - All other modules: Use standard fsdp mesh with default sharding

    This mixed sharding approach allows experts to be sharded across both EP and DP
    dimensions while non-expert modules use standard data parallelism.

    Wrapping hierarchy:
    1. Audio encoder (model.audio_tower) - Standard FSDP [TODO: May need explicit wrapping]
    2. Vision encoder (model.visual) - Standard FSDP [TODO: May need explicit wrapping]
    3. Text decoder layers (model.model.layers):
       - Expert modules (decoder_layer.mlp for MoE layers) - EP-aware FSDP
       - Attention modules (decoder_layer.self_attn) - Standard FSDP
    4. Embeddings (model.model.embed_tokens) - Standard FSDP
    5. Root model - Standard FSDP

    Args:
        model: Qwen3-Omni MoE model (potentially with EP already applied)
        train_args: Training configuration with FSDP settings

    Note:
        - Gradient checkpointing is enabled if specified in train_args
        - Mixed precision policy is configured based on bf16/fp16 settings
        - When ep_size > 1, expert modules use dim-1 sharding to compose with EP's dim-0
        - Vision and audio encoders currently rely on implicit root-level wrapping
          (may need explicit wrapping for better memory efficiency)
    """
    if not train_args.fsdp_config.get("transformer_layer_cls_to_wrap", None):
        logger.warning(
            "By default, we wrap the decoder layers for Qwen3-Omni MoE; "
            "transformer_layer_cls_to_wrap will be ignored"
        )

    # Configure mixed precision policy
    if train_args.bf16:
        param_dtype = torch.bfloat16
    else:
        param_dtype = torch.float16

    # Enable gradient checkpointing if requested
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    reduce_dtype = getattr(torch, train_args.reduce_dtype)
    output_dtype = getattr(torch, train_args.output_dtype)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
    )

    # Standard FSDP mesh for data parallelism
    dp_mesh = pgm.process_group_manager.device_mesh["fsdp"]

    fsdp_kwargs = {
        "reshard_after_forward": getattr(train_args, "fsdp_config", {}).get("reshard_after_forward", True),
        "mp_policy": mp_policy,
        "mesh": dp_mesh,
    }

    ep_size = pgm.process_group_manager.ep_size
    if ep_size > 1:
        # Configure expert-specific FSDP settings
        # Prefer dim-1 sharding for expert weights to compose with EP's dim-0 sharding
        def _experts_shard_placement_fn(param):
            return Shard(1)

        expert_fsdp_kwargs = dict(fsdp_kwargs)
        expert_fsdp_kwargs["mesh"] = pgm.process_group_manager.device_mesh["dp_shard_mod_ep"]
        expert_fsdp_kwargs["shard_placement_fn"] = _experts_shard_placement_fn

    # Wrap multimodal encoders with standard FSDP
    # Vision encoder: processes image/video inputs
    if hasattr(model, "visual") and model.visual is not None:
        fully_shard(model.visual, **fsdp_kwargs)
        logger.info("Applied standard FSDP to vision encoder (model.visual)")

    # Audio encoder: processes audio inputs
    if hasattr(model, "audio_tower") and model.audio_tower is not None:
        fully_shard(model.audio_tower, **fsdp_kwargs)
        logger.info("Applied standard FSDP to audio encoder (model.audio_tower)")

    # Wrap text decoder layers with mixed sharding strategy
    for decoder_layer in model.model.layers:
        # Check if this is a MoE layer
        is_moe_layer = isinstance(decoder_layer.mlp, Qwen3OmniMoeThinkerTextSparseMoeBlock) and hasattr(
            decoder_layer.mlp, "experts"
        )

        if is_moe_layer and ep_size > 1:
            # MoE layers with EP: wrap expert module with EP-aware FSDP
            fully_shard(decoder_layer.mlp, **expert_fsdp_kwargs)
        elif is_moe_layer:
            # MoE layers without EP: standard FSDP wrapping
            fully_shard(decoder_layer.mlp, **fsdp_kwargs)

        # Wrap attention module with standard FSDP
        fully_shard(decoder_layer.self_attn, **fsdp_kwargs)

    # Wrap the embedding layer
    fully_shard(model.model.embed_tokens, **fsdp_kwargs)

    # Wrap the root model (includes lm_head and any remaining unwrapped components)
    fully_shard(model, **fsdp_kwargs)

    logger.info(
        f"Applied FSDP2 to Qwen3-Omni MoE model with ep_size={ep_size}, "
        f"mixed_precision={param_dtype}"
    )


def apply_qwen3_omni_moe_parallelize_fn(
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    train_args: TrainingArguments,
    **kwargs,
):
    """
    Main orchestration function for Qwen3-Omni MoE parallelization.

    Applies the complete parallelization pipeline:
    1. Stack expert parameters (convert ModuleList to stacked tensors)
    2. Save full state dict for later loading
    3. Apply expert parallelism (if ep_size > 1)
    4. Apply FSDP2 with mixed sharding strategies
    5. Load the full state dict into the parallelized model

    This function coordinates the entire distributed training setup for multimodal
    Qwen3-Omni MoE models, handling vision, audio, and text modalities with
    appropriate parallelization strategies for each.

    Args:
        model: Qwen3-Omni MoE model to parallelize
        train_args: Training configuration
        **kwargs: Additional arguments passed to sub-functions

    Note:
        - State dict is saved before parallelization and restored after
        - EP is only applied when ep_size > 1
        - FSDP2 is always applied with mixed sharding for multimodal components
        - The final model is ready for distributed training with proper sharding
    """
    ep_size = pgm.process_group_manager.ep_size

    # Step 1: Stack expert parameters for efficient computation
    stack_expert_params_qwen3_omni_moe(model)

    # Step 2: Save full state dict before applying parallelization
    # This will be loaded back after FSDP wrapping
    full_state_dict = model.state_dict()

    # Step 3: Apply expert parallelism if ep_size > 1
    if ep_size > 1:
        # Use dp_shard_in_ep as the EP mesh dimension
        ep_mesh = pgm.process_group_manager.device_mesh["dp_shard_in_ep"]
        apply_qwen3_omni_moe_parallel(model, ep_mesh=ep_mesh, **kwargs)

    # Step 4: Apply FSDP2 with mixed sharding strategies
    apply_qwen3_omni_moe_fsdp2(model, train_args, **kwargs)

    # Step 5: Load the full state dict into the FSDP-wrapped model
    fsdp2_load_full_state_dict(model, full_state_dict)

    logger.info("Successfully parallelized Qwen3-Omni MoE model")
