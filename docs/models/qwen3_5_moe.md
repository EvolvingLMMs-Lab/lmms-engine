# Qwen3.5-MoE Training

## Overview

Qwen3.5-MoE (`Qwen/Qwen3.6-35B-A3B`) is a **multimodal** Mixture-of-Experts model
with a vision tower plus a hybrid-attention MoE language model. Each decoder
layer is either a **linear-attention** layer (gated delta net) or a **full
softmax-attention** layer, selected per layer via
`config.text_config.layer_types[i]`. The MoE block contains a
**shared_expert** alongside the routed experts.

The top-level multimodal class is `Qwen3_5MoeForConditionalGeneration`
(`model_type = "qwen3_5_moe"`).

## Supported Features

| Feature | Support |
|---------|---------|
| **FSDP2** | ✅ |
| **USP / Sequence Parallel** | ❌ (linear-attention path is not SP-safe) |
| **Muon Optimizer** | ✅ |
| **Liger Kernel** | ✅ |
| **Packing** | ✅ (rmpad) |
| **NSA** | ❌ |
| **Expert Parallelism (EP)** | ✅ |

**Highlights**: Hybrid attention (linear / full), `shared_expert` + routed
experts, Expert Parallelism via the custom `Qwen3_5MoeExperts` `ParallelStyle`.

## Quick Start

See the example configuration and run script:
- **Example Config**: [examples/qwen3_5_moe/qwen3_5_moe_ep8.yaml](../../examples/qwen3_5_moe/qwen3_5_moe_ep8.yaml)
- **Run Script**: [examples/qwen3_5_moe/run.sh](../../examples/qwen3_5_moe/run.sh)

Verified end-to-end with `cicd/run_traincicd.sh --model-name qwen3_5_moe --gpu-count 4`.

## Key Configuration

```yaml
model_config:
  load_from_pretrained_path: "Qwen/Qwen3.6-35B-A3B"
  # CRITICAL: Qwen3_5MoeConfig is registered in both causal_lm and
  # image_text_to_text auto-mappings. Without this line we'd silently load the
  # text-only Qwen3_5MoeForCausalLM instead of the multimodal
  # Qwen3_5MoeForConditionalGeneration.
  model_general_type: image_text_to_text
  attn_implementation: flash_attention_2
  monkey_patch_kwargs:
    # Two patches registered separately for qwen3_5_moe; runner applies them
    # in order. "rmpad" accepts no kwargs; the listed kwargs go to "liger".
    patch_type: ["liger", "rmpad"]
    fused_linear_cross_entropy: true
    rms_norm: true
    swiglu: true

trainer_args:
  use_liger_kernel: true
  use_rmpad: true
  fsdp2: true
  fsdp_config:
    transformer_layer_cls_to_wrap: ["Qwen3_5MoeDecoderLayer"]
  sp_ulysses_degree: 1   # SP is not supported
  ep_degree: 8           # Expert Parallelism degree
```

## Expert Parallelism

Expert Parallelism (EP) distributes the routed MoE experts across GPUs.
Configure `ep_degree` to match your GPU count (e.g., 2, 4, 8). The FSDP wrap
branches on `decoder_layer.layer_type` (`linear_attn` vs `self_attn`) so that
the gated-delta-net and softmax-attention layers each get the right sharding
plan, while the experts are sharded along the expert dimension via the
`Qwen3_5MoeExperts` `ParallelStyle`.

## Merging EP Checkpoints

FSDP2 + EP checkpoints store expert weights as **multi-axis DTensors** with
placements like `(Shard(dim=1), Shard(dim=0))` on a 2D mesh
`(dp_shard_mod_ep, ep)`. The checkpoint merger consolidates these correctly
as of this branch.

Merge a checkpoint into a single HF-loadable directory with:

```bash
python -m lmms_engine.merger \
    --checkpoint_path ./output/qwen3_5_moe_a3b_ep8/checkpoint-1000 \
    --output_path ./output/qwen3_5_moe_a3b_ep8/merged-1000 \
    --model_general_type image_text_to_text
```

`--model_general_type image_text_to_text` is **required** for the same reason
as at train time: without it the merger instantiates `Qwen3_5MoeForCausalLM`
(text-only) from the saved config and crashes with
`'Qwen3_5MoeConfig' has no attribute 'vocab_size'` (the vocab lives on
`config.text_config`, which the multimodal wrapper knows about but the
text-only causal-LM does not).
