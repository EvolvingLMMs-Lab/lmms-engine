# Qwen3-MoE Training

## Overview

Qwen3-MoE is a Mixture-of-Experts language model with Expert Parallelism support.

## Supported Features

| Feature | Support |
|---------|---------|
| **FSDP2** | ✅ |
| **USP** | ❌ |
| **Muon Optimizer** | ✅ |
| **Liger Kernel** | ✅ |
| **Packing** | ✅ |
| **NSA** | ❌ |
| **Expert Parallelism (EP)** | ✅ |

**Highlights**: Mixture-of-Experts with Expert Parallelism

## Quick Start

See the example configuration and run script:
- **Example Config**: [examples/qwen3_moe/qwen3_moe_ep8.yaml](../../examples/qwen3_moe/qwen3_moe_ep8.yaml)
- **Run Script**: [examples/qwen3_moe/run.sh](../../examples/qwen3_moe/run.sh)

## Key Configuration

```yaml
dataset_config:
  dataset_type: vision_iterable
  processor_config:
    processor_type: qwen2

model_config:
  load_from_pretrained_path: "Qwen/Qwen3-VL-30B-A3B-Instruct"
  attn_implementation: flash_attention_2
  monkey_patch_kwargs:
    patch_type: ["liger"]
    fused_linear_cross_entropy: true

trainer_args:
  use_liger_kernel: true
  use_rmpad: true
  fsdp2: true
  fsdp_config:
    transformer_layer_cls_to_wrap: ["Qwen3MoeDecoderLayer"]
  ep_degree: 8  # Expert Parallelism degree
```

## Expert Parallelism

Expert Parallelism (EP) distributes MoE experts across GPUs. Configure `ep_degree` to match your GPU count (e.g., 2, 4, 8).

