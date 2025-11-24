# Qwen3-VL MoE Training

## Overview

Qwen3-VL MoE is a Mixture-of-Experts vision-language model supporting image, video, and text with Expert Parallelism.

## Supported Features

| Feature | Support |
|---------|---------|
| **FSDP2** | ✅ |
| **USP** | ✅ |
| **Muon Optimizer** | ✅ |
| **Liger Kernel** | ✅ |
| **Packing** | ✅ |
| **NSA** | ❌ |
| **Expert Parallelism (EP)** | ✅ |

**Highlights**: Vision-Language MoE with EP (image, video, text)

## Quick Start

See the example configuration and run script:
- **Example Config**: [examples/qwen3_vl_moe/qwen3_vl_moe_ep8.yaml](../../examples/qwen3_vl_moe/qwen3_vl_moe_ep8.yaml)
- **Run Script**: [examples/qwen3_vl_moe/run.sh](../../examples/qwen3_vl_moe/run.sh)

## Key Configuration

```yaml
dataset_config:
  dataset_type: qwen3_vl_iterable
  processor_config:
    processor_type: qwen3_vl

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
    transformer_layer_cls_to_wrap: ["Qwen3VLMoeTextDecoderLayer"]
  ep_degree: 8  # Expert Parallelism degree
```

## Expert Parallelism

Expert Parallelism (EP) distributes MoE experts across GPUs to scale training. Set `ep_degree` to match your available GPUs (e.g., 2, 4, 8).

