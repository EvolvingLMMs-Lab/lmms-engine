# Qwen3-Omni MoE Training

## Overview

Qwen3-Omni MoE is a multimodal Mixture-of-Experts model supporting image, audio, and text with Expert Parallelism.

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

**Highlights**: Multimodal MoE with EP (image, audio, text)

## Quick Start

See the example configuration:
- **Example Config**: [examples/qwen3_omni_moe_ep2.yaml](../../examples/qwen3_omni_moe_ep2.yaml)

## Key Configuration

```yaml
dataset_config:
  dataset_type: qwen_omni_iterable
  processor_config:
    processor_type: Qwen2_5OmniProcessor
  video_backend: qwen_omni_utils

model_config:
  attn_implementation: flash_attention_2
  monkey_patch_kwargs:
    patch_type: ["liger"]

trainer_args:
  use_liger_kernel: true
  use_rmpad: true
  fsdp2: true
  ep_degree: 2  # Expert Parallelism degree
```

## Expert Parallelism

Expert Parallelism (EP) distributes MoE experts across GPUs for efficient training. Set `ep_degree` based on your GPU availability.

