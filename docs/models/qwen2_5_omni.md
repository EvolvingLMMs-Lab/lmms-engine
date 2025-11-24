# Qwen2.5-Omni Training

## Overview

Qwen2.5-Omni is a unified multimodal model supporting image, audio, and text understanding.

## Supported Features

| Feature | Support |
|---------|---------|
| **FSDP2** | ✅ |
| **USP** | ✅ |
| **Muon Optimizer** | ✅ |
| **Liger Kernel** | ✅ |
| **Packing** | ✅ |
| **NSA** | ❌ |
| **Expert Parallelism** | ❌ |

**Highlights**: Unified multimodal (image, audio, text)

## Quick Start

See the example configuration and run script:
- **Example Config**: [examples/qwen2_5_omni/example_config.yaml](../../examples/qwen2_5_omni/example_config.yaml)
- **Run Script**: [examples/qwen2_5_omni/run.sh](../../examples/qwen2_5_omni/run.sh)

## Key Configuration

```yaml
dataset_config:
  dataset_type: qwen_omni_iterable
  processor_config:
    processor_type: Qwen2_5OmniProcessor
    audio_max_length: 60
  video_backend: qwen_omni_utils

model_config:
  load_from_pretrained_path: Qwen/Qwen2.5-Omni-7B
  attn_implementation: flash_attention_2

trainer_args:
  use_liger_kernel: true
  use_rmpad: true
  fsdp2: true
```

