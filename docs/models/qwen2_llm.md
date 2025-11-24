# Qwen2.5 LLM Training

## Overview

Qwen2.5 is a large language model series for text generation and understanding tasks.

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

## Quick Start

See the example configuration and run script:
- **Example Config**: [examples/qwen2_5_llm/example_config.yaml](../../examples/qwen2_5_llm/example_config.yaml)
- **Run Script**: [examples/qwen2_5_llm/run.sh](../../examples/qwen2_5_llm/run.sh)

## Key Configuration

```yaml
model_config:
  load_from_pretrained_path: Qwen/Qwen2.5-1.5B-Instruct
  attn_implementation: flash_attention_2

trainer_args:
  use_liger_kernel: true
  use_rmpad: true
  fsdp2: true
  packing: true
```

