# dLLM (Diffusion Language Model) Training

## Overview

dLLM (Diffusion Language Model) is a masked diffusion-based language model adapted from Qwen3 architecture.

## Supported Features

| Feature | Support |
|---------|---------|
| **FSDP2** | ✅ |
| **USP** | ❌ |
| **Muon Optimizer** | ✅ |
| **Liger Kernel** | ❌ |
| **Packing** | ❌ |
| **NSA** | ❌ |
| **Expert Parallelism** | ❌ |

**Highlights**: Masked diffusion language model

## Quick Start

See the example configuration and run script:
- **Example Configs**: [examples/diffusion_language_model/](../../examples/diffusion_language_model/)
- **Run Script**: [examples/diffusion_language_model/run.sh](../../examples/diffusion_language_model/run.sh)
- **Documentation**: [examples/diffusion_language_model/README.md](../../examples/diffusion_language_model/README.md)

## Available Configurations

- `dllm_train_muon_single_gpu.yaml`: Single GPU with Muon optimizer
- `dllm_train_muon_multi_gpu_fsdp2.yaml`: Multi-GPU FSDP2 with Muon
- `dllm_train_adam_multi_gpu_deepspeed.yaml`: Multi-GPU DeepSpeed with Adam

## Key Configuration

```yaml
trainer_type: dllm_trainer

dataset_config:
  dataset_type: fineweb_edu
  dataset_format: hf_dataset
  dataset_path: HuggingFaceFW/fineweb-edu
  extra_kwargs:
    collator_type: dllm

model_config:
  load_from_config:
    model_type: qwen3_dllm
    config:
      vocab_size: 151936
      hidden_size: 1024
      num_hidden_layers: 24

trainer_args:
  use_muon: true
  fsdp2: true
  bf16: true
```

## Architecture

dLLM uses non-causal attention for masked diffusion language modeling, enabling bidirectional context understanding.

