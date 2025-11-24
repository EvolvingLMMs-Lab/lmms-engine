# SiT (Scalable Interpolant Transformer) Training

## Overview

SiT is a diffusion transformer for image generation using interpolant framework, supporting ImageNet-1K training with Classifier-Free Guidance (CFG).

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

**Highlights**: Interpolant Transformer, CFG, ImageNet-1K

## Quick Start

See the example configuration and run script:
- **Example Config**: [examples/scalable_interpolant_transformer/sit_xl_2.yaml](../../examples/scalable_interpolant_transformer/sit_xl_2.yaml)
- **Run Script**: [examples/scalable_interpolant_transformer/run.sh](../../examples/scalable_interpolant_transformer/run.sh)
- **Documentation**: [examples/scalable_interpolant_transformer/README.md](../../examples/scalable_interpolant_transformer/README.md)

## Model Variants

| Model | Parameters | Hidden Size | Depth | Heads |
|-------|-----------|-------------|-------|-------|
| SiT-S/2 | ~33M | 384 | 12 | 6 |
| SiT-B/2 | ~130M | 768 | 12 | 12 |
| SiT-L/2 | ~458M | 1024 | 24 | 16 |
| SiT-XL/2 | ~675M | 1152 | 28 | 16 |

## Key Configuration

```yaml
model_config:
  load_from_config:
    model_type: "sit"
    hidden_size: 1152      # XL model
    depth: 28              # XL model
    num_heads: 16
    vae_path: "stabilityai/sd-vae-ft-ema"
    path_type: "Linear"
    prediction: "velocity"
    cfg_scale: 1.0

trainer_args:
  bf16: true
  fsdp2: true
```

## Features

- **Interpolant Paths**: Linear, GVP, VP
- **EMA**: Exponential Moving Average for stable generation
- **CFG**: Classifier-Free Guidance support
- **VAE**: Stable Diffusion VAE for latent space encoding

