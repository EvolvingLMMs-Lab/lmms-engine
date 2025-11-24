# WanVideo Training

## Overview

WanVideo is a diffusion-based video generation model supporting Text-to-Video (T2V), Image-to-Video (I2V), and Video-to-Video (V2V) generation.

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

**Highlights**: T2V/I2V/V2V generation (1.3B/14B)

## Quick Start

See the example configuration and run script:
- **Example Configs**: [examples/wanvideo/configs/](../../examples/wanvideo/configs/)
- **Run Script**: [examples/wanvideo/run.sh](../../examples/wanvideo/run.sh)
- **Documentation**: [examples/wanvideo/README.md](../../examples/wanvideo/README.md)

## Model Variants

- **Wan2.1-T2V-1.3B**: Text-to-Video (480×832)
- **Wan2.1-T2V-14B**: High-quality Text-to-Video (480×832)
- **Wan2.1-I2V-14B**: Image-to-Video (720×1280)

## Key Configuration

```yaml
model_config:
  load_from_config:
    model_type: wanvideo
    model_variant: "Wan2.1-T2V-1.3B"
    dit_enable_flash_attn: true
    gradient_checkpointing: true
    scheduler_type: "flow_match"

trainer_args:
  bf16: true
  tf32: true
  fsdp2: true
```

