# RAE-SigLip Training

## Overview

RAE (Representation AutoEncoder) with SigLip is a vision representation learning model with LPIPS loss and EMA support.

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

**Highlights**: Representation AutoEncoder, LPIPS loss, EMA

## Quick Start

See the example run script:
- **Run Script**: [examples/representation_autoencoder/run.sh](../../examples/representation_autoencoder/run.sh)
- **Reconstruction Script**: [examples/representation_autoencoder/reconstruct.py](../../examples/representation_autoencoder/reconstruct.py)

## Key Features

- **LPIPS Loss**: Perceptual loss for better visual quality
- **EMA**: Exponential Moving Average for stable representations
- **SigLip Encoder**: Strong vision encoder backbone

## Usage

```bash
# Training
bash examples/representation_autoencoder/run.sh

# Reconstruction
python examples/representation_autoencoder/reconstruct.py
```

