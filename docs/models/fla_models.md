# FLA Models (DGN) Training

## Overview

FLA (Fast Linear Attention) models including Gated DeltaNet (DGN) provide efficient architectures with FineWeb-Edu pretraining support.

## Supported Features

| Feature | Support |
|---------|---------|
| **FSDP2** | ✅ |
| **USP** | ❌ |
| **Muon Optimizer** | ✅ |
| **Liger Kernel** | ❌ |
| **Packing** | ✅ |
| **NSA** | ❌ |
| **Expert Parallelism** | ❌ |

**Highlights**: Efficient architecture, FineWeb-Edu pretraining

## Quick Start

See the example configuration and run script:
- **Example Config**: [examples/dgn/train_dgn_1b.yaml](../../examples/dgn/train_dgn_1b.yaml)
- **Run Script**: [examples/dgn/run.sh](../../examples/dgn/run.sh)

## Key Configuration

```yaml
dataset_config:
  dataset_type: fineweb_edu
  dataset_format: hf_dataset
  dataset_path: HuggingFaceFW/fineweb-edu
  packing_length: 2048

model_config:
  load_from_config:
    model_type: gated_deltanet
    config:
      vocab_size: 151936
      hidden_size: 1024
      intermediate_size: 4096
      num_hidden_layers: 24

trainer_args:
  use_muon: true
  fsdp2: true
  bf16: true
```

## About FLA

FLA models use linear attention mechanisms for improved efficiency on long sequences compared to standard attention.

