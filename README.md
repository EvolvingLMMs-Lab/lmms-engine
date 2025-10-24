# LMMs Engine

A simple, unified multimodal models training engine. Lean, flexible, and built for hacking, and hacking at scale.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint](https://github.com/EvolvingLMMs-Lab/lmms-engine/actions/workflows/lint.yml/badge.svg)](https://github.com/EvolvingLMMs-Lab/lmms-engine/actions/workflows/lint.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/EvolvingLMMs-Lab/lmms-engine?style=social)](https://github.com/LMMs-Lab/lmms-engine/stargazers)

<div align="center">

[Features](#-key-features) • [Quick Start](#-quick-start) • [Examples](#-featured-examples) • [Documentation](#-documentation) • [Architecture](#-modular-architecture)

</div>

---

## Overview

**LMMs Engine** is a highly efficient, modular training framework for training Unified Multimodal Models at scale. It provides:

- **19+ model architectures** including vision-language models (Qwen2/3-VL, LLaVA-OV), diffusion models (dLLM, SiT, WanVideo), and specialized architectures for research purposes (RAE)
- **Scalability Optimizations** with FSDP2, Ulysses Sequence Parallel, Flash Attention 2, Liger Kernel, and state-of-the-art Muon optimizer for training large models on thousands of GPUs.
- **Modular design** using Factory and Builder patterns for easy extensibility to support many types of models and datasets.

## 🎯 Key Features

### 1. Modular Architecture

> TODO: KAICHEN

### 2. State-of-the-Art Optimizations

Production-grade efficiency from distributed training to kernel fusion.

#### Native PyTorch Distributed Training

- **FSDP2** - PyTorch 2.0+ DTensor-based sharding for parameters, gradients, and optimizer states. Improved composability over original FSDP enables flexible parallelism composition.

- **Ulysses Sequence Parallel** - Splits sequence dimension across GPUs for ultra-long contexts. Critical for vision-language models like Qwen3-VL with 10K+ visual tokens.

- **Multi-dimensional Parallelism** - Compose TP × CP × PP × DP meshes for cluster-scale training.

#### Kernel Fusion & Memory Efficiency

- **Flash Attention + Unpadding** - Tiled attention with `use_rmpad` eliminates all padding computation. 2-3× speedup on variable-length sequences.

- **Liger Kernel** - Triton fused kernels (CrossEntropy, RMSNorm, RoPE, SwiGLU) achieve 30% memory reduction by avoiding intermediate materializations.

- **Monkey Patching System** - Runtime kernel injection via `lmms_engine/configs/monkey_patch/` for model-specific optimizations without code modification.

#### Advanced Optimizers & Data Efficiency

- **Muon Optimizer** - Newton-Schulz orthogonalization with Triton kernels, distributed via DTensor. Selective 2D-parameter application outperforms AdamW convergence.

- **Sequence Packing** - First-fit bin packing achieves 35-40% MFU vs 20-25% without packing. Combined with unpadding for zero padding waste.

- **Streaming Datasets** - `IterableDataset` for trillion-token pretraining without full data loading.

### 3. Comprehensive Model Support

**19+ architectures spanning vision-language, diffusion, and language models.**

#### Vision-Language Models
- **Qwen2.5-VL** - Multi-resolution vision-language model
- **Qwen3-VL** - Latest Qwen vision-language with Ulysses SP
- **Qwen2.5-Omni** - Unified vision + audio + text modalities
- **LLaVA-OneVision** - Multi-resolution visual understanding
- **Bagel** - Vision-language with NSA operations
- **AERO** - 3D-aware video understanding

#### Diffusion & Generative Models
- **DLLM (Qwen3)** - Diffusion Language Model with masked prediction
- **WanVideo (1.3B/14B)** - Text/Image-to-Video generation (T2V/I2V/V2V)
- **SiT (XL/2)** - Scalable Interpolant Transformers for class-conditional image generation
- **RAE-SigLip** - Representation AutoEncoder with adversarial discriminator

#### Language Models
- **Qwen2/2.5/3 series** - Full Liger kernel support with fused operations
- **Gated DeltaNet (DGN)** - Recurrent architecture optimized for Muon
- **Custom architectures** - Extensible via `@register_model()` decorator

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LMMs-Lab/lmms-engine.git
cd lmms-engine

# Install dependencies
uv sync

# Optional: Performance optimizations
uv pip install flash-attn --no-build-isolation
uv pip install liger-kernel
```

### Launch Training

**Recommended: torchrun (native PyTorch)**
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=12355 \
  -m lmms_engine.launch.cli --config examples/qwen3_vl/example_config.yaml
```

**Alternative: Accelerate**
```bash
accelerate launch --use_fsdp \
  -m lmms_engine.launch.cli --config examples/qwen3_vl/example_config.yaml
```

**Single GPU**
```bash
python -m lmms_engine.launch.cli --config examples/qwen3_vl/example_config.yaml
```

## 🔥 Featured Examples

| Model | Architecture | FSDP2 | Ulysses SP | Muon | Packing | Highlights | Quick Start |
|-------|-------------|-------|------------|------|---------|------------------|-------------|
| **[Qwen3-VL](examples/qwen3_vl)** | Vision-Language | ✅ | ✅ | ✅ | ✅ | Native-resolution, long context (10K+ tokens) | [run.sh](examples/qwen3_vl/run.sh) |
| **[Qwen2.5-Omni](examples/qwen2_5_omni)** | Vision+Audio+Text | ✅ | ✅ | ✅ | ✅ | Unified multimodal (image, audio, text) | [run.sh](examples/qwen2_5_omni/run.sh) |
| **[dLLM (Qwen3)](examples/diffusion_language_model)** | Diffusion LM | ✅ | ❌ | ✅ | ❌ | Masked diffusion | [run.sh](examples/diffusion_language_model/run.sh) |
| **[Gated DeltaNet](examples/dgn)** | Gated Linear Attn | ✅ | ❌ | ✅ | ✅ | Linear Attention Arch, FineWeb-Edu pretraining | [run.sh](examples/dgn/run.sh) |
| **[WanVideo](examples/wanvideo)** | Video Generation | ✅ | ❌ | ❌ | ❌ | T2V/I2V/V2V generation (1.3B/14B) | [run.sh](examples/wanvideo/run.sh) |
| **[SiT](examples/scalable_interpolant_transformer)** | Diffusion Transformer | ✅ | ❌ | ❌ | ❌ | Interpolant Transformer, CFG, ImageNet-1K | [run.sh](examples/scalable_interpolant_transformer/run.sh) |
| **[RAE-SigLip](examples/representation_autoencoder)** | Visual AutoEncoder | ✅ | ❌ | ❌ | ❌ | Representation AutoEncoder, LPIPS, EMA | [run.sh](examples/representation_autoencoder/run.sh) |

**Optimization Legend:**
- **FSDP2**: Fully Sharded Data Parallel v2 for distributed training
- **Ulysses SP**: Sequence Parallel for long contexts (10K+ visual tokens)
- **Muon**: Advanced optimizer with Newton-Schulz orthogonalization
- **Packing**: First-fit bin packing for 35-40% MFU vs 20-25% without

> 💡 **Tip:** Each `run.sh` file contains detailed setup instructions, prerequisites, and configuration options.

## 📖 Documentation

### Getting Started
- [Dataset Preparation](docs/data_prep.md) - How to prepare and structure your data
- [Dataset & Packing Guide](docs/datasets.md) - Detailed dataset implementations and packing strategies
- [Training Guide](docs/train.md) - Comprehensive training walkthrough

### Advanced Topics
- [Design Principles](docs/design_principle.md) - Architectural patterns and philosophy
- [API Reference](docs/api.md) - Detailed API documentation

### Step-by-Step Guides
1. **Process the dataset** into OpenAI chat format (JSONL/JSON/Arrow/CSV)
   ```bash
   hf download kcz358/open-thoughts-debug --local-dir data/open_thoughts_debug --repo-type dataset
   ```

2. **Prepare dataset YAML** (optional for single data source)
   ```yaml
   datasets:
     - path: data/open_thoughts_debug
       data_folder: ""
       data_type: arrow
   ```

3. **Configure training** - See [examples/config_example.yaml](examples/config_example.yaml)

## 🏗️ Modular Architecture

### Component Registry

**Factory Pattern** enables easy extensibility:

```python
# Register a custom dataset
from lmms_engine.datasets import register_dataset, BaseDataset

@register_dataset("my_custom_dataset")
class MyCustomDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        # Custom initialization

    def __getitem__(self, idx):
        # Custom data loading
        return item

# Register a custom processor
from lmms_engine.datasets.processor import register_processor

@register_processor("my_custom_processor")
class MyCustomProcessor:
    def __call__(self, raw_data):
        # Custom processing
        return processed_data
```

### Training Pipeline

**Builder Pattern** for flexible composition:

```python
from lmms_engine.train import TrainRunner

# Configuration defines the pipeline
runner = TrainRunner(config)
runner.build()  # Lazy initialization of components
runner.run()    # Execute training
```

**Pipeline stages:**
1. **Model initialization** - From pretrained or config
2. **Dataset creation** - With processor and collator
3. **Monkey patching** - Apply kernel optimizations
4. **Trainer setup** - FSDP2, DeepSpeed, or custom
5. **Training execution** - With checkpointing and logging

### Supported Trainers

| Trainer Type | Use Case | Key Features |
|-------------|----------|--------------|
| `hf_trainer` | General VLM/LM training | FSDP2, Muon, Liger, Flash Attn |
| `dllm_trainer` | Diffusion language models | Masked LM, custom loss, DLLM collator |
| `wan_trainer` | Video generation | Flow-matching, multi-modal inputs |
| `rae_trainer` | Visual autoencoders | Adversarial loss, EMA, LPIPS |
| `sit_trainer` | Diffusion transformers | Interpolant framework, CFG, EMA |

## ⚙️ Advanced Optimizations

### Sequence Packing

Achieve 35-40% MFU with full unpadding:

```yaml
dataset_config:
  packing: true
  packing_strategy: first_fit
  packing_length: 32000

trainer_args:
  use_rmpad: true  # Requires flash-attn
  use_liger_kernel: true
```

### Liger Kernel

Enable LinkedIn's Triton kernels for 30% memory reduction:

```yaml
trainer_args:
  use_liger_kernel: true
```

**Fused operations:**
- CrossEntropy (major memory savings)
- RMSNorm, RoPE, SwiGLU
- Automatically applied via monkey patching

### Muon Optimizer

State-of-the-art optimizer for LLMs:

```yaml
trainer_args:
  use_muon: true
  learning_rate: 0.001
  adam_beta1: 0.9
  adam_beta2: 0.999
  weight_decay: 0.01
  # ns_steps: 5  # Newton-Schulz iterations (default)
```

**Features:**
- Newton-Schulz orthogonalization with Triton kernels
- Distributed via DTensor (FSDP2)
- Selective 2D parameter application
- Superior convergence vs AdamW

### FSDP2 Configuration

```yaml
trainer_args:
  fsdp2: true
  fsdp_config:
    transformer_layer_cls_to_wrap: ["Qwen2VLDecoderLayer"]
    reshard_after_forward: false
    activation_checkpointing: true
```

### Ulysses Sequence Parallel

For long-sequence VLMs:

```yaml
trainer_args:
  sp_ulysses_degree: 2  # Sequence parallel degree
```

**Benefits:**
- Splits sequence length across GPUs
- Reduces memory footprint for long contexts
- Works with Flash Attention

## 🎯 Use Cases

- **Vision-Language Pretraining** - Qwen-VL, LLaVA on large multimodal datasets
- **Video Understanding** - AERO on 3D video data
- **Diffusion Models** - DLLM, SiT, WanVideo for generation tasks
- **Representation Learning** - RAE for visual representations
- **Language Model Pretraining** - DGN, Qwen with Muon optimizer
- **Multimodal Fine-tuning** - Efficient SFT with sequence packing

## 🤝 Contributing

We welcome contributions! Please see our [Design Principles](docs/design_principle.md) for coding guidelines:

- **Simplicity**: Write simple, straightforward code
- **Readability**: Prioritize clarity over cleverness
- **Testability**: Create testable components
- **Minimal Changes**: Only modify code related to the task
- **Less Code = Less Debt**: Minimize code footprint

## 📝 Citation

If you use LMMs Engine in your research, please cite:

```bibtex
@software{lmms_engine2024,
  title={LMMs Engine: A simple, unified multimodal framework for pretraining and finetuning.},
  author={LMMs-Lab},
  year={2024},
  url={https://github.com/LMMs-Lab/lmms-engine}
}
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **GitHub**: https://github.com/LMMs-Lab/lmms-engine
- **LMMs-Lab**: https://lmms-lab.com
- **Documentation**: [docs/](docs/)
- **Issues**: https://github.com/LMMs-Lab/lmms-engine/issues

---

<div align="center">

**Built with ❤️ by [LMMs-Lab](https://lmms-lab.com/)**

⭐ **Star us on GitHub to support the project!** ⭐

</div>
