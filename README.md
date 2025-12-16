<div align="center">

<img width="2816" height="1157" alt="Gemini_Generated_Image_s42giss42giss42g" src="https://github.com/user-attachments/assets/3c727b3d-4710-4f92-af95-c8e29dc6ac3e" />

<h4>A simple, unified multimodal models training engine. Lean, flexible, and built for hacking at scale.</h4>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint](https://github.com/EvolvingLMMs-Lab/lmms-engine/actions/workflows/lint.yml/badge.svg)](https://github.com/EvolvingLMMs-Lab/lmms-engine/actions/workflows/lint.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/EvolvingLMMs-Lab/lmms-engine?style=social)](https://github.com/LMMs-Lab/lmms-engine/stargazers)

[Quick Start](#-quick-start) • [Examples](#-featured-examples) • [Model Support](#-model-support) • [Optimizations](#️-optimizations) • [Codebase Architecture](#️-codebase-architecture) • [Documentation](#-documentation)

</div>

---

## Annoucement

- [2025-10] 🎉🎉 **Efficiency Report**: We provide comprehensive Model FLOPs Utilization (MFU) metrics for various model architectures and training configurations. See [MFU Reference](docs/reference/mfu.md) for detailed benchmarks.
- [2025-10] 🚀🚀 **LMMs-Engine v0.1** is here! a lean, efficient framework built to train unified multimodal model at scale.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LMMs-Lab/lmms-engine.git
cd lmms-engine

# Install editable packages
uv pip install -e ".[all]"
# or install as a packages
uv pip install -e .
# Install a stable release
uv pip install lmms-engine

# Install dependencies using uv sync
# For Linux systems (recommended - auto-detects platform):
bash uv_sync_linux.sh

# For other systems or if encountering errors:
uv sync
# If uv sync fails, try: 
uv pip install -r requirements.txt

# Optional: Performance optimizations
uv pip install flash-attn --no-build-isolation
uv pip install liger-kernel
```

### Docker
We provide [Docker images](https://hub.docker.com/r/fatbao55/lmms-engine/tags) with pre-built environments including PyTorch, CUDA, and all necessary dependencies.

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \  
  fatbao55/lmms-engine:v1.0 \
  bash
```

### Launch Training

**Recommended: torchrun (native PyTorch)**
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=12355 \
  -m lmms_engine.launch.cli config_yaml=examples/qwen3_vl/example_config.yaml
```

**Alternative: Accelerate**
```bash
accelerate launch --use_fsdp \
  -m lmms_engine.launch.cli config_yaml=examples/qwen3_vl/example_config.yaml
```

**Single GPU**
```bash
python -m lmms_engine.launch.cli config_yaml=examples/qwen3_vl/example_config.yaml
```

## 🔥 Featured Examples

| Model | Quick Start | FSDP2 | USP | Muon | Liger | Packing | NSA | EP | Highlights |
|-------|-------------|-------|-----|------|-------|---------|-----|----|------------------|
| **[BAGEL](src/lmms_engine/models/bagel)** | [run.sh](examples/bagel/run.sh) | ✅ | TBD | ✅ | ❌ | ✅ | ✅ | ❌ | Unified visual understanding & generation |
| **[Qwen2.5](src/lmms_engine/models/qwen2)** | [run.sh](examples/qwen2_5_llm/run.sh) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | Large Language Model |
| **[Qwen2.5-VL](src/lmms_engine/models/qwen2_5_vl/)** | [run.sh](examples/qwen2_5_vl/run.sh) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | Multimodal Model |
| **[Qwen2.5-Omni](examples/qwen2_5_omni)** | [run.sh](examples/qwen2_5_omni/run.sh) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | Unified multimodal (image, audio, text) |
| **[Qwen3-VL](examples/qwen3_vl)** | [run.sh](examples/qwen3_vl/run.sh) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | Native-resolution, long context (10K+ tokens) |
| **[Qwen3-VL MoE](examples/qwen3_vl_moe)** | [run.sh](examples/qwen3_vl_moe/run.sh) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | Vision-Language MoE with EP (image, video, text) |
| **[Qwen3-MoE](examples/qwen3_moe)** | [run.sh](examples/qwen3_moe/run.sh) | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | Mixture-of-Experts, Expert Parallelism |
| **[Qwen3-Omni MoE](examples/qwen3_omni_moe)** | [config](examples/qwen3_omni_moe_ep2.yaml) | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | Multimodal MoE with EP (image, audio, text) |
| **[WanVideo](examples/wanvideo)** | [run.sh](examples/wanvideo/run.sh) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | T2V/I2V/V2V generation (1.3B/14B) |
| **[FLA models](examples/dgn)** | [run.sh](examples/dgn/run.sh) | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | Efficient architecture, FineWeb-Edu pretraining |
| **[dLLM (Qwen3)](examples/diffusion_language_model)** | [run.sh](examples/diffusion_language_model/run.sh) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | Masked diffusion language model |
| **[RAE-SigLip](examples/representation_autoencoder)** | [run.sh](examples/representation_autoencoder/run.sh) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | Representation AutoEncoder, LPIPS, EMA |
| **[SiT](examples/scalable_interpolant_transformer)** | [run.sh](examples/scalable_interpolant_transformer/run.sh) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | Interpolant Transformer, CFG, ImageNet-1K |

**Optimization Legend:**
- **FSDP2**: Fully Sharded Data Parallel v2 for distributed training
- **USP**: Ulysses Sequence Parallel for long contexts
- **Muon**: Advanced optimizer with Newton-Schulz orthogonalization
- **Liger**: Triton fused kernels (CrossEntropy, RMSNorm, RoPE, SwiGLU) for 30% memory reduction
- **Packing**: First-fit bin packing for peaking at 35-40% MFU vs 20-25% (w/o in Qwen2.5-VL finetuning)
- **NSA**: Native Sparse Attention for efficient long-context processing
- **EP**: Expert Parallelism for Mixture-of-Experts models, sharding experts across GPUs

> 💡 **Tip:** Each `run.sh` file contains detailed setup instructions, prerequisites, and configuration options.

## 🤖 Model Support

**20+ architectures spanning vision-language, diffusion, and language models.**

### Multimodal Models
- **Qwen2.5-VL** - SOTA level performance vision-language model
- **Qwen3-VL** - SOTA level performance vision-language model
- **Qwen3-VL MoE** - Vision-Language Mixture-of-Experts with Expert Parallelism and Sequence Parallelism support
- **Qwen2.5-Omni** - Unified vision + audio + text modalities
- **Qwen3-Omni MoE** - Multimodal Mixture-of-Experts with vision + audio + text and Expert Parallelism support
- **LLaVA-OneVision** - Fully open-source vision-language model
- **Bagel** - Unified multimodal model for visual understanding and generation
- **Aero** - Lightweight audio-language model

### Diffusion & Generative Models
- **dLLM (Qwen3)** - Diffusion Language Model with masked prediction
- **WanVideo (1.3B/14B)** - Text/Image-to-Video generation (T2V/I2V/V2V)
- **SiT (XL/2)** - Scalable Interpolant Transformers for class-conditional image generation
- **RAE-SigLip** - Representation AutoEncoder with adversarial discriminator

### Language Models
- **Qwen2/2.5/3 series** - Full Liger kernel support with fused operations
- **Linear Attention Models** - Recurrent architecture optimized for Muon; Please install [FLA](https://github.com/fla-org/flash-linear-attention) first.
- **Custom architectures** - Extensible via `@register_model()` decorator

## ⚡️ Optimizations

Production-grade efficiency from distributed training to kernel fusion.

### Core Distributed Training

- **FSDP2** - PyTorch 2.0+ DTensor-based sharding for parameters, gradients, and optimizer states. Improved composability over original FSDP enables flexible parallelism composition.

- **Ulysses Sequence Parallel** - Splits sequence dimension across GPUs for ultra-long contexts. Critical for vision-language models like Qwen3-VL with 10K+ visual tokens.

- **Multi-dimensional Parallelism** - Compose TP x PP × DP meshes for cluster-scale training.

### Memory & Compute Optimizations

- **Flash Attention + Unpadding** - Tiled attention with `use_rmpad` eliminates all padding computation.

- **Native Sparse Attention (NSA)** - Hybrid attention mechanism combining compressed attention, topk sparse attention, and sliding window attention.

- **Liger Kernel** - Triton fused kernels (CrossEntropy, RMSNorm, RoPE, SwiGLU) achieve memory reduction by avoiding intermediate materializations.

- **Monkey Patching System** - Runtime kernel injection via `lmms_engine/configs/monkey_patch/` for model-specific optimizations without code modification.

- **Sequence Packing** - Faster first-fit bin packing.

### Advanced Optimizer

- **Muon Optimizer** - Newton-Schulz orthogonalization with Triton kernels, distributed via DTensor. Selective 2D-parameter application outperforms AdamW convergence.

### Data Pipeline

- **Streaming Datasets** - `IterableDataset` for trillion-token pretraining without full data loading.

### Configuration Examples

<details>
<summary><b>Sequence Packing</b> - with full unpadding</summary>

```yaml
dataset_config:
  packing: true
  packing_strategy: first_fit
  packing_length: 32000

trainer_args:
  use_rmpad: true  # Requires flash-attn
  use_liger_kernel: true
```
</details>

<details>
<summary><b>Liger Kernel</b> - Enable LinkedIn's Triton kernels for 30% memory reduction</summary>

```yaml
trainer_args:
  use_liger_kernel: true
```

**Fused operations:**
- CrossEntropy (major memory savings)
- RMSNorm, RoPE, SwiGLU
- Automatically applied via monkey patching
</details>

<details>
<summary><b>Muon Optimizer</b> - State-of-the-art optimizer for LLMs</summary>

```yaml
trainer_args:
  use_muon: true # enable muonwithadam optimizer
  adam_beta1: 0.9 # for the adam part in muonwithadam optimizer
  adam_beta2: 0.999 # for the adam part in muonwithadam optimizer
  adam_epsilon: 1.0e-8 # for the adam part in muonwithadam optimizer
  learning_rate: 0.001
  weight_decay: 0.01
  # ns_steps: 5  # Newton-Schulz iterations (default)

  # for some modules which the user hope to 
```

**Features:**
- Newton-Schulz orthogonalization with Triton kernels
- Distributed via DTensor (FSDP2)
- Selective 2D parameter application

**Note**
If users wish to specify whether a module should be optimized using Muon or Adam, they can designate this in `lmms_engine.train.hf.trainer.create_optimizer`. By default, modules excluded from Muon optimization include those containing the following substrings in their names: `["emb", "norm", "lm_head", "bias", "wte", "wpe", "output", "a_proj", "b_proj", "conv1d", "rotary"]`
as well as any parameters whose dimension does not equal 2.

</details>

<details>
<summary><b>FSDP2 Configuration</b></summary>

```yaml
trainer_args:
  fsdp2: true
  fsdp_config:
    transformer_layer_cls_to_wrap: ["Qwen2VLDecoderLayer"]
    reshard_after_forward: false
    activation_checkpointing: true
```
</details>

<details>
<summary><b>Ulysses Sequence Parallel</b> - For long-sequence VLMs</summary>

```yaml
trainer_args:
  sp_ulysses_degree: 2  # Sequence parallel degree
```

**Benefits:**
- Splits sequence length across GPUs
- Reduces memory footprint for long contexts
- Works with Flash Attention
</details>

<details>
<summary><b>Native Sparse Attention (NSA)</b> - Efficient long-context attention for BAGEL</summary>

```yaml
model_config:
  load_from_pretrained_path: "lmms-lab/BAGEL-7B-MoT-ver.LE"

monkey_patch:
  - type: nsa
    model_type: bagel
    kwargs:
      block_size: 64
      compress_type: "weightedpool"  # weightedpool, linear, avgpool
      kernel_size: 32
      kernel_stride: 16
      topk: 16
      init_blocks: 1
      local_blocks: 2
      window_size: 512
```

**Features:**
- Compressed attention with key-value compression
- TopK sparse attention for efficiency
- Sliding window attention for local context
- Hybrid mechanism combines all three attention types
- Requires: `pip install git+https://github.com/XunhaoLai/native-sparse-attention-triton.git`

**Note:** Currently only supported for BAGEL model.
</details>

## 📖 Documentation

### Step-by-Step Workflow

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

3. **Configure training** - See [examples/qwen3_vl/example_config.yaml](examples/qwen3_vl/example_config.yaml) or any model-specific config in [examples/](examples/)

### Comprehensive Guides

**Getting Started:**
- [Dataset Preparation](docs/user_guide/data_prep.md) - How to prepare and structure your data
- [Dataset & Packing Guide](docs/user_guide/datasets.md) - Detailed dataset implementations and packing strategies
- [Training Guide](docs/getting_started/train.md) - Comprehensive training walkthrough

**Advanced Topics:**
- [Design Principles](docs/reference/design_principle.md) - Architectural patterns and philosophy
- [API Reference](docs/reference/api.md) - Detailed API documentation

## 🏗️ Codebase Architecture

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

## 🎯 Use Cases

- **Vision-Language Pretraining** - Qwen-VL, LLaVA on large multimodal datasets
- **Video Understanding** - AERO on 3D video data
- **Diffusion Models** - DLLM, SiT, WanVideo for generation tasks
- **Representation Learning** - RAE for visual representations
- **Language Model Pretraining** - DGN, Qwen with Muon optimizer
- **Multimodal Fine-tuning** - Efficient SFT with sequence packing

## 🤝 Contributing

We welcome contributions! Please see our [Design Principles](docs/reference/design_principle.md) for coding guidelines:

- **Simplicity**: Write simple, straightforward code
- **Readability**: Prioritize clarity over cleverness
- **Testability**: Create testable components
- **Minimal Changes**: Only modify code related to the task
- **Less Code = Less Debt**: Minimize code footprint

## 😊 Acknowledgement

Thanks to the following projects for their excellent work:

- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [nanotron](https://github.com/huggingface/nanotron)
- [veScale](https://github.com/volcengine/veScale)
- [veOmni](https://github.com/ByteDance-Seed/VeOmni)

## 📝 Citation

If you use LMMs Engine in your research, please cite:

```bibtex
@software{lmms_engine2025,
  title={LMMs Engine: A simple, unified multimodal framework for pretraining and finetuning.},
  author={LMMs-Lab},
  year={2025},
  url={https://github.com/LMMs-Lab/lmms-engine}
}
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **GitHub**: https://github.com/EvolvingLMMs-Lab/lmms-engine
- **LMMs-Lab**: https://lmms-lab.com
- **Documentation**: [docs/](docs/)
- **Issues**: https://github.com/EvolvingLMMs-Lab/lmms-engine/issues

## 🎉 Awesome projects using LMMs-Engine

- [LongVT](https://github.com/EvolvingLMMs-Lab/LongVT): Incentivizing "Thinking with Long Videos" via Native Tool Calling ![GitHub Repo stars](https://img.shields.io/github/stars/EvolvingLMMs-Lab/LongVT)

- [OpenMMReasoner](https://github.com/EvolvingLMMs-Lab/OpenMMReasoner): Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe![GitHub Repo stars](https://img.shields.io/github/stars/EvolvingLMMs-Lab/OpenMMReasoner)

---

<div align="center">

**Built with ❤️ by [LMMs-Lab](https://lmms-lab.com/)**

⭐ **Star us on GitHub to support the project!** ⭐

</div>
