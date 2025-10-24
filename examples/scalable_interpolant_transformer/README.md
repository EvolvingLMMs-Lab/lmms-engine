# SiT (Scalable Interpolant Transformers) Training

**Paper:** [Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://arxiv.org/abs/2401.08740) (ECCV 2024)
**Official Repo:** https://github.com/willisma/SiT

## Overview

SiT is a family of generative models built on Diffusion Transformers (DiT) that uses an **interpolant framework** to flexibly connect two distributions.

Key features:
- Transformer-based architecture with VAE latent space encoding
- Flexible interpolant paths (Linear, GVP, VP)
- Multiple prediction targets (velocity, noise, score)
- Classifier-Free Guidance (CFG) support
- Exponential Moving Average (EMA) for stable generation

## Quick Start

### 1. Install Dependencies

```bash
# Install SiT dependencies
uv pip install -e ".[sit]"

# Optional performance optimizations
uv pip install flash-attn --no-build-isolation
uv pip install liger-kernel
```

### 2. Prepare Dataset

Your dataset should contain:
- `image`: Image data (PIL Image or tensor)
- `label`: Class label (integer, e.g., 0-999 for ImageNet)

Supported formats: Arrow, Parquet, HuggingFace Dataset

### 3. Configure Training

Edit `sit_xl_2.yaml` and set your dataset path:

```yaml
dataset_config:
  dataset_type: "sit"
  dataset_format: "hf_dataset"
  dataset_path: ILSVRC/imagenet-1k
```

### 4. Set Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"  # Required for VAE download
export HF_HOME="$HOME/.cache/huggingface"  # Optional: cache directory
```

### 5. Launch Training

**Single Node (8 GPUs):**
```bash
bash examples/scalable_interpolant_transformer/run.sh
```

**Custom GPU Count:**
```bash
NPROC_PER_NODE=4 bash examples/scalable_interpolant_transformer/run.sh
```

**Multi-Node Training (Master Node):**
```bash
NNODES=4 NODE_RANK=0 MASTER_ADDR=192.168.1.100 bash examples/scalable_interpolant_transformer/run.sh
```

**Multi-Node Training (Worker Nodes):**
```bash
NNODES=4 NODE_RANK=1 MASTER_ADDR=192.168.1.100 bash examples/scalable_interpolant_transformer/run.sh
```

## Model Variants

The default configuration uses **SiT-XL/2** (~675M parameters). To use other variants, modify the model configuration in `sit_xl_2.yaml`:

| Model | hidden_size | depth | num_heads | patch_size | Params | GFLOPs |
|-------|-------------|-------|-----------|------------|--------|--------|
| SiT-S/2 | 384 | 12 | 6 | 2 | ~33M | ~4 |
| SiT-B/2 | 768 | 12 | 12 | 2 | ~130M | ~16 |
| SiT-L/2 | 1024 | 24 | 16 | 2 | ~458M | ~80 |
| SiT-XL/2 | 1152 | 28 | 16 | 2 | ~675M | ~119 |

### Example Configurations

<details>
<summary><b>SiT-S/2 Configuration</b></summary>

```yaml
model_config:
  load_from_config:
    model_type: "sit"
    input_size: 32
    patch_size: 2
    in_channels: 4
    hidden_size: 384       # S model
    depth: 12              # S model
    num_heads: 6           # S model
    mlp_ratio: 4.0
    class_dropout_prob: 0.1
    num_classes: 1000
    learn_sigma: true
    vae_path: "stabilityai/sd-vae-ft-ema"
    path_type: "Linear"
    prediction: "velocity"
    cfg_scale: 1.0
```
</details>

<details>
<summary><b>SiT-B/2 Configuration</b></summary>

```yaml
model_config:
  load_from_config:
    model_type: "sit"
    input_size: 32
    patch_size: 2
    in_channels: 4
    hidden_size: 768       # B model
    depth: 12              # B model
    num_heads: 12          # B model
    mlp_ratio: 4.0
    class_dropout_prob: 0.1
    num_classes: 1000
    learn_sigma: true
    vae_path: "stabilityai/sd-vae-ft-ema"
    path_type: "Linear"
    prediction: "velocity"
    cfg_scale: 1.0
```
</details>

<details>
<summary><b>SiT-L/2 Configuration</b></summary>

```yaml
model_config:
  load_from_config:
    model_type: "sit"
    input_size: 32
    patch_size: 2
    in_channels: 4
    hidden_size: 1024      # L model
    depth: 24              # L model
    num_heads: 16          # L model
    mlp_ratio: 4.0
    class_dropout_prob: 0.1
    num_classes: 1000
    learn_sigma: true
    vae_path: "stabilityai/sd-vae-ft-ema"
    path_type: "Linear"
    prediction: "velocity"
    cfg_scale: 1.0
```
</details>

## Batch Size Recommendations

Adjust `per_device_train_batch_size` based on your GPU memory:

| GPU | SiT-S/2 | SiT-B/2 | SiT-L/2 | SiT-XL/2 |
|-----|---------|---------|---------|----------|
| A100 40GB | 32 | 16 | 8 | 4 |
| A100 80GB | 64 | 32 | 16 | 8 |
| H100 80GB | 128 | 64 | 32 | 16 |

If you encounter OOM errors, reduce batch size and increase `gradient_accumulation_steps` proportionally to maintain effective batch size.

## Key Configuration Parameters

### Model Architecture
- **`input_size`**: Latent space size (image_size = input_size × 8, default: 32 → 256px)
- **`patch_size`**: Patch size for tokenization (2, 4, or 8). Smaller = finer detail but more compute
- **`hidden_size`**: Transformer hidden dimension (384/768/1024/1152 for S/B/L/XL)
- **`depth`**: Number of transformer blocks (12/12/24/28 for S/B/L/XL)
- **`num_heads`**: Number of attention heads
- **`mlp_ratio`**: MLP expansion ratio (default: 4.0)

### Interpolant Configuration
- **`path_type`**: Interpolant path type
  - `Linear`: Linear interpolation (simplest, recommended, used in paper)
  - `GVP`: Geodesic Variance Preserving
  - `VP`: Variance Preserving
- **`prediction`**: Model prediction target
  - `velocity`: Velocity field (recommended, default)
  - `noise`: Noise prediction
  - `score`: Score function
- **`loss_weight`**: Optional loss weighting scheme
- **`train_eps`**: Training epsilon for numerical stability (optional)
- **`sample_eps`**: Sampling epsilon for numerical stability (optional)

### Conditional Generation
- **`num_classes`**: Number of classes (1000 for ImageNet-1K)
- **`class_dropout_prob`**: Class dropout probability for CFG training (default: 0.1)
- **`cfg_scale`**: Classifier-Free Guidance scale during inference (>1.0 enables CFG, default: 1.0)

### VAE Configuration
- **`vae_path`**: Pre-trained VAE model (default: "stabilityai/sd-vae-ft-ema")
- Images are encoded to 4-channel latent space (32×32 for 256×256 images)

## Training Features

1. **EMA Model**: Automatically maintains exponential moving average model (decay=0.9999) for improved sample quality
2. **VAE Encoding**: Images are encoded to 4-channel latent space using Stable Diffusion VAE
3. **CFG Support**: Classifier-Free Guidance with configurable dropout and scale
4. **FSDP2**: Fully Sharded Data Parallel for large-scale distributed training
5. **Transport Framework**: Flexible interpolant paths and prediction targets

### Checkpoint Structure

Checkpoints are saved to `{output_dir}/checkpoint-{step}/`:
```
output/sit_xl_2_training/
├── checkpoint-1000/
│   ├── model.safetensors      # Main model weights
│   ├── ema.pt                 # EMA model weights
│   └── trainer_state.json     # Training state
├── checkpoint-2000/
└── ...
```

Both main and EMA models are saved for proper resumption.

## Troubleshooting

### Out of Memory (OOM)
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use a smaller model variant (e.g., SiT-L/2, SiT-B/2, or SiT-S/2)
- Use larger `patch_size` (4 or 8 instead of 2)

### Slow Training
- Increase `dataloader_num_workers` (default: 4, try 8 or 16)
- Enable `tf32=true` for A100+ GPUs (faster compute)
- Use `bf16=true` for mixed precision training
- Consider using larger `patch_size` for faster training (tradeoff: lower quality)

### Missing Dependencies
```bash
# Install all SiT dependencies
pip install lmms_engine[sit]

# Or install individually
pip install timm diffusers torchdiffeq
```

### Import Errors
If you see `"Install with: pip install lmms_engine[sit]"` error, the SiT optional dependencies are missing. Run:
```bash
uv pip install -e ".[sit]"
```

## Performance Benchmarks

Results from the paper on ImageNet 256×256:

| Model | FID-50K ↓ | Inception Score ↑ | Precision ↑ | Recall ↑ |
|-------|-----------|-------------------|-------------|----------|
| DiT-XL(cfg = 1.5) | 2.27 | 4.60 | 278.24 | 0.83 | 0.57 |
| SiT-XL(cfg = 1.5, ODE) | **2.15** | **4.60** | **258.09** | **0.81** | **0.60** |
| SiT-XL(cfg = 1.5, SDE) | **2.06** | **4.49** | **277.50** | **0.83** | **0.59** |

## References

- **Paper**: [SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://arxiv.org/abs/2401.08740)
- **Official GitHub**: https://github.com/willisma/SiT
- **Project Website**: https://scalable-interpolant.github.io/
- **VAE**: [Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-ema)
- **Base Architecture**: Built on [Diffusion Transformers (DiT)](https://github.com/facebookresearch/DiT)

## Citation

```bibtex
@inproceedings{ma2024sit,
  title={Scalable Interpolant Transformers},
  author={Ma, Nanye and Goldstein, Mark and Albergo, Michael and Boffi, Nicholas and Vanden-Eijnden, Eric and Xie, Saining},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
