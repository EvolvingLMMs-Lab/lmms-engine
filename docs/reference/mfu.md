# Model FLOPs Utilization (MFU) Reference

This document contains MFU metrics for various model training configurations using LMMs-Engine, measured with FSDP distributed training across multiple GPUs. All the experiment are being conducted with 4*8 A800 80G SXM using `fsdp2_trainer` with `vision_iterable` dataset for on-the-fly stream packing.

## Overview

Model FLOPs Utilization (MFU) measures the efficiency of GPU usage during training, representing the ratio of achieved FLOPs to theoretical peak FLOPs. All configurations use:
- **Sequence Packing**: First-fit bin packing strategy for optimal throughput
- **Unpadding**: Remove padding via `use_rmpad` to eliminate wasted computation
- **Liger Kernel**: Fused operations for memory efficiency
- **Iterable Dataset**: Streaming data loading for trillion-token pretraining
- **FSDP**: Fully Sharded Data Parallel v2 for distributed training

## Text Models

### Qwen2.5 7B & Qwen2.5-VL-7B

**Configuration:**

- 4 Node x 8 A800-SXM4 GPU 
- Packing length: 81,920
- Optimization: Remove padding + Liger kernel + Iterable dataset
- Training mode: FSDP distributed

**Achieved MFU: 0.50-0.55 (50-55%)**

**Key settings:**
```yaml
dataset_config:
  packing: true
  packing_strategy: first_fit
  packing_length: 81920

trainer_args:
  use_rmpad: true
  use_liger_kernel: true
  fsdp2: true
```

---

## Image Models (Vision-Language)

### Qwen2.5-VL-7B & Qwen3-VL-8B

**Configuration:**

- 4 Node x 8 A800-SXM4 GPU
- Packing length: 61,440
- Optimization: Remove padding + Liger kernel + Iterable dataset
- Training mode: FSDP distributed

**Achieved MFU: 0.30-0.40 (30-40%)**

**Key settings:**
```yaml
dataset_config:
  packing: true
  packing_strategy: first_fit
  packing_length: 61440

trainer_args:
  use_rmpad: true
  use_liger_kernel: true
  fsdp2: true
```

**Note:** Reduced packing length compared to text models due to Vision Transformer (ViT) overhead. See [Important Considerations](#important-considerations) below.

---

## Video Models (Vision-Language)

### Qwen2.5-VL-7B

**Configuration:**

- 4 Node x 8 A800-SXM4 GPU
- Packing length: 51,200
- Optimization: Remove padding + Liger kernel + Iterable dataset
- Training mode: FSDP distributed

**Achieved MFU: 0.25-0.35 (25-35%)**

**Key settings:**
```yaml
dataset_config:
  packing: true
  packing_strategy: first_fit
  packing_length: 51200

trainer_args:
  use_rmpad: true
  use_liger_kernel: true
  fsdp2: true
```

---

### Qwen3-VL-8B with Sequence Parallel

**Configuration:**

- 4 Node x 8 A800-SXM4 GPU
- Packing length: 51,200
- Sequence Parallel degree: 2
- Optimization: Remove padding + Liger kernel + Iterable dataset
- Training mode: FSDP distributed + Ulysses Sequence Parallel

**Achieved MFU: 0.20-0.25 (20-25%)**

**Key settings:**
```yaml
dataset_config:
  packing: true
  packing_strategy: first_fit
  packing_length: 51200

trainer_args:
  use_rmpad: true
  use_liger_kernel: true
  fsdp2: true
  sp_ulysses_degree: 2  # Sequence parallel degree
```

**Note:** Sequence parallelism reduces MFU due to communication overhead, but enables training with longer sequences and higher batch sizes that wouldn't fit in single GPU memory.

---

## Unified Models

### Bagel

**Configuration:**

- 4 Node x 8 H100 GPU
- Packing length: 10,240
- Optimization: Liger kernel + Iterable dataset
- Training mode: FSDP distributed + Ulysses Sequence Parallel

**Achieved MFU: 0.25-0.30 (25-30%)**

**Key settings:**
```yaml
dataset_config:
  packing: true
  packing_strategy: first_fit
  packing_length: 10240

trainer_args:
  use_rmpad: false
  use_liger_kernel: true
  fsdp2: true
```

---

## Important Considerations

### ViT FLOPs Not Included in MFU Calculation

The reported MFU metrics **do not include** Vision Transformer (ViT) FLOPs computation. This is important because:

1. **ViT FLOPs are non-negligible**: The actual total computational work includes ViT encoding of image/video tokens, but MFU calculation focuses on language model FLOPs
2. **Memory overhead**: ViT processing requires additional GPU memory for intermediate activations and attention computations
3. **Packing length reduction**: For multimodal models, you may need to reduce packing length compared to text-only models to accommodate ViT memory requirements

### Packing Length Recommendations

When transitioning between modalities, consider:

- **Text-only models**: Can use longest packing length (81,920) for maximum throughput
- **Image models**: Reduce packing length to ~61,440 due to ViT visual token processing
- **Video models**: Further reduce to ~51,200 due to accumulated visual tokens from multiple frames

If you encounter out-of-memory errors, try reducing packing length in increments while monitoring MFU to find the optimal balance between throughput and memory usage.

### Optimization Trade-offs

| Optimization | Memory Savings | Speed Improvement | Notes |
|-------------|-----------------|-------------------|-------|
| **Remove Padding (`use_rmpad`)** | 20-30% | 2-3× on variable sequences | Requires Flash Attention |
| **Liger Kernel** | ~30% | 10-20% | Fused operations for common kernels |
| **Sequence Packing** | Varies | 35-40% MFU vs 20-25% | First-fit bin packing with variable lengths |
| **Sequence Parallel (SP)** | 2-3× (SP degree) | Reduced efficiency | Enables ultra-long contexts |
