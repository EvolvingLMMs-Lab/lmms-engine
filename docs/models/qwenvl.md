# Qwen-VL Model Training Guide

Qwen-VL models are state-of-the-art multimodal models that support image and video understanding. This guide covers training both Qwen2.5-VL and Qwen3-VL models using the LMMS Engine.

## Overview

### Qwen2.5-VL
- **Architecture**: Advanced vision-language model with M-RoPE (Multimodal Rotary Position Embedding)
- **Position Encoding**: 3D RoPE for temporal (T), height (H), width (W) dimensions
- **Modalities**: Image and Video understanding
- **Context Length**: Up to 128K tokens
- **Key Features**: 3D M-RoPE, Dynamic resolution ViT, Flash Attention 2, Liger Kernel, RMPad, Sequence Parallelism

### Qwen3-VL
- **Architecture**: Latest generation with Interleaved-MRoPE and DeepStack visual feature fusion
- **Position Encoding**: Interleaved-MRoPE with enhanced text-timestamp alignment
- **Unique Feature**: DeepStack - multi-layer visual embeddings fused into early language model layers
- **Modalities**: Image and Video understanding (optimized for long videos)
- **Context Length**: 256K tokens (native), extendable to 1M tokens
- **Key Features**: DeepStack fusion, Interleaved 3D M-RoPE, Long video support (>1 hour), Flash Attention 2, Sequence Parallelism

## Prerequisites

- LMMS Engine installation
- CUDA-compatible GPU with sufficient memory
- PyTorch with FSDP2 support
- Flash Attention 2 (recommended)

### Install Flash Attention

```bash
uv pip install flash-attn --no-build-isolation
```

If you encounter symbol errors:

```bash
uv pip install --no-build-isolation --no-cache-dir flash-attn
```

## Quick Start

### 1. Prepare Your Dataset

Prepare your dataset in OpenAI chat messages format with image/video/audio content types. See [Data Preparation Guide](../user_guide/data_prep.md) for details.

Example data structure:
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}},
        {"type": "text", "text": "Describe this image"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "This image shows..."}
      ]
    }
  ]
}
```

### 2. Configure Training

Create a YAML configuration file for your model.

## Training Configuration (Example)

### Qwen2.5-VL Configuration

```yaml
- type: trainer
  config:
    trainer_type: fsdp2_trainer
    
    # Dataset configuration
    dataset_config:
      dataset_type: vision                    # Or vision_audio for audio support
      dataset_format: yaml
      
      datasets:
        - path: "path/to/your/dataset.parquet"
          data_folder: ""
          data_type: parquet
      
      # Processor configuration
      processor_config:
        processor_name: "Qwen/Qwen2.5-VL-7B-Instruct"  # Or 3B/72B variants
        processor_type: "qwen2_5_vl"
      
      # Packing configuration
      packing: true
      packing_strategy: first_fit
      packing_length: 16384
      
      # Video configuration
      video_backend: qwen_vl_utils
      video_sampling_strategy: fps
      video_max_pixels: 50176                 # 224 * 224
      video_max_frames: 512
      fps: 1
    
    # Model configuration
    model_config:
      load_from_pretrained_path: "Qwen/Qwen2.5-VL-7B-Instruct"
      attn_implementation: "flash_attention_2"
    
    # Training hyperparameters
    per_device_train_batch_size: 1
    learning_rate: 1.0e-06
    weight_decay: 0.0
    gradient_accumulation_steps: 1
    gradient_checkpointing: true
    num_train_epochs: 1
    save_steps: 100
    save_total_limit: 1
    report_to: "wandb"
    output_dir: "./output/qwen2_5_vl"
    warmup_ratio: 0.0
    run_name: "qwen2_5_vl_training"
    eval_strategy: "no"
    logging_steps: 1
    group_by_length: true
    dataloader_num_workers: 8
    bf16: true
    lr_scheduler_type: "cosine"
    
    # Optional: Freeze vision encoder
    freeze_modules: ["visual"]
    
    # Performance optimizations
    use_liger_kernel: true
    use_rmpad: true
    
    # FSDP2 configuration
    fsdp2: true
    fsdp_config:
      transformer_layer_cls_to_wrap: ["Qwen2_5_VLDecoderLayer"]
      reshard_after_forward: false
    
    # Optional: Sequence parallelism
    sp_ulysses_degree: 1                       # Set to 2, 4, 8 for sequence parallel
```

### Qwen3-VL Configuration

```yaml
- type: trainer
  config:
    trainer_type: fsdp2_trainer
    
    # Dataset configuration
    dataset_config:
      dataset_type: qwen3_vl_iterable           # Use iterable dataset for Qwen3-VL
      dataset_format: yaml
      
      datasets:
        - path: "path/to/your/dataset.parquet"
          data_folder: ""
          data_type: parquet
      
      # Processor configuration
      processor_config:
        processor_name: "Qwen/Qwen3-VL-8B-Instruct"  # Or 4B variant
        processor_type: "qwen3_vl"
      
      # Packing configuration
      packing: false                             # Note: packing for Qwen3-VL
      packing_length: 51200
      filter_overlong: true
      
      # Video configuration - Qwen3-VL optimized
      video_backend: qwen_vl_utils
      video_sampling_strategy: fps
      video_max_pixels: 50176                    # 224 * 224
      video_max_frames: 512
      fps: 1
    
    # Model configuration
    model_config:
      load_from_pretrained_path: "Qwen/Qwen3-VL-8B-Instruct"
      attn_implementation: "flash_attention_2"
    
    # Training hyperparameters
    per_device_train_batch_size: 1
    learning_rate: 2.0e-04                       # Slightly higher for Qwen3-VL
    weight_decay: 0.0
    gradient_accumulation_steps: 1
    gradient_checkpointing: true
    max_steps: 1000                              # Use max_steps for iterable dataset
    save_steps: 1000
    save_total_limit: 1
    report_to: "wandb"
    output_dir: "./output/qwen3_vl"
    warmup_ratio: 0.1
    run_name: "qwen3_vl_training"
    eval_strategy: "no"
    logging_steps: 1
    dataloader_num_workers: 8
    bf16: true
    lr_scheduler_type: "cosine"
    
    # Performance optimizations
    use_liger_kernel: true
    use_rmpad: true
    
    # FSDP2 configuration
    fsdp2: true
    fsdp_config:
      transformer_layer_cls_to_wrap: ["Qwen3VLTextDecoderLayer"]
      reshard_after_forward: false
    
    # Optional: Sequence parallelism
    sp_ulysses_degree: 1
```

## Key Configuration Parameters

### Dataset Type (Example)

| Model | dataset_type | Description |
|-------|-------------------------|-------------|
| **Qwen2.5-VL** | `vision` | Map-style dataset, supports packing |
| **Qwen3-VL** | `qwen3_vl_iterable` | Streaming dataset optimized for Qwen3-VL |

### Processor Configuration

- **processor_name**: HuggingFace model identifier
  - Qwen2.5-VL: `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`
  - Qwen3-VL: `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`
- **processor_type**: Must match the model series
  - Qwen2.5-VL: `"qwen2_5_vl"`
  - Qwen3-VL: `"qwen3_vl"`

### FSDP2 Configuration

FSDP2 (Fully Sharded Data Parallel v2) is recommended for training large Qwen-VL models:

```yaml
fsdp2: true
fsdp_config:
  # Qwen2.5-VL
  transformer_layer_cls_to_wrap: ["Qwen2_5_VLDecoderLayer"] # include "Qwen3VLVisionBlock" to wrap ViT layers
  
  # Qwen3-VL
  # transformer_layer_cls_to_wrap: ["Qwen3VLTextDecoderLayer"]
  
  reshard_after_forward: false # If true, reshard parameters after each forward pass (saves memory but increases communication)
```

## Advanced Features

### Sequence Parallelism

Both Qwen2.5-VL and Qwen3-VL support Ulysses-style sequence parallelism for long context training:

```yaml
trainer_args:
  sp_ulysses_degree: 2  # Sequence parallel degree (1, 2, 4, 8)
```

**Benefits**:
- Enables training with longer sequences
- Reduces memory per GPU
- Scales efficiently across GPUs

**Requirements**:
- Flash Attention 2 must be installed
- `use_rmpad: true` recommended
- Number of attention heads must be divisible by `sp_ulysses_degree`

### Liger Kernel

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) provides fused kernels for efficient training:

```yaml
trainer_args:
  use_liger_kernel: true
```

**Optimizations**:
- Fused CrossEntropy kernel (~30% memory reduction)
- Fused RMSNorm
- Fused RoPE
- Fused SwiGLU

### RMPad (Remove Padding)

RMPad removes padding tokens for more efficient computation:

```yaml
trainer_args:
  use_rmpad: true
```

**Benefits**:
- ~15-25% speedup by removing pad token computation
- Works seamlessly with Flash Attention 2
- Essential for packing efficiency

### Freezing Modules

Freeze the vision encoder for faster training when only fine-tuning language understanding:

```yaml
trainer_args:
  freeze_modules: ["visual"]
```

### Mixed Precision Training

- **bf16**: Recommended for stability and performance
- **fp16**: Alternative if bf16 not supported

```yaml
trainer_args:
  bf16: true          # Preferred
  # fp16: true        # Alternative
```

### Gradient Checkpointing

Reduces memory at the cost of computation:

```yaml
trainer_args:
  gradient_checkpointing: true
```

## Run Training

### Launch Command

```bash
export NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false

# Optional: HuggingFace setup
export HF_TOKEN="<YOUR HF_TOKEN>"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER="1"

export NCCL_DEBUG=INFO

CONFIG="your_config.yaml"

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8000 \
    -m lmms_engine.launch.cli config_yaml=${CONFIG}
```

### Multi-Node Training

```bash
# Node 0
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<MASTER_NODE_IP> \
    --master_port=8000 \
    -m lmms_engine.launch.cli config_yaml=${CONFIG}

# Node 1
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<MASTER_NODE_IP> \
    --master_port=8000 \
    -m lmms_engine.launch.cli config_yaml=${CONFIG} \
    hydra.output_subdir=null hydra/job_logging=disabled
```

In multi-node training, simultaneous starts cause Hydra conflicts due to [timestamped working directories](https://hydra.cc/docs/configure_hydra/workdir/). Use `hydra.output_subdir=null` and `hydra/job_logging=disabled` to fix this.

## Model Architecture Details

### Qwen2.5-VL Architecture

**Core Components**:
- **Language Model**: Qwen2.5 decoder architecture (e.g., 3B/7B/72B variants)
- **Vision Encoder**: ViT-based encoder with dynamic resolution support
- **Position Encoding**: **M-RoPE (Multimodal Rotary Position Embedding)**
  - Separate position encodings for temporal (T), height (H), width (W) dimensions
  - Enables better alignment of visual tokens with text sequences
  - Uses `mrope_section` parameter to split RoPE across 3 dimensions
  - Computed via `apply_multimodal_rotary_pos_emb` with RoPE deltas
- **Video Processing**: 
  - Temporal-aware processing using RoPE deltas
  - Supports temporal grid (T, H, W) for video frames
  - Native video token integration in language model
- **Context Length**: Up to 128K tokens
- **Modality Support**: Image, Video, and optional Audio (via audio encoder)

**Key Features**:
- Dynamic resolution ViT allows variable image sizes
- M-RoPE provides fine-grained spatial-temporal position encoding
- Unified multimodal token processing in language model

### Qwen3-VL Architecture

**Core Components**:
- **Language Model**: Qwen3 decoder architecture (e.g., 4B/8B variants) with efficiency improvements
- **Vision Encoder**: Enhanced ViT with multi-layer feature extraction
- **Position Encoding**: **Interleaved-MRoPE**
  - Improved version of M-RoPE with better text-timestamp alignment
  - Optimized for long video processing with second-level indexing
  - Enhanced temporal understanding for video sequences
- **DeepStack Feature** (Unique to Qwen3-VL):
  - Extracts visual features from multiple vision encoder layers
  - Fuses multi-layer visual embeddings into language model's early layers
  - Provides fine-grained visual-language alignment
  - Reference: [DeepStack Paper](https://arxiv.org/abs/2406.04334)
- **Video Processing**:
  - Optimized for long videos (supports >1 hour)
  - Second-level timestamp alignment with text
  - Enhanced temporal reasoning capabilities
- **Context Length**: Native support for 256K tokens, extendable to 1M tokens
- **Modality Support**: Image and Video (optimized for long-form video understanding)

**Key Features**:
- DeepStack multi-layer visual feature fusion
- Interleaved-MRoPE for superior temporal alignment
- Extended context length for long videos and documents
- Improved efficiency in video token processing

### Architecture Comparison

| Feature | Qwen2.5-VL | Qwen3-VL |
|---------|-----------|----------|
| **Position Encoding** | M-RoPE (3D: T, H, W) | Interleaved-MRoPE |
| **Visual Feature Fusion** | Single-layer fusion | DeepStack multi-layer fusion |
| **Video Temporal Alignment** | RoPE deltas | Second-level timestamp alignment |
| **Context Length** | 128K tokens | 256K-1M tokens |
| **Long Video Support** | Good | Excellent (>1 hour) |
| **Model Sizes** | 3B, 7B, 72B | 4B, 8B |
| **Primary Use Case** | General multimodal | Long-form video & document understanding |

### Model Selection Guide

**Choose Qwen2.5-VL if you:**
- Need audio understanding capabilities
- Want larger model options (72B for best performance)
- Require general-purpose multimodal understanding
- Work with images, short-medium videos, and audio
- Need mature, well-tested architecture

**Choose Qwen3-VL if you:**
- Focus on long video understanding (>1 hour)
- Need extended context length (>128K tokens)
- Require fine-grained visual-language alignment (DeepStack)
- Work primarily with video analysis and temporal reasoning
- Want improved efficiency with smaller model sizes
- Need second-level timestamp alignment for videos

**Performance Considerations**:
- **Qwen2.5-VL 7B**: Balanced choice for most multimodal tasks
- **Qwen2.5-VL 72B**: Best performance, requires significant compute
- **Qwen3-VL 8B**: Optimal for long video understanding with moderate compute
- **Qwen3-VL 4B**: Efficient choice for video tasks with limited resources

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Solutions**:
- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing: true`
- Reduce `video_max_pixels` or `video_max_frames`
- Increase `gradient_accumulation_steps`
- Enable sequence parallelism with `sp_ulysses_degree: 2`

#### 2. Flash Attention Installation Issues

**Problem**: Symbol not found or compilation errors

**Solution**:
```bash
# Clear cache and reinstall
pip uninstall flash-attn -y
uv pip install --no-build-isolation --no-cache-dir flash-attn
```

#### 3. Slow Training Speed

**Optimizations**:
- Enable `use_liger_kernel: true`
- Enable `use_rmpad: true`
- Enable `group_by_length: true` for better batching
- Increase `dataloader_num_workers`
- Use `bf16` instead of `fp16`
- Enable packing for Qwen2.5-VL: `packing: true`

#### 4. Video Loading Errors

**Problem**: Video cannot be loaded or processed

**Solutions**:
- Ensure `qwen-vl-utils` is installed: `pip install qwen-vl-utils`
- Check video file format compatibility
- Reduce `video_max_frames` if videos are too long
- Verify `video_backend: qwen_vl_utils` is set

#### 5. Qwen3-VL Dataset Length Unknown

**Problem**: Can't calculate steps per epoch with iterable dataset

**Solution**: Always use `max_steps` instead of `num_train_epochs`:
```yaml
trainer_args:
  max_steps: 1000                # Required for iterable datasets
  # num_train_epochs: 1          # Required for map-style datasets
```

## Performance Tips

### Optimizing Training Speed

1. **Use appropriate batch size**:
   - Start with `per_device_train_batch_size: 1`
   - Increase `gradient_accumulation_steps` to simulate larger batches

2. **Enable all optimizations**:
   ```yaml
   use_liger_kernel: true
   use_rmpad: true
   group_by_length: true
   bf16: true
   ```

3. **Video preprocessing**:
   - Use lower `fps` for faster loading (e.g., `fps: 0.5` for 1 frame per 2 seconds)
   - Reduce `video_max_frames` if full video not needed

4. **Sequence parallelism for long sequences**:
   - Set `sp_ulysses_degree: 2` or higher for sequences > 32K tokens

### Memory Management

1. **Estimate memory usage**:
   - 7B model with batch_size=1: ~40GB
   - 72B model with batch_size=1: ~150GB

2. **Reduce memory footprint**:
   - Enable gradient checkpointing
   - Use FSDP2 for multi-GPU training
   - Freeze visual encoder if only training language understanding

## Best Practices

1. **Start with pretrained models**: Always use official Qwen checkpoints from HuggingFace
2. **Use BF16 training**: More stable than FP16 for these models
3. **Enable packing for Qwen2.5-VL**: Significantly improves throughput
4. **Monitor training metrics**: Use WandB or TensorBoard for tracking
5. **Save checkpoints frequently**: Set reasonable `save_steps` values
6. **Test with small dataset first**: Verify configuration before full training

## Model Variants

### Qwen2.5-VL

| Model | Parameters | Context Length | Recommended Use |
|-------|-----------|----------------|-----------------|
| Qwen2.5-VL-3B-Instruct | 3B | 128K | Fast inference, limited resources |
| Qwen2.5-VL-7B-Instruct | 7B | 128K | Balanced performance and efficiency |
| Qwen2.5-VL-72B-Instruct | 72B | 128K | Best performance, requires significant resources |

### Qwen3-VL

| Model | Parameters | Context Length | Recommended Use |
|-------|-----------|----------------|-----------------|
| Qwen3-VL-4B-Instruct | 4B | Extended | Efficient training and inference |
| Qwen3-VL-8B-Instruct | 8B | Extended | Enhanced performance with DeepStack |

## Additional Resources

### Official Documentation
- [Qwen2.5-VL Blog](https://qwenlm.github.io/blog/qwen2-vl/)
- [Qwen3-VL Announcement](https://qwenlm.github.io/)
- [Qwen2.5-VL HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen3-VL HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

### Technical Papers
- [DeepStack: Multi-Layer Visual Feature Fusion](https://arxiv.org/abs/2406.04334) - The paper behind Qwen3-VL's unique architecture
- [M-RoPE: Multimodal Rotary Position Embedding](https://arxiv.org/abs/2308.10882) - Position encoding for multimodal models

### LMMS Engine Guides
- [Data Preparation Guide](../user_guide/data_prep.md)
- [Dataset Configuration](../user_guide/datasets.md)
- [Video Configuration Reference](../reference/video_configuration.md)
- [Design Principles](../reference/design_principle.md)

### Community Resources
- [LMMS Engine GitHub](https://github.com/EvolvingLMMs-Lab/lmms-engine)
- [Qwen GitHub](https://github.com/QwenLM/Qwen2-VL)

