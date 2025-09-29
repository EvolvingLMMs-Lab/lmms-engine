# BAGEL Model Training Guide

BAGEL is a multimodal model that combines visual understanding and generation capabilities. This guide covers how to train BAGEL models using the LMMS Engine.

## Overview

BAGEL integrates:
- **Language Model**: Qwen2-based architecture with MoT (Mixture of Tokens) support
- **Vision Understanding**: SigLIP vision transformer for image comprehension
- **Visual Generation**: VAE (Variational Autoencoder) for image generation
- **Unified Training**: Joint training for both understanding and generation tasks

## Prerequisites

- LMMS Engine installation
- CUDA-compatible GPU with sufficient memory
- PyTorch with FSDP2 support

## QuickStart

### 1. Prepare Your Dataset

An example dataset is available on `https://huggingface.co/datasets/kcz358/bagel-example`

### 2. Ovewrite config

Since the original Bagel config is not hf compatible and can't be used in AutoConfig, we prepare a config here `https://huggingface.co/kcz358/bagel_hf/blob/main/config.json`. Please overwrite this json file to the original `config.json` in the Bagel model path

### 3. Configure Training

Create a YAML configuration file based on the template above, adjusting:
- Dataset paths and format
- Model checkpoint path
- Training hyperparameters
- Output directory



### Basic Training Configuration

```yaml
- type: trainer
  config:
    trainer_type: fsdp2_trainer
    
    # Dataset configuration
    dataset_config:
      dataset_type: bagel_iterable
      dataset_format: parquet  # Supports: parquet, arrow, json, jsonl, yaml
      
      datasets:
        - path: "parquet path"  # Hugging Face dataset ID
          data_folder: ""
          data_type: parquet
      
      # Processor configuration
      processor_config:
        processor_name: "your-model-checkpoint-path"
        processor_type: "bagel"
      
      # Packing configuration (recommended)
      packing: true
      packing_strategy: first_fit
      packing_length: 4096
      video_backend: qwen_vl_utils
    
    # Model configuration
    model_config:
      load_from_pretrained_path: "your-model-checkpoint-path"
      attn_implementation: "eager"  # or "flash_attention_2"
      extra_kwargs:
        visual_und: false  # Enable/disable visual understanding
      # Optional: Enable Native Sparse Attention
      # monkey_patch_kwargs:
      #   patch_type: ["nsa"]
    
    # Training hyperparameters
    per_device_train_batch_size: 1
    learning_rate: 1.0e-06
    weight_decay: 0.0
    gradient_accumulation_steps: 1
    gradient_checkpointing: true
    max_steps: 1000
    save_steps: 500
    output_dir: "./output/bagel-training"
    bf16: true
    
    # FSDP2 configuration
    fsdp2: true
    fsdp_config:
      transformer_layer_cls_to_wrap: ["Qwen2MoTDecoderLayer"]
      reshard_after_forward: false
```

## Dataset Format

BAGEL expects datasets with the following structure:

### Required Fields

- `messages`: Conversation format with roles (system, user, assistant)

### Example Dataset Entry

```json
{
  "messages": [
    {
      "role": "user", 
      "content": [
        { "type": "image_url", "image_url": { "url": "path/to/image.jpg" } },
        { "type": "text", "text": "Describe this image" }
      ]
    },
    {
      "role": "assistant",
      "content": [
        { "type": "text", "text": "This image shows..." }
      ]
    }
  ]
}
```

## Key Configuration Options

To overwrite the arguments in Bagel training, we suggest the user to use the `extra_kwargs` in the config and get the parameters from that. You can check how it is being used in `src/lmms_engine/datasets/processor/bagel_processor.py` and `src/lmms_engine/models/bagel/bagel.py` in the `from_pretrained` method

## FSDP2 Configuration

FSDP2 (Fully Sharded Data Parallel v2) is recommended for training large BAGEL models:

```yaml
fsdp2: true
fsdp_config:
  transformer_layer_cls_to_wrap: ["Qwen2MoTDecoderLayer"]
  reshard_after_forward: false
```

## Advanced Features

### Native Sparse Attention (NSA) Support

We supports Native Sparse Attention training on BAGEL through monkey patching to improve memory efficiency and training speed for long sequences. NSA replaces the standard attention mechanism with a sparse variant that reduces computational complexity.

#### Prerequisites

Install the native sparse attention library:

```bash
pip install git+https://github.com/XunhaoLai/native-sparse-attention-triton.git
```

#### Configuration

Enable NSA by adding the monkey patch configuration to your model config:

```yaml
model_config:
  load_from_pretrained_path: "your-model-checkpoint-path"
  attn_implementation: "eager/sdpa"
  extra_kwargs:
    visual_und: false
  monkey_patch_kwargs:
    patch_type: ["nsa"]
    # NSA configuration parameters (all optional with defaults shown)
    block_size: 64              # Size of attention blocks
    compress_type: "weightedpool"  # Options: weightedpool, linear, avgpool
    kernel_size: 32             # Compression kernel size
    kernel_stride: 16           # Compression kernel stride
    topk: 16                    # Number of top-k blocks to keep
    init_blocks: 1              # Number of initial blocks to always include
    local_blocks: 2             # Number of local blocks around current position
    window_size: 512            # Local attention window size
```

#### NSA Parameters

We recommend you to find out the meaning for parameters from [here](https://github.com/XunhaoLai/native-sparse-attention-triton/tree/main/native_sparse_attention/ops#readme)

#### Usage Notes

- NSA is most beneficial for longer seqence
- The sparse attention pattern is learned during training and adapts to the data
- All NSA parameters can be tuned based on your specific use case and hardware constraints


### Sequence Packing

BAGEL supports efficient sequence packing to maximize GPU utilization:
- `first_fit`: Pack sequences to minimize padding
- Configurable `packing_length` for optimal memory usage

### Mixed Precision Training

- `bf16`: Recommended for stability and performance
- Automatic loss scaling for gradient stability

### Gradient Checkpointing

- Reduces memory usage at the cost of computation
- Essential for training large models


## Model Architecture Details

### Components

- **Language Model**: Qwen2 architecture with MoT extensions
- **Vision Encoder**: SigLIP for image understanding
- **VAE**: Autoencoder for image generation
- **Connectors**: MLPs bridging different modalities

### Training Objectives

- Cross-entropy loss for language modeling
- MSE loss for visual generation
- Configurable loss weighting
