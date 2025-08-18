# WanVideo Training with LMMs-Engine-Mini

This directory contains examples and configurations for training WanVideo models using the LMMs-Engine-Mini framework.

## Overview

WanVideo is a family of diffusion-based video generation models that support:
- **Text-to-Video (T2V)**: Generate videos from text descriptions
- **Image-to-Video (I2V)**: Generate videos from a starting image
- **Video-to-Video (V2V)**: Transform existing videos with text guidance
- **VACE**: Video aesthetic and consistency enhancement
- **Fun Controls**: Advanced control mechanisms for video generation

## Model Variants

The implementation supports various WanVideo model sizes:
- **1.3B**: Efficient model for quick iterations
- **5B**: Balanced model for quality and speed
- **14B**: High-quality model for production use

## Quick Start

### 1. Prepare Your Dataset

First, organize your video dataset and create metadata:

```bash
# Create metadata for T2V training
python prepare_dataset.py create \
    --video_dir /path/to/videos \
    --output data/metadata.json

# Create metadata for I2V training
python prepare_dataset.py create_i2v \
    --video_dir /path/to/videos \
    --image_dir /path/to/first_frames \
    --output data/i2v_metadata.json

# Validate dataset
python prepare_dataset.py validate \
    --metadata data/metadata.json \
    --video_root /path/to/videos \
    --check_files
```

### 2. Configure Training

We provide pre-configured YAML files for different model variants:

- `configs/wan2.1_t2v_1.3b.yaml`: Text-to-Video 1.3B model
- `configs/wan2.1_t2v_14b.yaml`: Text-to-Video 14B model
- `configs/wan2.1_i2v_14b.yaml`: Image-to-Video 14B model

Modify the configuration files to match your dataset paths and training requirements.

### 3. Start Training

#### Single GPU Training

```bash
python train_wanvideo.py --config configs/wan2.1_t2v_1.3b.yaml
```

#### Multi-GPU Training with torchrun

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr="127.0.0.1" --master_port="8000" \
    train_wanvideo.py --config configs/wan2.1_t2v_14b.yaml
```

#### Multi-GPU Training with Accelerate

```bash
accelerate launch --config_file accelerate_config.yaml \
    train_wanvideo.py --config configs/wan2.1_t2v_14b.yaml
```

#### Resume Training

```bash
python train_wanvideo.py --config configs/wan2.1_t2v_1.3b.yaml --resume
```

## Configuration Details

### Model Configuration

Key parameters in the model configuration:

```yaml
model_config:
  load_from_config:
    model_type: wanvideo
    model_size: "1.3B"  # or "5B", "14B"
    
    # DiT architecture
    dit_hidden_size: 2432
    dit_num_layers: 28
    dit_num_heads: 19
    
    # Training settings
    gradient_checkpointing: true
    use_lora: false  # Enable for efficient fine-tuning
    lora_rank: 32
    
    # Generation settings
    num_frames: 49
    height: 480
    width: 832
```

### Training Arguments

Important training parameters:

```yaml
trainer_args:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  
  # Mixed precision
  bf16: true
  tf32: true
  
  # Checkpointing
  save_steps: 500
  save_total_limit: 3
```

## Advanced Features

### LoRA Fine-tuning

For efficient fine-tuning with limited GPU memory:

```yaml
model_config:
  load_from_config:
    use_lora: true
    lora_rank: 128
    lora_target_modules: ["q", "k", "v", "o", "ffn.0", "ffn.2"]
```

### FSDP for Large Models

For training 14B models across multiple GPUs:

```yaml
trainer_args:
  fsdp: "full_shard auto_wrap"
  fsdp_config:
    backward_prefetch: "backward_pre"
    forward_prefetch: true
    activation_checkpointing: true
```

### Custom Dataset Format

The expected dataset format for T2V:

```json
[
  {
    "video_path": "path/to/video.mp4",
    "caption": "A description of the video content",
    "video_id": "unique_video_id",
    "duration": 10.5,
    "fps": 30,
    "resolution": "1920x1080"
  }
]
```

For I2V, add an `image_path` field:

```json
[
  {
    "video_path": "path/to/video.mp4",
    "image_path": "path/to/first_frame.jpg",
    "caption": "A description of the video content",
    "video_id": "unique_video_id"
  }
]
```

## Monitoring Training

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir ./output/wan2.1_t2v_1.3b
```

## Inference

After training, use the model for inference:

```python
from lmms_engine.models.wanvideo import (
    WanVideoForConditionalGeneration,
    WanVideoProcessor,
    WanVideoConfig,
)

# Load model
config = WanVideoConfig.from_pretrained("./output/wan2.1_t2v_1.3b")
model = WanVideoForConditionalGeneration.from_pretrained(
    "./output/wan2.1_t2v_1.3b",
    config=config,
)
processor = WanVideoProcessor()

# Generate video
prompt = "A serene lake surrounded by mountains at sunset"
video = model.generate(
    prompt=prompt,
    num_frames=49,
    height=480,
    width=832,
    num_inference_steps=20,
    guidance_scale=5.0,
)
```

## Troubleshooting

### Out of Memory Issues

1. Enable gradient checkpointing:
   ```yaml
   gradient_checkpointing: true
   ```

2. Reduce batch size and increase gradient accumulation:
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 16
   ```

3. Use LoRA for fine-tuning instead of full training

4. Enable FSDP for multi-GPU setups

### Slow Training

1. Ensure Flash Attention is installed:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. Use mixed precision training (bf16)

3. Enable TF32 for Ampere GPUs:
   ```yaml
   tf32: true
   ```

## Citation

If you use WanVideo in your research, please cite:

```bibtex
@article{wanvideo2024,
  title={WanVideo: Unified Video Generation with Diffusion Models},
  author={WanVideo Team},
  year={2024}
}
```

## License

This implementation is provided under the Apache 2.0 License.
