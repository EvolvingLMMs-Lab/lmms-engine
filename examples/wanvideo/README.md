# WanVideo Training with LMMs-Engine-Mini

This directory contains examples and configurations for training WanVideo models using the LMMs-Engine-Mini framework.

## Overview

WanVideo is a family of diffusion-based video generation models that support:
- **Text-to-Video (T2V)**: Generate videos from text descriptions
- **Image-to-Video (I2V)**: Generate videos from a starting image
- **Video-to-Video (V2V)**: Transform existing videos with text guidance
- **VACE**: Video All-in-one Creation and Editing model
- **Fun Controls**: Advanced control mechanisms for video generation

## Model Variants

The implementation currently supports the following WanVideo model configurations:
- **1.3B T2V**: Efficient text-to-video model for quick iterations (480×832 resolution)
- **14B T2V**: High-quality text-to-video model (480×832 resolution)
- **14B I2V**: Image-to-video model with higher resolution support (720×1280)

## Quick Start

### 1. Prepare Your Dataset

Organize your video dataset in CSV format with the following structure:

```bash
modelscope download --dataset DiffSynth-Studio/example_video_dataset --local_dir ./data/example_video_dataset
```

#### For T2V (Text-to-Video) training:
```csv
video,prompt
path/to/video1.mp4,"A beautiful sunset over the mountains"
path/to/video2.mp4,"A cat playing with a ball in a garden"
```

#### For I2V (Image-to-Video) training:
```csv
video,image,prompt
path/to/video1.mp4,path/to/first_frame1.jpg,"The scene comes to life with movement"
path/to/video2.mp4,path/to/first_frame2.jpg,"Camera pans across the landscape"
```

Save your metadata CSV file (e.g., `data/metadata.csv`) and update the dataset path in the configuration files.

### 2. Configure Training

We provide pre-configured YAML files for different model variants:

- `examples/wanvideo/configs/wan2.1_t2v_1.3b.yaml`: Text-to-Video 1.3B model
- `configs/wan2.1_t2v_14b.yaml`: Text-to-Video 14B model
- `configs/wan2.1_i2v_14b.yaml`: Image-to-Video 14B model

Modify the configuration files to match your dataset paths and training requirements.

### 3. Start Training

#### Single GPU Training

```bash
python -m lmms_engine.launch.cli --config examples/wanvideo/configs/wan2.1_t2v_1.3b.yaml
```

#### Multi-GPU Training with torchrun

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr="127.0.0.1" --master_port="8000" \
    -m lmms_engine.launch.cli --config examples/wanvideo/configs/wan2.1_t2v_1.3b.yaml
```

#### Multi-GPU Training with Accelerate

```bash
accelerate launch --use_fsdp \
    -m lmms_engine.launch.cli --config examples/wanvideo/configs/wan2.1_t2v_1.3b.yaml
```

#### Resume Training

To resume training, update the configuration file to include:
```yaml
trainer_args:
  resume_from_checkpoint: "./output/wan2.1_t2v_1.3b/checkpoint-XXX"
```

Then run the training command again.

## Configuration Details

### Model Configuration

Key parameters in the model configuration:

```yaml
model_config:
  load_from_config:
    model_type: wanvideo
    model_variant: "Wan2.1-T2V-1.3B"  # Model variant identifier
    model_size: "1.3B"  # or "14B"
    
    # DiT architecture
    dit_hidden_size: 2432  # Hidden dimension
    dit_num_layers: 28     # Number of transformer layers
    dit_num_heads: 19      # Number of attention heads
    dit_enable_flash_attn: true  # Enable Flash Attention
    
    # Training settings
    gradient_checkpointing: true
    scheduler_type: "flow_match"  # Scheduler for diffusion
    
    # Generation settings
    num_frames: 49
    height: 480
    width: 832
    guidance_scale: 5.0
    num_inference_steps: 20
```

### Training Arguments

Important training parameters:

```yaml
trainer_args:
  output_dir: "./output/wan2.1_t2v_1.3b"
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  
  # Mixed precision
  bf16: true
  tf32: true
  
  # Optimizer settings
  optim: "adamw_torch"
  weight_decay: 0.01
  lr_scheduler_type: "cosine"
  warmup_steps: 500
  
  # Checkpointing
  save_steps: 500
  save_total_limit: 3
  eval_steps: 500
```

## Advanced Features

### LoRA Fine-tuning

LoRA support is planned for future releases. Currently, use gradient checkpointing and mixed precision training to reduce memory usage:

```yaml
model_config:
  load_from_config:
    gradient_checkpointing: true
  attn_implementation: flash_attention_2

trainer_args:
  gradient_checkpointing: true
  bf16: true
  tf32: true
```

### FSDP for Large Models

For training 14B models across multiple GPUs (as configured in `wan2.1_i2v_14b.yaml`):

```yaml
trainer_args:
  fsdp: "full_shard auto_wrap"
  fsdp_config:
    backward_prefetch: "backward_pre"
    forward_prefetch: true
    use_orig_params: false
    cpu_ram_efficient_loading: true
    sync_module_states: true
    limit_all_gathers: true
    activation_checkpointing: true
    sharding_strategy: "FULL_SHARD"
```

### Dataset Configuration

The dataset is configured in the YAML files:

```yaml
dataset_config:
  dataset_type: vision
  dataset_format: csv  # Currently supports CSV format
  dataset_path: data/metadata.csv
  video_sampling_strategy: frame_num
  frame_num: 49  # Number of frames to sample
  video_backend: qwen_vl_utils  # Video processing backend
  
  # Processor configuration for video preprocessing
  processor_config:
    processor_type: wanvideo
    do_resize: true
    size:
      height: 480  # Target height
      width: 832   # Target width
    do_normalize: true
    image_mean: [0.5, 0.5, 0.5]
    image_std: [0.5, 0.5, 0.5]
```

#### CSV Format (Required)

The dataset must be in CSV format with the following columns:

**For T2V training:**
```csv
video,prompt
path/to/video1.mp4,"A scenic mountain landscape"
path/to/video2.mp4,"Urban cityscape at night"
```

**For I2V training:**
```csv
video,image,prompt
path/to/video1.mp4,path/to/frame1.jpg,"Movement starts from this frame"
path/to/video2.mp4,path/to/frame2.jpg,"Dynamic scene evolution"
```

## Monitoring Training

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir ./output/wan2.1_t2v_1.3b
```

## Inference

After training, you can load and use the model for inference. The model checkpoints are saved in the `output_dir` specified in your configuration.

```python
import torch
from lmms_engine.models.wanvideo import (
    WanVideoForConditionalGeneration,
    WanVideoProcessor,
    WanVideoConfig,
)

# Load the trained model
checkpoint_path = "./output/wan2.1_t2v_1.3b/checkpoint-XXX"
config = WanVideoConfig.from_pretrained(checkpoint_path)
model = WanVideoForConditionalGeneration.from_pretrained(
    checkpoint_path,
    config=config,
    torch_dtype=torch.bfloat16,  # Use bf16 for efficiency
)
processor = WanVideoProcessor()

# Move model to GPU
model = model.to("cuda")

# Generate video from text prompt
prompt = "A serene lake surrounded by mountains at sunset"
with torch.no_grad():
    outputs = model.generate(
        prompt=prompt,
        num_frames=49,
        height=480,
        width=832,
        num_inference_steps=20,
        guidance_scale=5.0,
    )

# Process and save the generated video
# Note: Video post-processing implementation depends on your specific requirements
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
