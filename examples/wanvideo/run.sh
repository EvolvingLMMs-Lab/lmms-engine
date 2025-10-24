#!/bin/bash

################################################################################
# WanVideo Training - Text/Image-to-Video Generation
################################################################################
#
# DESCRIPTION:
#   Train WanVideo diffusion-based video generation models supporting
#   Text-to-Video (T2V), Image-to-Video (I2V), and Video-to-Video (V2V).
#
# KEY FEATURES:
#   - Text-to-Video (T2V) generation
#   - Image-to-Video (I2V) generation
#   - Video-to-Video (V2V) transformation
#   - Flow-matching scheduler
#   - FSDP2 distributed training
#   - Model variants: 1.3B and 14B parameters
#
# REQUIREMENTS:
#   - 8x GPUs (A100/H100, 80GB VRAM recommended for 14B)
#   - Video dataset in JSONL/CSV format
#   - ffmpeg for video processing
#
# DATASET PREPARATION:
#   1. Download example dataset:
#      modelscope download --dataset DiffSynth-Studio/example_video_dataset \
#        --local_dir ./data/example_video_dataset
#
#   2. For T2V (Text-to-Video), prepare metadata.jsonl:
#      {"video": "path/to/video1.mp4", "prompt": "A beautiful sunset"}
#
#   3. For I2V (Image-to-Video), include first frame:
#      {"video": "path/to/video1.mp4", "image": "path/to/frame.jpg", "prompt": "Scene description"}
#
# CONFIGURATION:
#   Available configs in configs/:
#   - wan2.2_ti2v_5b_from_pretrained.yaml  # 5B T2V+I2V from pretrained
#   - wan2.2_ti2v_5b_from_scratch.yaml     # 5B T2V+I2V from scratch
#
#   Key parameters to customize:
#   - Model path: load_from_pretrained_path
#   - Dataset path: dataset_path
#   - Video resolution: size.height, size.width (default: 480x832)
#   - Frame count: frame_num (default: 49)
#   - Batch size: per_device_train_batch_size
#   - Learning rate: learning_rate (5e-6 for large models)
#
# MODEL VARIANTS:
#   - 1.3B T2V: Efficient, faster training (480×832)
#   - 14B T2V: High quality, slower training (480×832)
#   - 14B I2V: Image conditioning (720×1280)
#
# PERFORMANCE TIPS:
#   - Use gradient_checkpointing for large models (already enabled)
#   - Adjust gradient_accumulation_steps based on GPU memory
#   - 14B model requires higher accumulation steps (4-8)
#   - Monitor VRAM with: watch -n 1 nvidia-smi
#
################################################################################

# Configuration file (choose one)
CONFIG="wan2.2_ti2v_5b_from_pretrained.yaml"
# CONFIG="wan2.2_ti2v_5b_from_scratch.yaml"

# Number of GPUs
NGPUS=8

# Training command
torchrun --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12355 \
  -m lmms_engine.launch.cli \
  config_yaml=examples/wanvideo/configs/${CONFIG}

################################################################################
# SINGLE GPU (for debugging small models):
# python -m lmms_engine.launch.cli \
#   config_yaml=examples/wanvideo/configs/${CONFIG}
#
# MULTI-NODE TRAINING:
# Set --nnodes, --node_rank, and --master_addr accordingly
#
# WANDB LOGGING (recommended):
# export WANDB_PROJECT="wanvideo-training"
# export WANDB_ENTITY="your-entity"
#
# TROUBLESHOOTING:
# - OOM: Reduce per_device_train_batch_size or increase gradient_accumulation_steps
# - Video loading errors: Check video file formats and paths
# - Slow data loading: Increase dataloader_num_workers
#
# For more details, see: examples/wanvideo/README.md
################################################################################
