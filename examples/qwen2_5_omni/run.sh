#!/bin/bash

################################################################################
# Qwen2.5-Omni 7B Training - Unified Multimodal (Vision + Audio + Text)
################################################################################
#
# DESCRIPTION:
#   Train Qwen2.5-Omni unified multimodal model supporting vision, audio,
#   and text modalities with FSDP2 and Ulysses Sequence Parallel.
#
# KEY FEATURES:
#   - Unified image, audio, and text understanding
#   - Ulysses SP for long multimodal contexts
#   - Flash Attention 2 + unpadding
#   - Sequence packing support
#   - Liger Kernel fused operations
#   - FSDP2 distributed training
#
# REQUIREMENTS:
#   - 8x GPUs (A100/H100 recommended, 80GB VRAM)
#   - flash-attn: pip install flash-attn --no-build-isolation
#   - liger-kernel: pip install liger-kernel
#   - Audio processing: pip install librosa soundfile
#
# DATASET:
#   Prepare multimodal dataset in JSON format with image, audio, and text:
#   See: docs/data_prep.md
#
#   Example dataset format:
#   ```json
#   {
#     "conversations": [...],
#     "images": ["path/to/image.jpg"],
#     "audios": ["path/to/audio.wav"],
#     "videos": ["path/to/video.mp4"]
#   }
#   ```
#
# CONFIGURATION:
#   Edit example_config.yaml to customize:
#   - Dataset path: dataset_path
#   - Audio max length: audio_max_length (seconds)
#   - Video/Image max pixels: video_max_pixels, image_max_pixels
#   - SP degree: sp_ulysses_degree (1/2/4/8)
#   - Batch size: per_device_train_batch_size
#
# PERFORMANCE TIPS:
#   - Adjust sp_ulysses_degree for very long sequences
#   - Enable packing for better MFU: set packing: true
#   - Audio length affects memory: reduce audio_max_length if OOM
#   - Use gradient_checkpointing for larger models
#
################################################################################

# Number of GPUs
NGPUS=8

# Training command
torchrun --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12355 \
  -m lmms_engine.launch.cli \
  --config examples/qwen2_5_omni/example_config.yaml

################################################################################
# SINGLE GPU (for debugging):
# python -m lmms_engine.launch.cli --config examples/qwen2_5_omni/example_config.yaml
#
# MULTI-NODE TRAINING:
# Set --nnodes, --node_rank, and --master_addr accordingly
################################################################################
