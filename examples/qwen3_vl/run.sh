#!/bin/bash

################################################################################
# Qwen3-VL 8B Training with FSDP2 + Ulysses Sequence Parallel
################################################################################
#
# DESCRIPTION:
#   Train Qwen3-VL vision-language model with support for long sequences
#   using Ulysses Sequence Parallel and FSDP2 distributed training.
#
# KEY FEATURES:
#   - Multi-resolution visual understanding
#   - Ulysses SP for 10K+ visual tokens
#   - Flash Attention 2 + unpadding (use_rmpad)
#   - Sequence packing (35-40% MFU)
#   - Liger Kernel fused operations
#   - FSDP2 distributed training
#
# REQUIREMENTS:
#   - 8x GPUs (A100/H100 recommended, 80GB VRAM)
#   - flash-attn: pip install flash-attn --no-build-isolation
#   - liger-kernel: pip install liger-kernel
#
# DATASET:
#   Prepare your dataset in OpenAI chat format (JSONL/Arrow):
#   See: docs/data_prep.md
#
#   Example dataset YAML (data/video/debug.yaml):
#   ```yaml
#   datasets:
#     - path: /path/to/your/dataset
#       data_folder: ""
#       data_type: arrow
#   ```
#
# CONFIGURATION:
#   Edit example_config.yaml to customize:
#   - Model size (2B/8B/72B): change load_from_pretrained_path
#   - Sequence length: adjust packing_length
#   - SP degree: set sp_ulysses_degree (1/2/4/8)
#   - Batch size: per_device_train_batch_size
#   - Max frames: video_max_frames
#
# PERFORMANCE TIPS:
#   - Adjust sp_ulysses_degree based on sequence length:
#     * Degree 1: < 10K tokens
#     * Degree 2: 10K-20K tokens
#     * Degree 4: 20K-40K tokens
#     * Degree 8: 40K+ tokens
#   - Enable packing for better MFU: set packing: true
#   - Use gradient_checkpointing for larger models (already enabled)
#   - Monitor memory with: watch -n 1 nvidia-smi
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
  --config examples/qwen3_vl/example_config.yaml

################################################################################
# MULTI-NODE TRAINING:
#
# On rank 0 node:
# torchrun --nproc_per_node=8 \
#   --nnodes=2 \
#   --node_rank=0 \
#   --master_addr=<RANK_0_IP> \
#   --master_port=12355 \
#   -m lmms_engine.launch.cli \
#   --config examples/qwen3_vl/example_config.yaml
#
# On rank 1 node:
# torchrun --nproc_per_node=8 \
#   --nnodes=2 \
#   --node_rank=1 \
#   --master_addr=<RANK_0_IP> \
#   --master_port=12355 \
#   -m lmms_engine.launch.cli \
#   --config examples/qwen3_vl/example_config.yaml
#
################################################################################
