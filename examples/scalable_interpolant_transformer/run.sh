#!/bin/bash

################################################################################
# Scalable Interpolant Transformers (SiT) Training
################################################################################
#
# DESCRIPTION:
#   Train diffusion transformers for class-conditional image generation
#   using Scalable Interpolant framework on ImageNet-1K.
#
# KEY FEATURES:
#   - DiT-based architecture (XL/2 = 675M params)
#   - Flexible interpolant paths (Linear, GVP, VP)
#   - Classifier-Free Guidance (CFG)
#   - FSDP2 distributed training
#   - EMA for stable generation
#   - ImageNet-1K training
#
# PAPER:
#   https://arxiv.org/abs/2401.08740
#
# REQUIREMENTS:
#   - 8x GPUs (A100/H100 recommended)
#   - ImageNet-1K dataset: ILSVRC/imagenet-1k (HuggingFace)
#   - HuggingFace token for dataset access
#   - flash-attn: pip install flash-attn --no-build-isolation
#
# DATASET:
#   Uses HuggingFace ImageNet-1K (ILSVRC/imagenet-1k).
#   Requires HuggingFace token with access to the dataset.
#   Set HF_TOKEN environment variable.
#
# CONFIGURATION:
#   Edit sit_xl_2.yaml to customize:
#   - Model size: hidden_size, depth, num_heads
#   - Image size: image_size (default: 256x256)
#   - Batch size: per_device_train_batch_size
#   - Learning rate: learning_rate (default: 1e-4)
#   - CFG scale: cfg_scale (default: 4.0)
#   - Interpolant path: Change in trainer
#
# MODEL ARCHITECTURE:
#   - Input size: 32 (256÷8, patch_size=8)
#   - Hidden size: 384
#   - Depth: 12 layers
#   - Attention heads: 6
#   - Parameters: ~675M
#
# PERFORMANCE TIPS:
#   - Use Flash Attention (already enabled)
#   - Adjust batch size based on GPU memory
#   - Use gradient_accumulation_steps for larger effective batch
#   - EMA improves generation quality (already enabled)
#
################################################################################

# ==================== Environment Variables ====================
export NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN}"  # Set your HuggingFace token
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="1"

# ==================== Training Parameters ====================
# GPU Configuration
NPROC_PER_NODE=${NPROC_PER_NODE:-8}  # Number of GPUs per node
NNODES=${NNODES:-1}                  # Number of nodes
NODE_RANK=${NODE_RANK:-0}            # Current node rank
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}  # Master node address
MASTER_PORT=${MASTER_PORT:-12355}     # Master node port

# Configuration File (default: SiT-XL/2)
CONFIG_FILE=${CONFIG_FILE:-"examples/scalable_interpolant_transformer/sit_xl_2.yaml"}

# ==================== Print Configuration ====================
echo "======================================"
echo "SiT Training Configuration"
echo "======================================"
echo "GPUs per node: $NPROC_PER_NODE"
echo "Number of nodes: $NNODES"
echo "Node rank: $NODE_RANK"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Config file: $CONFIG_FILE"
echo "======================================"

# ==================== Launch Training ====================
torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  -m lmms_engine.launch.cli \
  config_yaml=$CONFIG_FILE

################################################################################
# USAGE EXAMPLES:
#
# Single node 8 GPUs (SiT-XL/2):
#   bash examples/scalable_interpolant_transformer/run.sh
#
# Custom GPU count:
#   NPROC_PER_NODE=4 bash examples/scalable_interpolant_transformer/run.sh
#
# Multi-node training (Node 0):
#   NNODES=4 NODE_RANK=0 MASTER_ADDR=192.168.1.100 \
#     bash examples/scalable_interpolant_transformer/run.sh
#
# Multi-node training (Node 1):
#   NNODES=4 NODE_RANK=1 MASTER_ADDR=192.168.1.100 \
#     bash examples/scalable_interpolant_transformer/run.sh
#
# TROUBLESHOOTING:
# - HuggingFace access: Set HF_TOKEN with ImageNet access
# - OOM: Reduce per_device_train_batch_size
# - NCCL errors: Check network connectivity between nodes
#
# For more details, see: examples/scalable_interpolant_transformer/README.md
################################################################################
