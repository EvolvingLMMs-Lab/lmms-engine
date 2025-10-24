#!/bin/bash

# SiT Training Launch Script
# Scalable Interpolant Transformers (Diffusion Model)
# Paper: https://arxiv.org/abs/2401.08740

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
MASTER_PORT=${MASTER_PORT:-8000}     # Master node port

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
  -m lmms_engine.launch.cli config_yaml=$CONFIG_FILE

# ==================== Usage Examples ====================
# Single node 8 GPUs (SiT-XL/2):
#   bash examples/scalable_interpolant_transformer/run.sh
#
# Custom GPU count:
#   NPROC_PER_NODE=4 bash examples/scalable_interpolant_transformer/run.sh
#
# Multi-node training (Node 0):
#   NNODES=4 NODE_RANK=0 MASTER_ADDR=192.168.1.100 bash examples/scalable_interpolant_transformer/run.sh
#
# Multi-node training (Node 1):
#   NNODES=4 NODE_RANK=1 MASTER_ADDR=192.168.1.100 bash examples/scalable_interpolant_transformer/run.sh
