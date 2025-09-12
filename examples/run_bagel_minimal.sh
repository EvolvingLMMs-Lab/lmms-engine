#!/usr/bin/env bash
set -euo pipefail

# Minimal single-GPU run for Bagel text-only training
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Optional validation of batches
# export LMMS_BAGEL_VALIDATE=1

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=12355 \
  -m lmms_engine.launch.cli --config examples/bagel_minimal.yaml

