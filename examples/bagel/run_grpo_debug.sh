#!/bin/bash

################################################################################
# BAGEL GRPO Training (RL) - Single GPU Debug Mode
################################################################################
#
# This script runs GRPO training on a single GPU for debugging purposes.
# Useful for:
#   - Debugging reward computation issues
#   - Testing configuration changes
#   - Memory profiling
#   - Step-by-step debugging with breakpoints
#
# USAGE:
#   bash examples/bagel/run_grpo_debug.sh
#
#   Or specify GPU:
#   CUDA_VISIBLE_DEVICES=0 bash examples/bagel/run_grpo_debug.sh
#
# NOTES:
#   - Make sure to reduce batch sizes in bagel_grpo_config.yaml for single GPU
#   - Consider reducing num_batches_per_epoch and num_image_per_prompt
#   - FSDP2 will still work but only on one GPU
#
################################################################################

# Set GPU (default to GPU 0)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Set environment variables for single GPU
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12357
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

echo "=========================================="
echo "Running BAGEL GRPO in Single GPU Debug Mode"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Single GPU training command (no torchrun needed)
python -m lmms_engine.launch.cli \
  config_yaml=examples/bagel/bagel_grpo_config.yaml

################################################################################
# ALTERNATIVE: Using torchrun with 1 GPU
#
# If the above doesn't work, you can also use:
# torchrun --nproc_per_node=1 \
#   --nnodes=1 \
#   --node_rank=0 \
#   --master_addr=127.0.0.1 \
#   --master_port=12357 \
#   -m lmms_engine.launch.cli \
#   config_yaml=examples/bagel/bagel_grpo_config.yaml
#
################################################################################

