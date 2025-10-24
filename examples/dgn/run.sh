#!/bin/bash

################################################################################
# Gated DeltaNet (DGN) 1B Training from Scratch with Muon
################################################################################
#
# DESCRIPTION:
#   Train a 1B parameter Gated DeltaNet model from scratch on FineWeb-Edu
#   using the Muon optimizer for superior convergence.
#
# KEY FEATURES:
#   - Gated Linear Attention architecture (recurrent)
#   - FineWeb-Edu pretraining dataset (streaming)
#   - Muon optimizer with Newton-Schulz orthogonalization
#   - FSDP2 distributed training
#   - Sequence packing for efficiency
#   - Flash Attention with unpadding (use_rmpad)
#
# REQUIREMENTS:
#   - 8x GPUs recommended (A100/H100, 80GB VRAM)
#   - Dataset: HuggingFaceFW/fineweb-edu (streaming, no download needed)
#   - flash-attn: pip install flash-attn --no-build-isolation
#
# DATASET:
#   Uses streaming FineWeb-Edu dataset for language model pretraining.
#   The dataset is automatically streamed during training.
#
# CONFIGURATION:
#   Edit train_dgn_1b.yaml to customize:
#   - Model architecture: hidden_size, num_hidden_layers in model_config
#   - Sequence length: packing_length (default: 2048)
#   - Batch size: per_device_train_batch_size (default: 32)
#   - Learning rate: learning_rate (default: 0.001 for Muon)
#   - Max steps: max_steps (default: 10000)
#
# MUON OPTIMIZER:
#   - Optimized for Gated Linear Attention architectures
#   - Newton-Schulz orthogonalization for better conditioning
#   - 10x higher learning rate than AdamW (0.001 vs 0.0001)
#   - Superior convergence on recurrent architectures
#
# PERFORMANCE TIPS:
#   - Adjust gradient_accumulation_steps to fit batch size
#   - Enable sequence packing (already configured)
#   - Use split_batches for balanced GPU memory
#   - Monitor tokens/second with include_num_input_tokens_seen
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
  --config examples/dgn/train_dgn_1b.yaml

################################################################################
# WANDB LOGGING (recommended):
# export WANDB_PROJECT="dgn-pretraining"
# export WANDB_ENTITY="your-entity"
#
# SINGLE GPU (for debugging):
# python -m lmms_engine.launch.cli --config examples/dgn/train_dgn_1b.yaml
#
# MULTI-NODE TRAINING:
# Set --nnodes, --node_rank, and --master_addr accordingly
#
# MONITORING:
# - Watch GPU usage: watch -n 1 nvidia-smi
# - Monitor loss: tensorboard --logdir output/
# - Check tokens/sec in training logs
################################################################################
