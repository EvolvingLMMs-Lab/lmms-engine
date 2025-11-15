#!/bin/bash

################################################################################
# Qwen3-VL MoE Training with FSDP2 + Expert Parallel + Sequence Parallel
################################################################################
#
# DESCRIPTION:
#   Train Qwen3-VL MoE vision-language model with MoE auxiliary loss support,
#   expert parallelism for efficient MoE distribution, and Ulysses Sequence
#   Parallel for long visual sequences.
#
# KEY FEATURES:
#   - MoE architecture with load-balanced auxiliary loss
#   - Expert Parallelism (EP) for distributing experts across GPUs
#   - Ulysses Sequence Parallel (SP) for 10K+ visual tokens
#   - Multi-resolution visual understanding
#   - Flash Attention 2 + unpadding (use_rmpad)
#   - Sequence packing for improved MFU
#   - Liger Kernel fused operations
#   - FSDP2 distributed training
#
# REQUIREMENTS:
#   - 8x GPUs minimum (A100/H100 recommended, 80GB VRAM)
#   - flash-attn: pip install flash-attn --no-build-isolation
#   - liger-kernel: pip install liger-kernel
#
# DATASET:
#   Prepare your dataset in OpenAI chat format (JSONL/Arrow):
#   See: docs/user_guide/data_prep.md
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
#   Edit qwen3_vl_moe_ep8.yaml to customize:
#   - Model: Qwen/Qwen3-VL-30B-A3B-Instruct (MoE model)
#   - EP degree: ep_degree (must match number of experts or divisor)
#   - SP degree: sp_ulysses_degree (1/2/4/8 for sequence length)
#   - Batch size: per_device_train_batch_size
#   - Packing length: packing_length
#   - Router aux loss coefficient: router_aux_loss_coef (default: 0.001)
#
# MoE TRAINING TIPS:
#   - Expert Parallelism (EP):
#     * Set ep_degree to number of experts for 1 expert per GPU
#     * Or use divisor of num_experts (e.g., ep_degree=4 for 8 experts)
#     * Reduces memory per GPU and enables scaling to larger models
#
#   - Auxiliary Loss:
#     * Ensures balanced expert usage during training
#     * Prevents expert collapse (all tokens → few experts)
#     * Default coefficient: 0.001 (tune based on loss curves)
#     * Monitor router_loss in logs
#
#   - Sequence Parallelism (SP):
#     * Use with EP for long sequences + MoE
#     * Degree 1: < 10K tokens
#     * Degree 2: 10K-20K tokens
#     * Degree 4: 20K-40K tokens
#     * Must be divisor of world_size / ep_degree
#
# PERFORMANCE TIPS:
#   - Use fused_linear_cross_entropy (enabled by default for EP)
#   - Enable packing for 35-40% MFU: set packing: true
#   - Monitor expert load balance in tensorboard/wandb
#   - Adjust router_aux_loss_coef if experts are imbalanced
#   - Use gradient_checkpointing for larger models
#
# EXAMPLE CONFIGURATIONS:
#   # 8 GPUs, EP=8, SP=1 (1 expert per GPU)
#   ep_degree: 8
#   sp_ulysses_degree: 1
#
#   # 16 GPUs, EP=8, SP=2 (2 ranks per expert, sequence split)
#   ep_degree: 8
#   sp_ulysses_degree: 2
#
################################################################################

# Number of GPUs
NGPUS=8

# Training command
torchrun --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12358 \
  -m lmms_engine.launch.cli \
  config_yaml=examples/qwen3_vl_moe/qwen3_vl_moe_ep8.yaml

################################################################################
# MULTI-NODE TRAINING:
#
# On rank 0 node:
# torchrun --nproc_per_node=8 \
#   --nnodes=2 \
#   --node_rank=0 \
#   --master_addr=<RANK_0_IP> \
#   --master_port=12358 \
#   -m lmms_engine.launch.cli \
#   config_yaml=examples/qwen3_vl_moe/qwen3_vl_moe_ep8.yaml
#
# On rank 1 node:
# torchrun --nproc_per_node=8 \
#   --nnodes=2 \
#   --node_rank=1 \
#   --master_addr=<RANK_0_IP> \
#   --master_port=12358 \
#   -m lmms_engine.launch.cli \
#   config_yaml=examples/qwen3_vl_moe/qwen3_vl_moe_ep8.yaml
#
################################################################################
