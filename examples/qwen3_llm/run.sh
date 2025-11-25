#!/bin/bash

################################################################################
# Qwen3 LLM Training with FSDP2
################################################################################
#
# DESCRIPTION:
#   Train Qwen3 language model (text-only) using FSDP2 distributed training.
#   This is for pure language modeling tasks without multimodal capabilities.
#
# KEY FEATURES:
#   - Text-only language modeling
#   - Flash Attention 2 + unpadding (use_rmpad)
#   - Sequence packing support
#   - Liger Kernel fused operations
#   - FSDP2 distributed training
#   - Group by length for efficiency
#
# REQUIREMENTS:
#   - 8x GPUs (A100/H100 recommended)
#   - flash-attn: pip install flash-attn --no-build-isolation
#   - liger-kernel: pip install liger-kernel
#
# DATASET:
#   Prepare your dataset in OpenAI chat format (JSONL/Arrow/Parquet):
#   See: docs/user_guide/data_prep.md
#
#   Example dataset entry:
#   ```json
#   {
#     "messages": [
#       {
#         "role": "user",
#         "content": "What is machine learning?"
#       },
#       {
#         "role": "assistant",
#         "content": "Machine learning is a subset of artificial intelligence..."
#       }
#     ]
#   }
#   ```
#
# CONFIGURATION:
#   Edit example_config.yaml to customize:
#   - Model size: change load_from_pretrained_path
#     * Qwen/Qwen3-0.6B (0.6B parameters)
#     * Qwen/Qwen3-1.7B (1.7B parameters)
#     * Qwen/Qwen3-4B (4B parameters)
#     * Qwen/Qwen3-8B (8B parameters)
#     * Qwen/Qwen3-14B (14B parameters)
#     * Qwen/Qwen3-32B (32B parameters)
#   - Sequence length: adjust packing_length
#   - Batch size: per_device_train_batch_size
#   - Learning rate: learning_rate
#
# PERFORMANCE TIPS:
#   - Enable packing for better GPU utilization (packing: true)
#   - Use gradient_checkpointing for larger models (already enabled)
#   - Adjust batch size and gradient accumulation for optimal throughput
#   - group_by_length improves efficiency (already enabled)
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
  --master_port=12358 \
  -m lmms_engine.launch.cli \
  config_yaml=examples/qwen3_llm/example_config.yaml

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
#   config_yaml=examples/qwen3_llm/example_config.yaml
#
# On rank 1 node:
# torchrun --nproc_per_node=8 \
#   --nnodes=2 \
#   --node_rank=1 \
#   --master_addr=<RANK_0_IP> \
#   --master_port=12358 \
#   -m lmms_engine.launch.cli \
#   config_yaml=examples/qwen3_llm/example_config.yaml
#
################################################################################
