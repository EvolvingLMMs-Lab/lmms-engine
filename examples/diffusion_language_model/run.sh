#!/bin/bash

################################################################################
# Diffusion Large Language Model (dLLM) Training with Muon Optimizer
################################################################################
#
# DESCRIPTION:
#   Train Qwen3-based diffusion language model with masked prediction
#   and superior convergence using Muon optimizer.
#
# KEY FEATURES:
#   - Qwen3 architecture with masked diffusion
#   - Muon optimizer (Newton-Schulz orthogonalization)
#   - Streaming FineWeb-Edu dataset
#   - FSDP2 distributed training (multi-GPU)
#   - Liger Kernel support
#
# REQUIREMENTS:
#   - Single GPU: 24GB+ VRAM (A6000/A100/4090)
#   - Multi-GPU: 8x GPUs recommended
#   - Dataset: HuggingFaceFW/fineweb-edu (streaming, no download needed)
#
# DATASET:
#   Uses streaming FineWeb-Edu dataset (no manual download required).
#   The dataset is automatically streamed during training.
#
# CONFIGURATION:
#   - Single GPU: dllm_train_muon_single_gpu.yaml
#   - Multi-GPU FSDP2: dllm_train_muon_multi_gpu_fsdp2.yaml
#
#   Key parameters to customize:
#   - Model size: adjust hidden_size, num_hidden_layers in model_config
#   - Sequence length: packing_length (default: 2048)
#   - Batch size: per_device_train_batch_size
#   - Learning rate: learning_rate (default: 0.001 for Muon)
#   - Max steps: max_steps
#
# MUON OPTIMIZER:
#   - Superior convergence compared to AdamW
#   - Newton-Schulz orthogonalization with Triton kernels
#   - Distributed via DTensor (FSDP2)
#   - Recommended LR: 0.001 (10x higher than AdamW)
#
################################################################################

# Configuration file (choose one)
CONFIG="dllm_train_muon_single_gpu.yaml"      # For single GPU
# CONFIG="dllm_train_muon_multi_gpu_fsdp2.yaml"  # For multi-GPU with FSDP2

# ============================================================================
# SINGLE GPU TRAINING
# ============================================================================
if [ "$CONFIG" = "dllm_train_muon_single_gpu.yaml" ]; then
  echo "Running single GPU training..."
  python -m lmms_engine.launch.cli \
    config_yaml=examples/diffusion_language_model/${CONFIG}

# ============================================================================
# MULTI-GPU TRAINING (FSDP2)
# ============================================================================
else
  echo "Running multi-GPU FSDP2 training..."
  NGPUS=8

  torchrun --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12355 \
    -m lmms_engine.launch.cli \
    config_yaml=examples/diffusion_language_model/${CONFIG}
fi

################################################################################
# WANDB LOGGING (optional):
# export WANDB_PROJECT="your-project-name"
# export WANDB_ENTITY="your-entity"
#
# TROUBLESHOOTING:
# - OOM: Reduce per_device_train_batch_size or increase gradient_accumulation_steps
# - Slow training: Increase dataloader_num_workers (default: 4)
# - HuggingFace timeout: export HF_HUB_DOWNLOAD_TIMEOUT=300
#
# MONITORING:
# - Watch GPU usage: watch -n 1 nvidia-smi
# - Monitor loss: tensorboard --logdir output/
################################################################################
