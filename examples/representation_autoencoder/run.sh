#!/bin/bash

################################################################################
# Representation AutoEncoder (RAE) Training with SigLip
################################################################################
#
# DESCRIPTION:
#   Train a visual representation autoencoder with adversarial discriminator
#   using SigLip encoder on ImageNet-1K.
#
# KEY FEATURES:
#   - SigLip encoder + general decoder + discriminator
#   - LPIPS perceptual loss
#   - Differentiable augmentation
#   - EMA for stable generation (decay: 0.9978)
#   - FSDP2 distributed training
#   - ImageNet-1K training
#
# REQUIREMENTS:
#   - 8x GPUs (A100/H100 recommended, 80GB VRAM)
#   - ImageNet-1K dataset: ILSVRC/imagenet-1k (HuggingFace)
#   - Pre-trained RAE-SigLip model
#   - HuggingFace token for dataset access
#
# DATASET & MODEL SETUP:
#   1. Download pre-trained RAE-SigLip model:
#      hf download luodian/rae_siglip2 --local-dir data/rae_siglip2
#
#   2. Dataset is loaded from HuggingFace: ILSVRC/imagenet-1k
#      Requires HuggingFace token with ImageNet access.
#
# CONFIGURATION:
#   Key parameters (edit below):
#   - DATASET_PATH: HuggingFace dataset path
#   - MODEL_PATH: Pre-trained model path
#   - PER_DEVICE_TRAIN_BATCH_SIZE: Batch size per GPU (default: 64)
#   - LEARNING_RATE: Learning rate (default: 2e-4)
#   - NUM_TRAIN_EPOCHS: Training epochs (default: 16)
#   - ADAM_BETA1/BETA2: Optimizer betas (0.5, 0.9)
#
# MODEL ARCHITECTURE:
#   - Encoder: SigLip (frozen)
#   - Decoder: General decoder
#   - Discriminator: Adversarial discriminator
#   - EMA decay: 0.9978 (hardcoded in trainer)
#
# TRAINING DETAILS:
#   - Optimizer: AdamW (beta1=0.5, beta2=0.9)
#   - LR schedule: Cosine with warmup (0.0625 = 1 epoch)
#   - Gradient clipping: 10.0
#   - Loss: Adversarial + LPIPS perceptual loss
#
################################################################################

# ==================== Model & Dataset Configuration ====================
DATASET_PATH="ILSVRC/imagenet-1k"
PROCESSOR_NAME="./data/rae_siglip2"
MODEL_PATH="./data/rae_siglip2"

# ==================== Training Hyperparameters ====================
ATTN_IMPLEMENTATION="flash_attention_2"
PER_DEVICE_TRAIN_BATCH_SIZE=64
LEARNING_RATE=2.0e-04
WEIGHT_DECAY=0.0
GRADIENT_ACCUMULATION_STEPS=1
GRADIENT_CHECKPOINTING=true
NUM_TRAIN_EPOCHS=16
MAX_STEPS=-1  # Use epochs instead of max_steps

# Optimizer configuration (matching original RAE)
ADAM_BETA1=0.5
ADAM_BETA2=0.9

# Learning rate schedule
LR_SCHEDULER_TYPE=cosine
WARMUP_RATIO=0.0625  # 1 epoch out of 16
MAX_GRAD_NORM=10.0

# Output configuration
RUN_NAME="rae_siglip"
OUTPUT_DIR="./output/rae_siglip_v2"

# ==================== Launch Training ====================
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12355" \
    -m lmms_engine.launch.cli \
    trainer_type=rae_trainer \
    dataset_config.dataset_path=${DATASET_PATH} \
    dataset_config.dataset_format=hf_dataset \
    dataset_config.processor_config.processor_name=${PROCESSOR_NAME} \
    dataset_config.dataset_type=rae \
    dataset_config.processor_config.processor_type=rae_siglip \
    dataset_config.processor_config.processor_name=${PROCESSOR_NAME} \
    model_config.load_from_pretrained_path=${MODEL_PATH} \
    model_config.attn_implementation=${ATTN_IMPLEMENTATION} \
    trainer_args.max_steps=${MAX_STEPS} \
    trainer_args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    trainer_args.learning_rate=${LEARNING_RATE} \
    trainer_args.weight_decay=${WEIGHT_DECAY} \
    trainer_args.adam_beta1=${ADAM_BETA1} \
    trainer_args.adam_beta2=${ADAM_BETA2} \
    trainer_args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer_args.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
    trainer_args.num_train_epochs=${NUM_TRAIN_EPOCHS} \
    trainer_args.max_grad_norm=${MAX_GRAD_NORM} \
    trainer_args.run_name=${RUN_NAME} \
    trainer_args.output_dir=${OUTPUT_DIR} \
    trainer_args.fsdp2=true \
    trainer_args.fsdp_config.transformer_layer_cls_to_wrap=["SiglipEncoderLayer"] \
    trainer_args.fsdp_config.reshard_after_forward=false \
    trainer_args.sp_ulysses_degree=1 \
    trainer_args.freeze_modules=["encoder"] \
    trainer_args.use_liger_kernel=false \
    trainer_args.use_rmpad=false \
    trainer_args.dataloader_num_workers=4 \
    trainer_args.dataloader_prefetch_factor=2 \
    trainer_args.bf16=true \
    trainer_args.lr_scheduler_type=${LR_SCHEDULER_TYPE} \
    trainer_args.warmup_ratio=${WARMUP_RATIO} \
    trainer_args.logging_steps=1 \
    trainer_args.group_by_length=false \
    trainer_args.report_to=['wandb']

################################################################################
# NOTES:
# - EMA decay (0.9978) is hardcoded in rae_trainer.py
# - SigLip encoder is frozen (freeze_modules=["encoder"])
# - Uses LPIPS perceptual loss + adversarial loss
# - Differentiable augmentation applied to discriminator inputs
#
# WANDB LOGGING:
# export WANDB_PROJECT="rae-training"
# export WANDB_ENTITY="your-entity"
#
# TROUBLESHOOTING:
# - Download model first: hf download luodian/rae_siglip2 --local-dir data/rae_siglip2
# - HuggingFace access: Set HF_TOKEN with ImageNet access
# - OOM: Reduce PER_DEVICE_TRAIN_BATCH_SIZE
# - Monitor generated images in output directory
################################################################################
