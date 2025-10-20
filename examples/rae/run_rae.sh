#!/bin/bash
# hf download luodian/rae_siglip2 --local-dir data/rae_siglip2 # to make sure you have the pre-trained model and processor for RAE
DATASET_PATH="ILSVRC/imagenet-1k"
PROCESSOR_NAME="./data/rae_siglip2"
MODEL_PATH="./data/rae_siglip2"
ATTN_IMPLEMENTATION="flash_attention_2"
PER_DEVICE_TRAIN_BATCH_SIZE=64
LEARNING_RATE=2.0e-04
WEIGHT_DECAY=0.0
GRADIENT_ACCUMULATION_STEPS=1
GRADIENT_CHECKPOINTING=true
NUM_TRAIN_EPOCHS=16
RUN_NAME="rae_siglip"
OUTPUT_DIR="./output/rae_siglip_v2"  # New directory to start fresh training
MAX_STEPS=-1  # Use epochs instead of max_steps
# Optimizer betas matching original RAE config
ADAM_BETA1=0.5
ADAM_BETA2=0.9
# Learning rate schedule matching original RAE config
LR_SCHEDULER_TYPE=cosine
WARMUP_RATIO=0.0625  # 1 epoch out of 16
# Gradient clipping - original RAE uses 0.0 (disabled)
MAX_GRAD_NORM=10.0
# Note: EMA decay (0.9978) is hardcoded in rae_trainer.py to match original RAE config

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="8000" \
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