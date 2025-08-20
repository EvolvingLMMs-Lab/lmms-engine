#!/bin/bash
# Example training script for WanVideo models using lmms-engine-mini

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Example 1: Train T2V 1.3B model on a single GPU
echo "Training WanVideo T2V 1.3B model..."
python -m lmms_engine.launch.cli \
    --config examples/wanvideo/configs/wan2.1_t2v_1.3b.yaml

# Example 2: Train T2V 14B model with multiple GPUs using torchrun
# echo "Training WanVideo T2V 14B model with 8 GPUs..."
# torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
#     --master_addr="127.0.0.1" --master_port="8000" \
#     -m lmms_engine.launch.cli \
#     --config examples/wanvideo/configs/wan2.1_t2v_14b.yaml

# Example 3: Train I2V 14B model with accelerate
# echo "Training WanVideo I2V 14B model with accelerate..."
# accelerate launch --config_file examples/wanvideo/configs/accelerate_config_14B.yaml \
#     -m lmms_engine.launch.cli \
#     --config examples/wanvideo/configs/wan2.1_i2v_14b.yaml

# Example 4: Resume training from checkpoint
# echo "Resuming training from checkpoint..."
# python -m lmms_engine.launch.cli \
#     --config examples/wanvideo/configs/wan2.1_t2v_1.3b.yaml \
#     --resume

# Example 5: Evaluation only mode
# echo "Running evaluation..."
# python -m lmms_engine.launch.cli \
#     --config examples/wanvideo/configs/wan2.1_t2v_1.3b.yaml \
#     --eval_only
