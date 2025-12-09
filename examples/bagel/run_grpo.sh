#!/bin/bash

################################################################################
# BAGEL GRPO Training (RL) with FSDP2
################################################################################
#
# DESCRIPTION:
#   Run the GRPO (Group Relative Policy Optimization) pipeline for BAGEL's
#   image generation capability inside lmms-engine, leveraging the custom
#   BagelGRPOTrainer and rl_prompt_* datasets.
#
# KEY FEATURES:
#   - Vision understanding and generation
#   - Qwen2-based LLM with MoT (Mixture of Tokens)
#   - SigLIP vision encoder
#   - VAE for image generation
#   - Sequence packing support
#   - FSDP2 distributed training
#   - Optional Native Sparse Attention (NSA)
#
# REQUIREMENTS:
#   - 8x GPUs (A100/H100 recommended, 80GB VRAM)
#   - flash-attn: pip install flash-attn --no-build-isolation
#   - Optional NSA: pip install git+https://github.com/XunhaoLai/native-sparse-attention-triton.git
#
# DATASET:
#   Prepare RL prompts following rl_prompt_text/rl_prompt_jsonl specs:
#   - Text prompts: one prompt per line (train.txt, test.txt, etc.)
#   - JSONL prompts: {"prompt": "...", "task": "...", ...}
#   - Image-edit JSONL: {"prompt": "...", "image": "rel/path.jpg", ...}
#
# MODEL CHECKPOINT:
#   You can use either:
#   1. Original BAGEL weights (requires config.json conversion)
#      See: https://huggingface.co/kcz358/bagel_hf/blob/main/config.json
#   2. Converted HF weights: lmms-lab/BAGEL-7B-MoT-ver.LE (recommended)
#
# CONFIGURATION:
#   Edit bagel_grpo_config.yaml to customize:
#   - Model checkpoint: model_config.load_from_pretrained_path
#   - Dataset path: dataset_config.dataset_path / eval_dataset_path
#   - RL settings: extra_kwargs.grpo_config (reward_fn, sample/train params)
#   - Logging/checkpointing: trainer_args
#
# PERFORMANCE TIPS:
#   - Enable packing for better GPU utilization (packing: true)
#   - Use NSA for long sequences (enable monkey_patch_kwargs)
#   - Adjust packing_length based on GPU memory (default: 4096)
#   - Monitor memory with: watch -n 1 nvidia-smi
#
# ADVANCED FEATURES:
#   - Native Sparse Attention: Uncomment monkey_patch_kwargs in config
#   - Mixed Understanding/Generation: Set visual_und: true/false
#
################################################################################

# Number of GPUs
NGPUS=8

# Training command
torchrun --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12357 \
  -m lmms_engine.launch.cli \
  config_yaml=examples/bagel/bagel_grpo_config.yaml

################################################################################
# MULTI-NODE TRAINING:
#
# On rank 0 node:
# torchrun --nproc_per_node=8 \
#   --nnodes=2 \
#   --node_rank=0 \
#   --master_addr=<RANK_0_IP> \
#   --master_port=12357 \
#   -m lmms_engine.launch.cli \
#   config_yaml=examples/bagel/bagel_grpo_config.yaml
#
# On rank 1 node:
# torchrun --nproc_per_node=8 \
#   --nnodes=2 \
#   --node_rank=1 \
#   --master_addr=<RANK_0_IP> \
#   --master_port=12357 \
#   -m lmms_engine.launch.cli \
#   config_yaml=examples/bagel/bagel_grpo_config.yaml
#
################################################################################
#
# TROUBLESHOOTING:
#   - If config.json is not HF compatible, use the converted weights
#   - Ensure reward server endpoints are reachable if using remote scoring
#   - Verify dataset paths exist before launching
#   - Check docs/developer_guide/new_trainer_guide.md for trainer details
#
################################################################################
