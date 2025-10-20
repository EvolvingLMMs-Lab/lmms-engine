#!/usr/bin/env bash

GPUS=0
export WANDB_PROJECT="your-project-name"
export HF_HUB_DOWNLOAD_TIMEOUT=200
export HF_HUB_ETAG_TIMEOUT=200

# For single GPU, simply run with python (no distributed launcher needed)
CUDA_VISIBLE_DEVICES=$GPUS python -m lmms_engine.launch.cli \
  --config your-single-gpu-config.yaml \
  2>&1 | tee outputs/output_single_gpu.log
