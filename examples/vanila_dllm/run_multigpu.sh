#!/usr/bin/env bash

GPUS=0,1,2,3
export WANDB_PROJECT="your-project-name"       # goes to wandb.ai/<entity>/my‑cool‑project
export HF_HUB_DOWNLOAD_TIMEOUT=200   # 单个文件下载过程中，服务器最长多久必须回包
export HF_HUB_ETAG_TIMEOUT=200      # 请求 repo 元数据（etag）时的等待上限
WORLD_SIZE=$(awk -F',' '{print NF}' <<<"$GPUS")

CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
  --num_processes $WORLD_SIZE \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 29504 \
  --dynamo_backend=no \
  --mixed_precision=no \
  --module lmms_engine.launch.cli \
  --config your-config-file.yaml \
  2>&1 | tee outputs/output_multi_gpu.log

  # --multi_gpu \