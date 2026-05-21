#!/bin/bash

NGPUS=${NGPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12355}

torchrun --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  -m lmms_engine.launch.cli \
  --config-path examples/qwen3_vl \
  --config-name vit_frame_parallel_sp
