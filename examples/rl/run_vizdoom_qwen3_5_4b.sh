#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PYTHON="${PYTHON:-.venv/bin/python}"
export RAY_TRAIN_NUM_WORKERS="${RAY_TRAIN_NUM_WORKERS:-7}"
export RAY_TRAIN_USE_GPU="${RAY_TRAIN_USE_GPU:-1}"
CONFIG="${CONFIG:-examples/rl/vizdoom_qwen3_5_4b.yaml}"

"${PYTHON}" -m lmms_engine.launch.rl config_yaml="${CONFIG}"
