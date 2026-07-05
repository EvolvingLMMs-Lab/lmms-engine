#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PYTHON="${PYTHON:-.venv/bin/python}"
export RAY_TRAIN_NUM_WORKERS="${RAY_TRAIN_NUM_WORKERS:-7}"
export RAY_TRAIN_USE_GPU="${RAY_TRAIN_USE_GPU:-1}"
CONFIG="${CONFIG:-examples/rl/vizdoom_qwen3_5_4b.yaml}"
MODEL_PATH="${REPO_ROOT}/data/checkpoints/Qwen3.5-4B"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Missing ${MODEL_PATH}; download Qwen/Qwen3.5-4B before launching." >&2
  exit 1
fi

"${PYTHON}" -m lmms_engine.launch.rl config_yaml="${CONFIG}"
