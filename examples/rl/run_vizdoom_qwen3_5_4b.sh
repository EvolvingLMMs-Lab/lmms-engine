#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON:-.venv-vllm/bin/python}"
PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd -P)"
PYTHON_BIN="${PYTHON_BIN_DIR}/$(basename "${PYTHON_BIN}")"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

RUNTIME_ARGS=()
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"

RUNTIME_ARGS+=("--nnodes=${NNODES}")
RUNTIME_ARGS+=("--gpus-per-node=${GPUS_PER_NODE}")

if (( NNODES > 1 )); then
  MASTER_ADDR_VALUE="${MASTER_ADDR:-}"
  if [[ -z "${MASTER_ADDR_VALUE}" ]]; then
    echo "MASTER_ADDR must be set when NNODES/WORLD_SIZE > 1" >&2
    exit 2
  fi
  NODE_RANK_VALUE="${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}"
  RAY_PORT_VALUE="${RAY_PORT:-${MASTER_PORT:-6379}}"
  TRAIN_NODE_RANK_VALUE="${TRAIN_NODE_RANK:-0}"
  HEAD_NODE_IP_VALUE="${HEAD_NODE_IP:-${MASTER_ADDR_VALUE}}"
  RAY_WAIT_TIMEOUT_VALUE="${RAY_WAIT_TIMEOUT:-300}"

  RUNTIME_ARGS+=("--master-addr=${MASTER_ADDR_VALUE}")
  RUNTIME_ARGS+=("--ray-port=${RAY_PORT_VALUE}")
  RUNTIME_ARGS+=("--node-rank=${NODE_RANK_VALUE}")
  RUNTIME_ARGS+=("--train-node-rank=${TRAIN_NODE_RANK_VALUE}")
  RUNTIME_ARGS+=("--head-node-ip=${HEAD_NODE_IP_VALUE}")
  RUNTIME_ARGS+=("--ray-wait-timeout=${RAY_WAIT_TIMEOUT_VALUE}")
fi

exec "${PYTHON_BIN}" -m lmms_engine.launch.rl \
  hydra.output_subdir=null \
  hydra/job_logging=stdout \
  "${RUNTIME_ARGS[@]}" \
  config_yaml="${CONFIG:-examples/rl/vizdoom_qwen3_5_4b.yaml}" \
  "$@"
