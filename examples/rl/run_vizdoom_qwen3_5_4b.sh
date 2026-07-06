#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON:-.venv-vllm/bin/python}"
PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd -P)"
PYTHON_BIN="${PYTHON_BIN_DIR}/$(basename "${PYTHON_BIN}")"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

RUNTIME_ARGS=()
NNODES_VALUE="${NNODES:-${WORLD_SIZE:-}}"
GPUS_PER_NODE_VALUE="${NUM_GPUS_PER_NODE:-${GPUS_PER_NODE:-}}"

if [[ -n "${NNODES_VALUE}" ]]; then
  RUNTIME_ARGS+=("--nnodes=${NNODES_VALUE}")
fi
if [[ -n "${GPUS_PER_NODE_VALUE}" ]]; then
  RUNTIME_ARGS+=("--gpus-per-node=${GPUS_PER_NODE_VALUE}")
fi

if [[ -n "${MASTER_ADDR:-}" ]]; then
  RUNTIME_ARGS+=("--master-addr=${MASTER_ADDR}")
  RUNTIME_ARGS+=("--head-node-ip=${HEAD_NODE_IP:-${MASTER_ADDR}}")
fi
if [[ -n "${RAY_PORT:-}" || -n "${MASTER_PORT:-}" ]]; then
  RUNTIME_ARGS+=("--ray-port=${RAY_PORT:-${MASTER_PORT:-6379}}")
fi
NODE_RANK_VALUE="${NODE_RANK:-${RANK:-${SLURM_NODEID:-}}}"
if [[ -n "${NODE_RANK_VALUE}" ]]; then
  RUNTIME_ARGS+=("--node-rank=${NODE_RANK_VALUE}")
fi
if [[ -n "${TRAIN_NODE_RANK:-}" ]]; then
  RUNTIME_ARGS+=("--train-node-rank=${TRAIN_NODE_RANK}")
fi
if [[ -n "${RAY_WAIT_TIMEOUT:-}" ]]; then
  RUNTIME_ARGS+=("--ray-wait-timeout=${RAY_WAIT_TIMEOUT}")
fi

if [[ -n "${NNODES_VALUE}" ]] && (( NNODES_VALUE > 1 )) && [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "MASTER_ADDR must be set when NNODES/WORLD_SIZE > 1" >&2
  exit 2
fi

exec "${PYTHON_BIN}" -m lmms_engine.launch.rl \
  hydra.output_subdir=null \
  hydra/job_logging=stdout \
  "${RUNTIME_ARGS[@]}" \
  config_yaml="${CONFIG:-examples/rl/vizdoom_qwen3_5_4b.yaml}" \
  "$@"
