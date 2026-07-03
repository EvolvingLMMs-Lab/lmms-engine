#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-examples/rl/vizdoom_qwen3_5_4b.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" -m torch.distributed.run --nproc-per-node="${NPROC_PER_NODE}" \
  -m lmms_engine.launch.cli \
  config_yaml="${CONFIG}"
