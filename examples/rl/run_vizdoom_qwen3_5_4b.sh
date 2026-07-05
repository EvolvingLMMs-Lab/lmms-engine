#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON:-.venv/bin/python}" -m lmms_engine.launch.rl config_yaml="${CONFIG:-examples/rl/vizdoom_qwen3_5_4b.yaml}"
