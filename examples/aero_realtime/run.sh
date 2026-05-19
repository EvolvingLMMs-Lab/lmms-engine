#!/bin/bash

################################################################################
# AeroRealtime Training — LLaVA-Video-178K (Normal Video QA)
################################################################################
#
# DESCRIPTION:
#   Train the AeroRealtime model (5.09B params) on LLaVA-Video-178K data.
#   Uses Hydra --config-path / --config-name so all settings can be overridden
#   from the command line.
#
# MODEL:
#   - Vision: Qwen3-VL-4B vision tower (with built-in merger)
#   - Audio: Qwen2-Audio-7B encoder (auto-extracted from video)
#   - Language: Qwen3-VL-4B text model + lm_head
#   - Fusion: timestep-aligned mean pooling (audio -> vision bins)
#   - Design: additive dual-stream (text_stream_ids + vision features)
#
# PREREQUISITES:
#   1. Prepare checkpoint:  python tools/prepare_init_weight/prepare_aero_realtime.py
#   2. Convert data:        python tools/convert_data/convert_llava_video_to_parquet.py
#   3. Install deps:        pip install flash-attn --no-build-isolation && pip install liger-kernel librosa
#
# USAGE:
#   Set MODEL_PATH, DATA_PATH, DATA_FOLDER below, then run:
#     bash scripts/launch/aero_realtime_train.sh
#
################################################################################

# ----- Paths (edit these) ----------------------------------------------------
MODEL_PATH=/path/to/aero_realtime_init
DATA_PATH=/path/to/llava_video_0_30_s_cap_oe.parquet
DATA_FOLDER=/path/to/LLaVA-Video-178K
# -----------------------------------------------------------------------------

NGPUS=8
CONFIG_DIR=$(cd "$(dirname "$0")/../config" && pwd)

torchrun --nproc_per_node="${NGPUS}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="8000" \
    -m lmms_engine.launch.cli \
    --config-path "${CONFIG_DIR}" \
    --config-name aero_realtime \
    model_config.load_from_pretrained_path="${MODEL_PATH}" \
    dataset_config.processor_config.processor_name="${MODEL_PATH}" \
    dataset_config.datasets.0.path="${DATA_PATH}" \
    dataset_config.datasets.0.data_folder="${DATA_FOLDER}"

################################################################################
# EXAMPLES:
#
# Quick debug (5 steps, single GPU):
#   python -m lmms_engine.launch.cli \
#       --config-path scripts/config \
#       --config-name aero_realtime \
#       model_config.load_from_pretrained_path=${MODEL_PATH} \
#       dataset_config.processor_config.processor_name=${MODEL_PATH} \
#       dataset_config.datasets.0.path=${DATA_PATH} \
#       dataset_config.datasets.0.data_folder=${DATA_FOLDER} \
#       trainer_args.max_steps=5 \
#       trainer_args.print_batch_input_steps=1
#
# Freeze vision tower:
#   ... trainer_args.freeze_modules='["visual"]'
#
# Change learning rate:
#   ... trainer_args.learning_rate=2e-5
#
# Multi-node (2 nodes):
#   torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
#       --master_addr=<RANK_0_IP> --master_port=8000 \
#       -m lmms_engine.launch.cli \
#       --config-path scripts/config \
#       --config-name aero_realtime \
#       model_config.load_from_pretrained_path=${MODEL_PATH} \
#       ...
#
################################################################################
