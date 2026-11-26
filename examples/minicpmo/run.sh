#!/bin/bash
# ####################################################################################
# # MiniCPM-o 4.5 Training Launch Script
# #
# # Supports both simplex (turn-based image/audio/video SFT) and duplex
# # (realtime listen/speak) training.  See docs/models/minicpmo.md for the
# # realtime data format.
# ####################################################################################

NGPUS=8

# ----- Paths (edit these) -----
MODEL_PATH="openbmb/MiniCPM-o-4_5"       # or a local checkpoint path
DATA_PATH="/path/to/train.parquet"
DATA_FOLDER="/path/to/media/root"

# ----- Mode: "simplex" or "realtime" -----
MODE="simplex"

if [ "$MODE" = "realtime" ]; then
    CONFIG_NAME="realtime_config"
else
    CONFIG_NAME="example_config"
fi

CONFIG_DIR="$(cd "$(dirname "$0")" && pwd)"

torchrun --nproc_per_node=${NGPUS} --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=12355 \
    -m lmms_engine.launch.cli \
    --config-path "${CONFIG_DIR}" \
    --config-name "${CONFIG_NAME}" \
    model_config.load_from_pretrained_path="${MODEL_PATH}" \
    dataset_config.processor_config.processor_name="${MODEL_PATH}" \
    dataset_config.datasets.0.path="${DATA_PATH}" \
    dataset_config.datasets.0.data_folder="${DATA_FOLDER}"

# EXAMPLES:
#   Quick debug run:
#     trainer_args.max_steps=5 trainer_args.print_batch_input_steps=1
#   Freeze the vision tower as well:
#     trainer_args.freeze_modules='["tts","vpm","resampler"]'
#   Multi-node (2 nodes, 8 GPUs each), on node 0 / node 1 respectively:
#     torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
#         --master_addr=<node0-ip> --master_port=12355 ...
