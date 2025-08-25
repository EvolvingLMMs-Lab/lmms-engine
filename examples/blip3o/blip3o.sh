#!/bin/bash

# Default values
TP_SIZE=1
USE_TP=false
NPROC_PER_NODE=1

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tp) USE_TP=true; TP_SIZE="$2"; shift ;;
        --nproc_per_node) NPROC_PER_NODE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Base command
CMD="torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355 -m lmms_engine.launch.cli --config examples/blip3o/blip3o_example.yaml"

# Add deepspeed config if TP is enabled
if [ "$USE_TP" = true ]; then
    # Update tp_size in ds_config_tp.json
    sed -i "s/\"tp_size\": [0-9]*/\"tp_size\": $TP_SIZE/" examples/blip3o/ds_config_tp.json
    CMD="$CMD --deepspeed examples/blip3o/ds_config_tp.json"
fi

# Execute the command
echo "Running command: $CMD"
eval $CMD
