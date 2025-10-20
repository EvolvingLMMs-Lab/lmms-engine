#!/bin/bash

# Test the training scripts using Python-based CICD launcher
# This allows for better GPU management and test orchestration

# Default values
MODEL_NAME=""
GPU_COUNT=2
VERBOSE="--verbose"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --no-verbose)
            VERBOSE=""
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-name NAME     Model to test (e.g., qwen2_5, qwen2_5_vl, qwen3_vl)"
            echo "                        Default: (empty - tests all models)"
            echo "  --gpu-count NUM       Number of GPUs to use (default: 2)"
            echo "  --no-verbose          Disable verbose output"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting training CICD tests..."
echo "Model: ${MODEL_NAME:-all models}"
echo "GPU Count: $GPU_COUNT"

# Get the absolute path of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data/lmms_engine_test"

# Check if data folder exists, otherwise download
if [ ! -d "$DATA_DIR" ]; then
    echo "Data folder not found at: $DATA_DIR"
    echo "Downloading data from hub..."
    # TODO: Fill in the download command here
    # Example: huggingface-cli download <repo-id> --repo-type dataset --local-dir "$DATA_DIR"
    hf download kcz358/lmms_engine_test --local-dir "$DATA_DIR" --repo-type dataset --cache-dir "$DATA_DIR/.cache"
else
    echo "Data folder found at: $DATA_DIR"
fi
# Build the command
CMD="python test/train/run_cicd.py $VERBOSE --gpu-count $GPU_COUNT"

# Add model name if specified
if [ -n "$MODEL_NAME" ]; then
    CMD="$CMD --model-name $MODEL_NAME"
fi

# Run the Python-based test launcher
$CMD

echo "Training CICD tests completed."