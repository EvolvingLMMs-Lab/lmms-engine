#!/bin/bash

# Aesthetic Scorer Service Launch Script
# Usage: bash run.sh [--conda-env ENV_NAME]

set -e  # Exit on error

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"

# # Parse arguments
# USE_CONDA=false
# CONDA_ENV=""
# while [[ $# -gt 0 ]]; do
#     case $1 in
#         --conda-env)
#             USE_CONDA=true
#             CONDA_ENV="$2"
#             shift 2
#             ;;
#         *)
#             echo "Unknown option: $1"
#             exit 1
#             ;;
#     esac
# done

# # Check if model weights exist
# if [ ! -f "sac+logos+ava1-l14-linearMSE.pth" ]; then
#     echo "ERROR: Model weights file 'sac+logos+ava1-l14-linearMSE.pth' not found!"
#     echo "Please download the model weights to: $SCRIPT_DIR"
#     exit 1
# fi

# # Activate conda environment if requested
# if [ "$USE_CONDA" = true ]; then
#     echo "Activating conda environment: $CONDA_ENV"
#     source $(conda info --base)/etc/profile.d/conda.sh
#     conda activate "$CONDA_ENV"
# fi

# # Check if gunicorn is installed
# if ! command -v gunicorn &> /dev/null; then
#     echo "ERROR: gunicorn is not installed!"
#     echo "Install it with: pip install gunicorn"
#     echo "Or install all dependencies: pip install -r requirements.txt"
#     exit 1
# fi

# # Check if required Python packages are available
# python3 -c "import flask, torch, transformers" 2>/dev/null || {
#     echo "ERROR: Missing required Python packages!"
#     echo "Install dependencies: pip install -r requirements.txt"
#     exit 1
# }

# echo "Starting Aesthetic Scorer Service..."
# echo "Service will be available at: http://0.0.0.0:18080"
# echo "Press Ctrl+C to stop, or run: pkill -f 'gunicorn.*aesthetic_scorer_service' to stop later"

# Launch gunicorn
gunicorn -c gunicorn.conf.py "app:create_app()"