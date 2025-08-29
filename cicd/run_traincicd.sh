#!/bin/bash

# Test the training scripts using Python-based CICD launcher
# This allows for better GPU management and test orchestration

echo "Starting training CICD tests..."

# Run the Python-based test launcher
# The default test pattern "test_*.py" will automatically discover tests in subdirectories
python test/train/run_cicd.py --verbose --gpu-count 2

echo "Training CICD tests completed."