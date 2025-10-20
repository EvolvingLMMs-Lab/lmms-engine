# CICD Training Tests

This directory contains scripts for running Continuous Integration/Continuous Deployment (CICD) tests for training workflows in the lmms-engine-mini project.

## Overview

The CICD system provides an automated way to test training scripts across multiple models and configurations. It handles GPU management, data dependencies, and test orchestration.

## Directory Structure

```
cicd/
├── README.md              # This file
└── run_traincicd.sh       # Main entry point for running CICD tests

test/train/                # Python test implementations (separate folder)
├── run_cicd.py           # Python unittest launcher
├── utils.py              # Utility functions for training tests
├── qwen2_5/              # Qwen 2.5 model tests
├── qwen2_5_vl/           # Qwen 2.5 VL model tests
├── qwen3_vl/             # Qwen 3 VL model tests
├── llava_onevision/      # LLaVA OneVision model tests
└── bagel/                # Bagel model tests
```

## Quick Start

### Basic Usage

Run all training tests with default settings (2 GPUs):

```bash
./cicd/run_traincicd.sh
```

### Test Specific Model

Test a specific model (e.g., qwen2_5_vl):

```bash
./cicd/run_traincicd.sh --model-name qwen2_5_vl
```

### Custom GPU Count

Specify the number of GPUs to use:

```bash
./cicd/run_traincicd.sh --gpu-count 4
```


## Data Requirements

The tests require test data to be present at `../data/lmms_engine_test` (relative to the cicd directory).

### Automatic Data Download

If the data folder doesn't exist, the script will automatically download it from Hugging Face:

```bash
hf download kcz358/lmms_engine_test --local-dir ../data/lmms_engine_test --repo-type dataset
```

Make sure you have the Hugging Face CLI installed:

```bash
pip install -U huggingface_hub
```

## Command Line Options

### `run_traincicd.sh` Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-name NAME` | Model to test (e.g., qwen2_5, qwen2_5_vl, qwen3_vl) | (empty - tests all models) |
| `--gpu-count NUM` | Number of GPUs to use | 2 |
| `--no-verbose` | Disable verbose output | (verbose enabled) |
| `--help` | Show help message | - |

### Available Models

- `qwen2_5` - Qwen 2.5 base model
- `qwen2_5_vl` - Qwen 2.5 Vision-Language model
- `qwen3_vl` - Qwen 3 Vision-Language model
- `llava_onevision` - LLaVA OneVision model
- `bagel` - Bagel model

## Python Test Runner

The bash script wraps around a Python-based test runner located at `test/train/run_cicd.py`. This Python script uses the unittest framework to discover and run tests.

### Direct Python Usage

You can also run tests directly with Python:

```bash
# Run all tests
python test/train/run_cicd.py --verbose

# Run tests for a specific model
python test/train/run_cicd.py --model-name qwen2_5_vl --verbose

# Specify GPU count
python test/train/run_cicd.py --gpu-count 4 --verbose

# Stop on first failure
python test/train/run_cicd.py --failfast
```

### Python Runner Options

| Option | Description |
|--------|-------------|
| `--test-pattern PATTERN` | Pattern to match test files (default: test_*.py) |
| `--verbose`, `-v` | Run tests in verbose mode |
| `--failfast` | Stop on first failure |
| `--gpu-count NUM` | Override GPU count for testing |
| `--model-name NAME` | Optional model name to test |

## Test Utilities

The `test/train/utils.py` file provides helper functions for training tests:

- **`with_temp_dir`**: Decorator that creates a temporary directory for tests
- **`get_available_gpus`**: Returns the number of available GPUs
- **`launch_torchrun_training`**: Launches training using torchrun with real-time output streaming
- **`with_multi_gpu_training`**: Decorator for tests that need multi-GPU training

## Examples

### Example 1: Quick Test of All Models

```bash
cd /path/to/lmms-engine-mini
./cicd/run_traincicd.sh
```

### Example 2: Test Single Model with 8 GPUs

```bash
./cicd/run_traincicd.sh --model-name qwen3_vl --gpu-count 8
```

### Example 3: Run Tests Without Verbose Output

```bash
./cicd/run_traincicd.sh --no-verbose
```

### Example 4: Direct Python Execution with Failfast

```bash
python test/train/run_cicd.py --model-name qwen2_5 --verbose --failfast
```

## Environment Variables

The test runner sets `CUDA_VISIBLE_DEVICES` based on the `--gpu-count` parameter to control GPU visibility for the tests.

## Troubleshooting

### Data Not Found

If you see an error about missing data:

```
Data folder not found at: /path/to/data/lmms_engine_test
```

The script will automatically attempt to download it. If this fails:
1. Ensure you have the Hugging Face CLI installed
2. Verify you have access to the dataset: `kcz358/lmms_engine_test`
3. Check your internet connection

### GPU Issues

If tests fail due to GPU problems:
1. Check available GPUs: `nvidia-smi`
2. Adjust `--gpu-count` to match your available GPUs
3. Verify CUDA is properly installed

### Test Failures

If tests fail:
1. Run with `--verbose` flag for detailed output
2. Use `--model-name` to isolate which model is failing
3. Check the test logs in the output

## Contributing

When adding new model tests:
1. Create a new directory under `test/train/` with the model name
2. Add test files with the pattern `test_*.py`
3. Update this README with the new model name
4. Ensure tests use the utilities from `utils.py` for consistency

## Related Files

- `test/train/run_cicd.py` - Python unittest test runner
- `test/train/utils.py` - Shared test utilities
- `test/train/*/` - Individual model test directories

