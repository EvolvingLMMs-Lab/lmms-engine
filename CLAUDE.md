# CLAUDE.md - LMMs Engine Repository Guide

## Repository Overview

**LMMs Engine** is a training framework for Large Multimodal Models (LMMs) developed by LMMs-Lab. This is a Python package focused on highly efficient training of multimodal models with support for various architectures and training paradigms.

## Configuration System

Training is configured via YAML files. YAML is recommended for better readability and comment support.

### YAML Configuration (Recommended)

```yaml
# Configuration supports comments in YAML
- type: trainer
  config:
    trainer_type: hf_trainer
    dataset_config:
      # Dataset configuration
      dataset_name: "example_dataset"
    model_config:
      # Model configuration  
      model_name: "example_model"
    # TrainingArguments parameters
    output_dir: "./output"
    num_train_epochs: 3
```

## Development Commands

### Training Launch

```bash
# Recommended: torchrun with YAML config
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr="127.0.0.1" --master_port="8000" \
  -m lmms_engine.launch.cli config_yaml=path/to/config.yaml

# Hydra overrides (no config file needed)
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr="127.0.0.1" --master_port="8000" \
  -m lmms_engine.launch.cli \
  trainer_type=fsdp2_trainer \
  model_config.load_from_pretrained_path="model/path" \
  trainer_args.output_dir="./output" \
  trainer_args.num_train_epochs=3
```

### Development Setup

```bash
# Install with development dependencies
uv pip install -e ".[all]"  # Includes preference learning and storage

# Performance optimizations
uv pip install flash-attn --no-build-isolation
uv pip install liger-kernel
```

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale
