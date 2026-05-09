# CLAUDE.md - LMMs Engine Repository Guide

## Repository Overview

**LMMs Engine** is a training framework for Large Multimodal Models (LMMs) developed by LMMs-Lab. It is a Python package focused on highly efficient training of multimodal models with support for various architectures and training paradigms (FSDP2, sequence parallelism, expert parallelism, etc.).

## Configuration System

Training is configured via Hydra (`hydra-core` + `omegaconf`). The default config lives at `src/lmms_engine/launch/config/default_config.yaml`. Users typically:

1. Write a YAML config (see `examples/`) and pass it via `config_yaml=...`, **or**
2. Override individual fields directly on the command line via Hydra dotted-path syntax.

The top-level schema (flat, no list wrapper) is:

```yaml
trainer_type: fsdp2_trainer        # e.g. fsdp2_trainer, hf_trainer
dataset_config:                    # -> DatasetConfig
  dataset_type: qwen3_vl_iterable
  dataset_format: yaml
  dataset_path: data/debug.yaml
  processor_config:
    processor_name: Qwen/Qwen3-VL-8B-Instruct
    processor_type: qwen3_vl
  packing: true
  packing_length: 32000
model_config:                      # -> ModelConfig
  load_from_pretrained_path: Qwen/Qwen3-VL-8B-Instruct
  attn_implementation: flash_attention_2   # or "sdpa"
trainer_args:                      # -> TrainingArguments (HF + lmms-engine extras)
  output_dir: ./output/run
  per_device_train_batch_size: 1
  learning_rate: 2.0e-04
  num_train_epochs: 1
  bf16: true
  fsdp2: true
  use_rmpad: true
  sp_ulysses_degree: 1
  ep_degree: 1
```

See `examples/qwen3_vl/example_config.yaml`, `examples/qwen3_vl_moe/qwen3_vl_moe_ep8.yaml`, etc. for full templates.

## Development Commands

### Training Launch (Hydra override is recommended)

Prefer **option 1 (pure Hydra override)** or **option 2 (file + override)**. Loading a YAML file alone is supported but discouraged — overriding individual fields on the command line is more reproducible and composes better with sweeps.

```bash
# 1) Hydra override only (recommended; defaults from default_config.yaml)
torchrun --nproc_per_node=8 -m lmms_engine.launch.cli \
  trainer_type=fsdp2_trainer \
  model_config.load_from_pretrained_path=Qwen/Qwen3-VL-8B-Instruct \
  model_config.attn_implementation=flash_attention_2 \
  trainer_args.output_dir=./output/debug \
  trainer_args.bf16=true \
  trainer_args.fsdp2=true \
  trainer_args.use_rmpad=true

# 2) Load a config file, then override individual fields (recommended)
torchrun --nproc_per_node=8 -m lmms_engine.launch.cli \
  --config-path examples/qwen3_vl \
  --config-name example_config \
  trainer_args.learning_rate=1.0e-05 \
  trainer_args.output_dir=./output/sweep_lr1e5

# 3) Pure YAML config (works, but no per-run overrides)
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=8000 \
  -m lmms_engine.launch.cli \
  --config-path examples/qwen3_vl \
  --config-name example_config
```

Hydra dotted paths map directly to YAML nesting (`trainer_args.fsdp_config.reshard_after_forward=true`). `--config-path` takes a directory (absolute or relative to CWD), `--config-name` is the YAML basename without extension.

There is also a legacy `config_yaml=path/to/config.yaml` field that loads a YAML and merges it on top of the defaults — it still works, but `--config-path` / `--config-name` is the preferred Hydra-native way.

### Checkpoint Merging

FSDP2 saves sharded checkpoints; merge them into a single HF-compatible checkpoint with `lmms_engine.merger`:

```bash
# Merge a regular checkpoint
python -m lmms_engine.merger --checkpoint_path ./output/run/checkpoint-1000

# Merge EMA weights
python -m lmms_engine.merger \
  --checkpoint_path ./output/run/checkpoint-1000 \
  --checkpoint_type ema \
  --output_path ./output/run/checkpoint-1000-ema-merged
```

Implementation: `src/lmms_engine/merger/` — `base.CheckpointMerger` (ABC) + `fsdp2.FSDP2Merger`. Pass a parent dir and the latest `checkpoint-*` is auto-detected.

### Development Setup

```bash
uv pip install -e ".[all]"
uv pip install flash-attn --no-build-isolation   # CUDA only
uv pip install liger-kernel
```

## Attention Backend

Use `lmms_engine.kernels.attention.varlen_attn` in new model `*_ops.py` files instead of importing `flash_attn` directly. It dispatches to either FA2 or PyTorch SDPA based on the HF config:

```python
from lmms_engine.kernels.attention import varlen_attn

attn_output = varlen_attn(
    q=query_states, k=key_states, v=value_states,
    cu_seqlens_q=cu_seq_lens, cu_seqlens_k=cu_seq_lens,
    max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
    causal=True, softmax_scale=self.head_dim**-0.5,
    backend=self.config._attn_implementation,   # "flash_attention_2" | "sdpa"
)
```

Backend keys match HF `_attn_implementation` strings, so no translation layer is needed. This keeps the codebase portable to environments without flash-attn (e.g. ROCm without the ROCm fork installed). Existing reference implementations: `src/lmms_engine/models/qwen3/qwen3_ops.py`, `qwen3_vl/qwen3_vl_ops.py`, `qwen3_vl_moe/qwen3_vl_moe_ops.py`.

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
- **Build Iteratively**: Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Clean Logic**: Keep core logic clean and push implementation details to the edges
- **File Organisation**: Balance file organization with simplicity — use an appropriate number of files for the project scale
