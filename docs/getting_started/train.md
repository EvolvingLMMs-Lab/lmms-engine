
## Train

To run training, prepare a YAML config. Below are two up-to-date examples that you can use as templates.

Following is an example config:

```yaml
trainer_type: fsdp2_trainer

# Dataset configuration - now includes the actual dataset definitions
dataset_config:
  dataset_type: vision
  dataset_format: yaml # Uses 'yaml' format for both external files and inline definitions

  # Inline dataset definitions (no dataset_path needed)
  datasets:
    - path: data/open_thoughts_debug
      data_folder: ""
      data_type: arrow

  # Processor configuration
  processor_config:
    processor_name: "Qwen/Qwen2.5-VL-7B-Instruct"
    processor_type: "qwen2_5_vl"

  # Packing configuration
  packing: true
  packing_strategy: first_fit
  packing_length: 16384

# Model configuration
model_config:
  load_from_pretrained_path: "Qwen/Qwen2.5-VL-7B-Instruct"
  attn_implementation: "flash_attention_2"

# Training arguments, mostly compatible with HuggingFace Trainer
trainer_args:
  per_device_train_batch_size: 1
  learning_rate: 1.0e-06 # we should use 1.0 to makes YAML recognize it as a float
  weight_decay: 0.0
  gradient_accumulation_steps: 1
  gradient_checkpointing: true
  num_train_epochs: 1
  save_steps: 100
  save_total_limit: 1
  report_to: "wandb"
  output_dir: "./output/debug"
  warmup_ratio: 0.0
  run_name: "qwen2_5_vl_config"
  eval_strategy: "no"
  logging_steps: 1
  group_by_length: true
  dataloader_num_workers: 8
  bf16: true
  lr_scheduler_type: "cosine"
  freeze_modules: ["visual"]
  use_liger_kernel: true
  use_rmpad: true
  fsdp2: true
  fsdp_config:
    transformer_layer_cls_to_wrap: ["Qwen2_5_VLDecoderLayer"]
    reshard_after_forward: false
```

You can visit the `config.py` file under each subfolder to see what parameters are configurable

### Key fields

- **trainer_type**: Use `hf_trainer` for standard HF Trainer or `fsdp2_trainer` for PyTorch FSDP2.
- **dataset_config.dataset_format**: `yaml`. You can either set `dataset_path` to an external YAML, or embed datasets inline via `datasets`.
- **datasets**: Each entry defines `path`, optional `data_folder`, and `data_type` (e.g., `arrow`, `parquet`).
- **processor_config**: Set `processor_name` (e.g., a Hugging Face model id) and `processor_type` (e.g., `qwen2_5_vl`).
- **packing**: Enable sequence packing with `packing: true`, and adjust `packing_strategy` and `packing_length`. Use `filter_overlong` to drop samples exceeding limits.
- **video options**: `video_backend`, `video_sampling_strategy`, `video_max_pixels`, `video_max_frames` control video preprocessing.
- **model_config**: Prefer `load_from_pretrained_path` and set `attn_implementation` (e.g., `flash_attention_2`).
- **freeze_modules**: List of submodules (e.g., `visual`) to freeze during training.
- **use_liger_kernel/use_rmpad**: Performance optimizations. Keep enabled if supported on your stack.
- **fsdp2/fsdp_config**: Enable FSDP2 sharding and wrap transformer layer classes via `transformer_layer_cls_to_wrap`. Tune `reshard_after_forward` for memory/perf trade-offs.
- **EMA (Exponential Moving Average)**: Enable EMA with `ema_enabled: true`. Configure `ema_decay` (default 0.9999), `ema_update_every`, `ema_start_step`, and optionally filter parameters via `ema_param_filter`. EMA checkpoints are saved alongside regular checkpoints and can be merged using `merge_fsdp.py` with `--state_dict_dirname pytorch_ema_model_fsdp_0`.

## Run

Example launch command:

```bash
export NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false

# Hugging Face setup (optional)
export HF_TOKEN="<YOUR HF_TOKEN>"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER="1"

export NCCL_DEBUG=INFO

CONFIG=$1  # path to your YAML config

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="8000" \
    -m lmms_engine.launch.cli config_yaml=${CONFIG}
```

## Run direct with cli and override with hydra

Instead of using a YAML config file, you can pass configuration directly via Hydra overrides on the command line. This is useful for quick experiments and parameter tuning.

### Basic Usage

Use the format `key=value` to override any configuration parameter. Hydra automatically creates the nested structure:

```bash
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="8000" \
    -m lmms_engine.launch.cli \
    trainer_type=fsdp2_trainer \
    dataset_config.dataset_path=/path/to/video_dataset.yaml \
    dataset_config.dataset_format=yaml \
    dataset_config.dataset_type=qwen3_vl_iterable \
    dataset_config.processor_config.processor_name="Qwen/Qwen3-VL-8B-Instruct" \
    dataset_config.processor_config.processor_type=qwen3_vl \
    model_config.load_from_pretrained_path="Qwen/Qwen3-VL-8B-Instruct" \
    model_config.attn_implementation=flash_attention_2 \
    trainer_args.per_device_train_batch_size=1 \
    trainer_args.learning_rate=2.0e-04 \
    trainer_args.num_train_epochs=1 \
    trainer_args.output_dir=./output/debug \
    trainer_args.bf16=true
```

### Common Overrides

Here are frequently used parameters you can override:

**Dataset Configuration:**
- `dataset_config.dataset_path`: Path to your YAML dataset config
- `dataset_config.dataset_format`: Format type (e.g., `yaml`, `json`)
- `dataset_config.dataset_type`: Dataset type (e.g., `vision`, `qwen3_vl_iterable`)
- `dataset_config.processor_config.processor_name`: Model name for the processor
- `dataset_config.processor_config.processor_type`: Processor type to use
- `dataset_config.packing`: Enable/disable sequence packing (e.g., `packing=true`)
- `dataset_config.packing_length`: Max sequence length for packing
- `dataset_config.video_backend`: Video processing backend (e.g., `qwen_vl_utils`)
- `dataset_config.video_sampling_strategy`: Video sampling method (e.g., `fps`)
- `dataset_config.video_max_frames`: Maximum frames per video

**Model Configuration:**
- `model_config.load_from_pretrained_path`: Path or HF model ID to load from
- `model_config.attn_implementation`: Attention implementation (e.g., `flash_attention_2`)

**Training Arguments:**
- `trainer_args.per_device_train_batch_size`: Batch size per device
- `trainer_args.learning_rate`: Learning rate (use float notation like `2.0e-04`)
- `trainer_args.num_train_epochs`: Number of training epochs
- `trainer_args.max_steps`: Maximum training steps
- `trainer_args.gradient_accumulation_steps`: Gradient accumulation steps
- `trainer_args.gradient_checkpointing`: Enable gradient checkpointing
- `trainer_args.output_dir`: Output directory for checkpoints
- `trainer_args.run_name`: Name for this training run
- `trainer_args.bf16`: Use bfloat16 precision
- `trainer_args.fsdp2`: Enable FSDP2 distributed training
- `trainer_args.use_liger_kernel`: Enable Liger kernel optimizations
- `trainer_args.use_rmpad`: Enable padding removal optimization
- `trainer_args.ema_enabled`: Enable EMA (default: `false`)
- `trainer_args.ema_decay`: EMA decay rate (default: `0.9999`)
- `trainer_args.ema_update_every`: Update EMA every N steps (default: `1`)
- `trainer_args.ema_start_step`: Start EMA from step N (default: `0`)
- `trainer_args.ema_requires_grad_only`: Only apply EMA to trainable parameters (default: `true`)
- `trainer_args.ema_param_filter`: Filter parameters by name (supports `mode`, `include`, `exclude`)
- `trainer_args.ema_resume_from_ema`: Resume training from EMA weights (default: `false`)

### Advanced Example

See `examples/qwen3_vl/qwen3_vl_8b_train.sh` for a complete training script using Hydra overrides with comprehensive parameter configuration for multi-GPU training.

### Overriding Existing YAML Config

You can use a YAML config file as a base and override specific parameters via CLI using Hydra's config-path and config-name:

```bash
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="8000" \
    -m lmms_engine.launch.cli \
    --config-path /path/to/config_yaml/directory \
    --config-name qwen2_5_vl_dp \
    trainer_args.max_steps=100
```

This loads all settings from `qwen2_5_vl_dp.yaml` in the specified directory and only overrides the specified parameters (CLI overrides take precedence).

### Tips

- Use quotes for string values: `processor_name="Qwen/Qwen2.5-VL-7B-Instruct"`
- Use dot notation for nested configs: `trainer_args.learning_rate=1.0e-06`
- Boolean values: `packing=true` or `packing=false`
- For complex values (lists/arrays), use Hydra's syntax: `trainer_args.fsdp_config.transformer_layer_cls_to_wrap=["Qwen2_5_VLDecoderLayer"]`
- Add new parameters with `+`: `+dataset_config.extra_kwargs.image_max_pixels=4194304`


