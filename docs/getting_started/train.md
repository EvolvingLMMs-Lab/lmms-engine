
## Train

To run training, prepare a YAML config. Below are two up-to-date examples that you can use as templates.

Following is an example config:

```yaml
- type: trainer
  config:
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

- **type/trainer**: Always `type: trainer` with a `config` block.
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
    -m lmms_engine.launch.cli --config ${CONFIG}
```
