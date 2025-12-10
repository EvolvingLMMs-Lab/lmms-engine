# LLaVA-OneVision 1.5 Training

## Overview

LLaVA-OneVision 1.5 is a vision-language model adapted for efficient FSDP2 training with lightweight data and fast startup.

## Supported Features

| Feature | Support |
|---------|---------|
| **FSDP2** | ✅ |
| **FlashAttention 2** | ✅ |
| **Liger Kernel** | ✅ |
| **RMPAD** | ✅ |
| **Packing** | ✅ |


## Quick Start

See the example configuration and run script:
See the example configuration and run script:
- **Example Config**: [examples/llava_onevision1_5/example.yaml](../../examples/llava_onevision1_5/example.yaml)
- **Run Script**: [examples/llava_onevision1_5/run.sh](../../llava_onevision1_5/run.sh)

## Key Configuration

```yaml
trainer_type: fsdp2_trainer

dataset_config:
  dataset_type: vision_iterable
  dataset_format: yaml
  datasets:
    - path: data/open_thoughts_debug
      data_folder: ''
      data_type: arrow
  processor_config:
    processor_name: Jinghao-Guo/llavaov1.5-4B-instruct-converted-qwen
    processor_type: llava
  packing: true
  shuffle: true
  video_backend: qwen_vl_utils

model_config:
  load_from_pretrained_path: Jinghao-Guo/llavaov1.5-4B-instruct-converted-qwen
  attn_implementation: flash_attention_2

trainer_args:
  per_device_train_batch_size: 1
  gradient_checkpointing: true
  num_train_epochs: 1
  max_steps: 1
  report_to: none
  output_dir: out
  warmup_ratio: 0.0
  eval_strategy: 'no'
  dataloader_num_workers: 1
  bf16: true
  lr_scheduler_type: cosine
  use_liger_kernel: true
  use_rmpad: true
  fsdp2: true
  group_by_length: true
  fsdp_config:
    transformer_layer_cls_to_wrap:
      - LLaVAOneVision1_5_DecoderLayer
      - RiceBlock
    reshard_after_forward: false
  sp_ulysses_degree: 1
  print_batch_input_steps: -1
```
