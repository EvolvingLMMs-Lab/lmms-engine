# Asynchronous Checkpoint Evaluation During Training

LMMs Engine supports asynchronous evaluation of model checkpoints during training. This allows you to evaluate your model without interrupting the training process, by submitting evaluation jobs to a separate LMMS-Eval server.

## Overview

When enabled, the training system:
1. Submits evaluation jobs to an LMMS-Eval server when checkpoints are saved
2. Continues training while evaluations run in the background
3. Polls for evaluation results periodically
4. Logs evaluation metrics when they become available

## Prerequisites

### Start the LMMS-Eval Server

You need to run the LMMS-Eval server before starting training. The server will handle evaluation requests and return results.

```bash
# Start the LMMS-Eval server on your evaluation machine
python -m lmms_eval.entrypoints.server --port 8000
```

The server will listen for evaluation requests and perform evaluations asynchronously.

## Configuration

Enable asynchronous evaluation in your training configuration YAML:

```yaml
trainer_args:
  # Enable evaluation at specific intervals
  eval_strategy: "steps"  # Options: "steps", "epoch", "no"
  eval_steps: 500  # Evaluate every N steps (when eval_strategy="steps")
  
  # Evaluation configuration
  eval_config:
    # Server configuration
    server_url: "http://192.168.8.249:8000"
    poll_interval: 10.0  # Poll server every 10 seconds
    
    # Model configuration
    model: "qwen_vl"  # Model name recognized by LMMS-Eval
    checkpoint_key: "model"  # Key to use in model_args for checkpoint path
    
    # Tasks to evaluate
    tasks:
      - "mmmu_val"
      - "textvqa_val"
      - "docvqa_val"
    
    # Model arguments passed to LMMS-Eval
    model_args:
      num_gpus: 8
      batch_size: 256
      max_length: 2048
      # Additional model-specific arguments
```

### Configuration Parameters

#### `eval_strategy`

- `"steps"`: Evaluate every `eval_steps` training steps
- `"epoch"`: Evaluate at the end of each epoch
- `"no"`: Disable evaluation (default)

#### `eval_config` Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `server_url` | string | URL of the LMMS-Eval server (e.g., `"http://localhost:8000"`) |
| `poll_interval` | float | Interval (seconds) to poll for evaluation results (default: `10.0`) |
| `model` | string | Model name recognized by LMMS-Eval (e.g., `"qwen_vl"`) |
| `tasks` | list | List of evaluation tasks (e.g., `["mmmu_val", "textvqa_val"]`) |
| `checkpoint_key` | string | Key used in model_args to specify checkpoint path |
| `model_args` | dict | Additional arguments passed to the model (e.g., `num_gpus`, `batch_size`) |

## How It Works

### 1. Checkpoint Saving

When a checkpoint is saved (according to `save_steps`), the trainer:
- Determines the checkpoint path (e.g., `./output/checkpoint-500`)
- Creates an evaluation output directory (e.g., `./output/checkpoint-500/eval`)
- Submits an evaluation job to the LMMS-Eval server

### 2. Background Polling

A background thread:
- Polls the LMMS-Eval server every `poll_interval` seconds
- Checks if evaluation jobs are completed
- Retrieves results when available

### 3. Metric Logging

When evaluation results are available:
- Metrics are logged to your tracking system (e.g., W&B, TensorBoard)
- Metrics include `global_step` to associate results with the training step
- Example logged metrics: `eval/mmmu_val/accuracy`, `eval/textvqa_val/accuracy`

### 4. Training Completion

At the end of training:
- The trainer waits for all pending evaluation jobs to complete
- All remaining evaluation results are logged
- Training exits only after all evaluations are finished

## Example Configuration

Here's a complete example with asynchronous evaluation enabled:

```yaml
trainer_type: fsdp2_trainer

dataset_config:
  dataset_type: vision
  dataset_format: yaml
  datasets:
    - path: data/your_dataset
      data_folder: ""
      data_type: arrow
  
  processor_config:
    processor_name: "Qwen/Qwen3-VL-8B-Instruct"
    processor_type: "qwen3_vl"
  
  packing: true
  packing_strategy: first_fit
  packing_length: 16384

model_config:
  load_from_pretrained_path: "Qwen/Qwen3-VL-8B-Instruct"
  attn_implementation: "flash_attention_2"

trainer_args:
  per_device_train_batch_size: 1
  learning_rate: 1.0e-06
  num_train_epochs: 1
  save_steps: 500
  eval_steps: 500  # Must equal save_steps for consistent evaluation
  eval_strategy: "steps"
  save_total_limit: 2
  
  # Evaluation configuration
  eval_config:
    server_url: "http://192.168.8.249:8000"
    poll_interval: 10.0
    checkpoint_key: "model"
    model: "qwen_vl"
    tasks:
      - "mmmu_val"
      - "textvqa_val"
    model_args:
      num_gpus: 8
      batch_size: 256
  
  report_to: "wandb"
  output_dir: "./output/qwen3_vl"
  bf16: true
  gradient_checkpointing: true
  fsdp2: true
  fsdp_config:
    transformer_layer_cls_to_wrap: ["Qwen3VLDecoderLayer"]
    reshard_after_forward: false
```

## EMA Checkpoint Evaluation

If you have EMA (Exponential Moving Average) enabled, the system will automatically evaluate both regular and EMA checkpoints:

```yaml
trainer_args:
  ema_enabled: true
  ema_decay: 0.9999
  ema_update_every: 1
  
  eval_config:
    server_url: "http://192.168.8.249:8000"
    # ... other config
```

The trainer will:
- Evaluate regular checkpoints with `checkpoint_type: "regular"`
- Evaluate EMA checkpoints with `checkpoint_type: "ema"`
- Log both sets of metrics separately

## Distributed Training

In distributed training (e.g., with `torchrun`), only rank 0:
- Submits evaluation jobs
- Polls for results
- Logs evaluation metrics

This avoids duplicate submissions and redundant logging.

## Monitoring Evaluation Progress

### Check W&B/TensorBoard

Evaluation metrics appear in your tracking dashboard:
- `eval/mmmu_val/accuracy`
- `eval/textvqa_val/accuracy`
- `eval/textvqa_val/anls`
- etc.

Each metric is associated with the training step via `global_step`.

### Check Evaluation Server Logs

The LMMS-Eval server logs:
- Received evaluation requests
- Evaluation progress
- Completed evaluations

### Check Training Logs

The training process logs:
- When evaluation jobs are submitted
- When results are received
- Any errors during polling or logging

## Troubleshooting

### Evaluations Not Starting

1. Verify the LMMS-Eval server is running at `server_url`
2. Check network connectivity from training machine to evaluation server
3. Verify the checkpoint path exists and contains valid weights

### Evaluation Results Not Appearing

1. Check `poll_interval` - increase if network is slow
2. Check LMMS-Eval server logs for errors
3. Verify task names are correct and supported by LMMS-Eval

### Duplicate Evaluations

Ensure `eval_steps` matches `save_steps` or adjust evaluation frequency to match checkpoint saving frequency.

## Best Practices

1. **Network Bandwidth**: Use a dedicated evaluation machine if network bandwidth is limited
2. **Resource Allocation**: Allocate sufficient GPUs for evaluation in `model_args.num_gpus`
3. **Checkpoint Frequency**: Balance between `save_steps` and evaluation frequency
4. **Task Selection**: Choose representative tasks that don't take too long
5. **Poll Interval**: Adjust `poll_interval` based on your network and evaluation speed
6. **Output Management**: Use `save_total_limit` to manage disk space for checkpoints

## Additional Resources

- [LMMS-Eval Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [Merge FSDP Checkpoints](merge_fsdp.md)
- [Training Guide](../getting_started/train.md)
