# Merging FSDP Checkpoints

LMMs Engine provides multiple ways to merge Fully Sharded Data Parallel (FSDP) model checkpoints into single consolidated checkpoints. This is particularly useful after training large models in a distributed setup.

## Recommended: Using the Built-in Merger

The built-in merger module (`lmms_engine.merger`) is the recommended way to merge FSDP2 checkpoints. It's simpler, more flexible, and integrated directly into the framework.

### CLI Usage

The easiest way to merge checkpoints is using the CLI:

```bash
# Merge regular checkpoint
python -m lmms_engine.merger --checkpoint_path checkpoint-1000

# Merge EMA checkpoint
python -m lmms_engine.merger --checkpoint_path checkpoint-1000 --checkpoint_type ema

# Merge and save to different location
python -m lmms_engine.merger --checkpoint_path checkpoint-1000 --output_path ./merged_checkpoint

# Merge latest checkpoint (specify parent directory)
python -m lmms_engine.merger --checkpoint_path ./output/checkpoints
```

**CLI Arguments:**

- `--checkpoint_path`: Path to checkpoint directory or parent directory containing checkpoints
- `--output_path` (optional): Where to save merged checkpoint. Defaults to checkpoint_path
- `--checkpoint_type` (optional): Type of checkpoint - `"regular"` (default) or `"ema"`

### API Usage

For programmatic use, you can use the merger API:

```python
from pathlib import Path
from lmms_engine.merger import FSDP2Merger

# Create merger instance
merger = FSDP2Merger(checkpoint_type="regular")

# Merge checkpoint
checkpoint_path = Path("checkpoint-1000")
output_path = merger.merge(checkpoint_path)

print(f"Merged checkpoint saved to: {output_path}")

# Merge EMA checkpoint
merger_ema = FSDP2Merger(checkpoint_type="ema")
output_path_ema = merger_ema.merge(checkpoint_path)
```

**API Parameters:**

- `checkpoint_path`: Path to sharded checkpoint directory or parent directory
- `output_path` (optional): Where to save merged checkpoint. If None, saves to checkpoint_path
- `model_cls` (optional): Model class to instantiate. Auto-detected if not provided
- `config` (optional): Model config. Loaded from checkpoint if not provided

### Features

- **Automatic path resolution**: Pass parent directory to auto-select latest checkpoint
- **Regular and EMA checkpoints**: Support for both main model and EMA weights
- **Parallel loading**: Uses thread pool for efficient shard loading
- **Progress tracking**: Built-in progress bar during shard loading
- **Flexible output**: Save to original location or custom path

## Legacy: Using `merge_fsdp.py` Tool

The `merge_fsdp.py` script is a legacy utility that can still be used, but we recommend the built-in merger above.

```bash
python tools/merge_fsdp.py --input_dir <path_to_checkpoints> --model_name_or_class <model_name> --type <hf|fsdp2> [--output_dir <output_path>] [--step <checkpoint_step>] [--state_dict_dirname <dirname>] [--merge]
```

**Arguments:**

- `--input_dir`: Directory containing the FSDP shards to merge
- `--model_name_or_class`: The name or class of the model to load
- `--type`: Type of checkpoint (`hf` or `fsdp2`)
- `--output_dir` (optional): Directory to save the merged checkpoint
- `--step` (optional): Specific checkpoint step to merge
- `--state_dict_dirname` (optional): Subfolder name containing shards
- `--merge` (optional): Merge all checkpoints by averaging weights

**Examples:**

```bash
# Hugging Face FSDP checkpoints
python tools/merge_fsdp.py --input_dir ./checkpoints --model_name_or_class Qwen/Qwen2.5-VL-7B-Instruct --type hf --output_dir ./merged_checkpoint

# FSDP version 2 checkpoints
python tools/merge_fsdp.py --input_dir ./checkpoints --type fsdp2

# EMA checkpoints
python tools/merge_fsdp.py --input_dir ./checkpoints --type fsdp2 --state_dict_dirname pytorch_ema_model_fsdp_0
```

## Prerequisites

- Ensure you have Python installed along with the required dependencies
- Make sure the FSDP checkpoints are available in the specified directory
- For FSDP2 checkpoints, the checkpoint directory should contain:
  - `pytorch_model_fsdp_0/` for regular checkpoints
  - `pytorch_ema_model_fsdp_0/` for EMA checkpoints (if EMA is enabled)

## Evaluation

### Manual Evaluation

After merging the checkpoints, you can evaluate the model using the `lmms-eval` tool. Refer to the [lmms-eval repository](https://github.com/EvolvingLMMs-Lab/lmms-eval) for detailed instructions on setting up and running evaluations.

### Automatic Evaluation During Training

LMMs Engine also supports **asynchronous evaluation during training**, which automatically merges FSDP2 checkpoints and evaluates them without interrupting training. See [Asynchronous Checkpoint Evaluation](async_eval.md) for details.

#### How Automatic Merging Works

When using asynchronous evaluation, the system:

1. **Detects FSDP2 Checkpoints**: Automatically identifies FSDP2-sharded checkpoints during training
2. **Merges Before Evaluation**: The LMMS-Eval server handles merging of FSDP2 checkpoints using `lmms_engine_kwargs`
3. **Evaluates Merged Checkpoints**: Runs evaluation on the merged checkpoint
4. **Returns Results**: Evaluation results are polled and logged back to your training run

#### Configuration

Enable automatic merging and evaluation:

```yaml
trainer_args:
  eval_strategy: "steps"
  eval_steps: 500
  save_steps: 500  # Must match eval_steps
  
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
```

The LMMS-Eval server will automatically:
- Detect FSDP2 checkpoint format
- Merge the shards using the appropriate method
- Evaluate the merged checkpoint
- Return results to your training run

#### Benefits

- **No Manual Merging**: Checkpoints are merged automatically as part of evaluation
- **Non-Blocking**: Training continues while merging and evaluation happen in background
- **Distributed Evaluation**: Merge and evaluation run on a separate server, freeing training resources
- **Automatic Tracking**: Evaluation results are logged with the correct training step

See [Asynchronous Checkpoint Evaluation](async_eval.md) for complete documentation.