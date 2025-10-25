# Using the `merge_fsdp.py` Tool

The `merge_fsdp.py` script is a utility for merging Fully Sharded Data Parallel (FSDP) model checkpoints into a single consolidated checkpoint. This is particularly useful after training large models in a distributed setup. Below is a guide on how to use this tool effectively.

## Prerequisites

- Ensure you have Python installed along with the required dependencies.

- Make sure the FSDP checkpoints are available in the specified directory.

## Usage

The script can be executed with the following command:

```bash
python tools/merge_fsdp.py --input_dir <path_to_checkpoints> --model_name_or_class <model_name> --type <hf|fsdp2> [--output_dir <output_path>] [--step <checkpoint_step>] [--merge]
```

### Arguments

- `--input_dir`: Directory containing the FSDP shards to merge.
- `--model_name_or_class`: The name or class of the model to load. (Not required for fsdp2 type mergine)
- `--type`: Type of checkpoint to merge. Options are:
  - `hf`: Hugging Face checkpoint.
  - `fsdp2`: FSDP version 2 checkpoint.
- `--output_dir` (optional): Directory to save the merged checkpoint. Defaults to the input directory.
- `--step` (optional): Specific checkpoint step to merge. If not provided, the latest checkpoint is used.
- `--merge` (optional): Merge all checkpoints by averaging their weights.

### Example

To merge Hugging Face FSDP checkpoints:

```bash
python tools/merge_fsdp.py --input_dir ./checkpoints --model_name_or_class Qwen/Qwen2.5-VL-7B-Instruct --type hf --output_dir ./merged_checkpoint
```

To merge FSDP version 2 checkpoints:

```bash
python tools/merge_fsdp.py --input_dir ./checkpoints --type fsdp2
```

## Evaluation

After merging the checkpoints, you can evaluate the model using the `lmms-eval` tool. Refer to the [lmms-eval repository](https://github.com/EvolvingLMMs-Lab/lmms-eval) for detailed instructions on setting up and running evaluations.