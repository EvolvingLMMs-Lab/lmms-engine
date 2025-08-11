import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
from accelerate import init_empty_weights
from transformers import AutoProcessor

from lmms_engine.mapping_func import create_model_from_pretrained


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge FSDP shards into a single checkpoint."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the FSDP shards to merge.",
    )
    parser.add_argument("--model_name_or_class", type=str, default="")
    return parser.parse_args()


def main(args):
    model_path = args.model_name_or_class
    model_cls = create_model_from_pretrained(model_path)
    input_path = Path(args.input_dir)
    checkpoint_folder = list(input_path.glob("checkpoint-*"))
    # Find the latest checkpoint with the highest index
    if not checkpoint_folder:
        raise ValueError(f"No checkpoint found in {args.input_dir}")
    checkpoint_folder.sort(key=lambda x: int(x.name.split("-")[-1]))
    latest_checkpoint = checkpoint_folder[-1]
    print(f"Using latest checkpoint: {latest_checkpoint}")
    shard_state_dict = latest_checkpoint / "pytorch_model_fsdp_0"
    model = model_cls.from_pretrained(
        model_path,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    state_dict = {"model": model.state_dict()}
    dist_cp.load(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(shard_state_dict),
        no_dist=True,
    )
    model.load_state_dict(state_dict["model"])
    model.save_pretrained(str(input_path))


if __name__ == "__main__":
    args = parse_args()
    main(args)
