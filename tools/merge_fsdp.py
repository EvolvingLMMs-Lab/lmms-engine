import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoProcessor

from lmms_engine.mapping_func import create_model_from_pretrained
from lmms_engine.models import *


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
    parser.add_argument("--type", type=str, default="hf", choices=["hf", "fsdp2"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all checkpoints by averaging their weights",
    )
    return parser.parse_args()


def merge_hf_fsdp_checkpoint(input_dir: Path, model_path: str, merge: bool = False):
    model_cls = create_model_from_pretrained(model_path)
    checkpoint_folder = list(input_dir.glob("checkpoint-*"))
    # Find the latest checkpoint with the highest index
    if not checkpoint_folder:
        raise ValueError(f"No checkpoint found in {args.input_dir}")
    checkpoint_folder.sort(key=lambda x: int(x.name.split("-")[-1]))
    if args.step is not None:
        checkpoint_folder = [
            x for x in checkpoint_folder if int(x.name.split("-")[-1]) == args.step
        ]
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
    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
    else:
        model.save_pretrained(str(input_dir))
    return model


def load_one_checkpoint(checkpoint_path: Path):
    shard_state_dict = checkpoint_path / "pytorch_model_fsdp_0"
    total_shards = len(list(shard_state_dict.glob("*.pt")))

    model_state_dict_lst = [None] * total_shards

    def process_one_shard(rank: int, model_state_dict_lst: list):
        model_path = (
            shard_state_dict / f"model_world_size_{total_shards}_rank_{rank}.pt"
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(total_shards, os.cpu_count())) as executor:
        futures = [
            executor.submit(process_one_shard, rank, model_state_dict_lst)
            for rank in range(total_shards)
        ]
        for future in tqdm(futures, desc="Loading shards"):
            future.result()

    return model_state_dict_lst


def prepare_full_sd(model_state_dict_list):
    state_dict = {}
    for key in set(model_state_dict_list[0].keys()):
        state_dict[key] = []
        for model_state_shard in model_state_dict_list:
            tensor = model_state_shard.pop(key)
            state_dict[key].append(tensor._local_tensor.bfloat16())

    # Merge tensors
    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        # Assume all dp for now
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    return state_dict


def merge_fsdp2_checkpoint(input_dir: Path, model_path: str, merge: bool = False):
    checkpoint_folder = list(input_dir.glob("checkpoint-*"))
    # Find the latest checkpoint with the highest index
    if not checkpoint_folder:
        raise ValueError(f"No checkpoint found in {args.input_dir}")
    checkpoint_folder.sort(key=lambda x: int(x.name.split("-")[-1]))
    if args.step is not None:
        checkpoint_folder = [
            x for x in checkpoint_folder if int(x.name.split("-")[-1]) == args.step
        ]

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = str(input_dir)
    model_cls = create_model_from_pretrained(checkpoint_folder[-1])
    processor = AutoProcessor.from_pretrained(checkpoint_folder[-1])
    config = AutoConfig.from_pretrained(checkpoint_folder[-1])
    processor.save_pretrained(output_dir)
    if not merge:
        latest_checkpoint = checkpoint_folder[-1]
        print(f"Using latest checkpoint: {latest_checkpoint}")
        model_state_dict_lsts = [load_one_checkpoint(latest_checkpoint)]
    else:
        print(f"Merging {len(checkpoint_folder)} checkpoints")
        model_state_dict_lsts = [
            load_one_checkpoint(checkpoint) for checkpoint in checkpoint_folder
        ]

    full_state_dict_lst = [
        prepare_full_sd(model_state_dict_lst)
        for model_state_dict_lst in model_state_dict_lsts
    ]
    state_dict = full_state_dict_lst[0]
    for full_state_dict in full_state_dict_lst[1:]:
        for key in state_dict:
            state_dict[key] += full_state_dict[key]
        del full_state_dict
    for key in state_dict:
        state_dict[key] /= len(full_state_dict_lst)

    del full_state_dict_lst

    with init_empty_weights():
        model = model_cls.from_config(config)
    model.load_state_dict(state_dict, assign=True)
    model.save_pretrained(output_dir)
    return model


def main(args):
    input_dir = Path(args.input_dir)
    if args.type == "hf":
        merge_hf_fsdp_checkpoint(input_dir, args.model_name_or_class, args.merge)
    elif args.type == "fsdp2":
        merge_fsdp2_checkpoint(input_dir, args.model_name_or_class, args.merge)
    else:
        raise ValueError(f"Invalid type: {args.type}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
