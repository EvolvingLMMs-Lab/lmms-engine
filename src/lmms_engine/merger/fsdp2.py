"""FSDP2 checkpoint merger implementation."""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import torch
from accelerate import init_empty_weights
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

from lmms_engine.mapping_func import create_model_from_pretrained
from lmms_engine.merger.base import CheckpointMerger
from lmms_engine.models import *

CheckpointType = Literal["regular", "ema"]

# Mapping from checkpoint type to subdirectory name
STATE_DICT_DIRNAME_MAP: dict[CheckpointType, str] = {
    "regular": "pytorch_model_fsdp_0",
    "ema": "pytorch_ema_model_fsdp_0",
}


class FSDP2Merger(CheckpointMerger):
    """Merger for FSDP2 sharded checkpoints.

    This class handles merging of FSDP2 sharded checkpoints into single
    consolidated checkpoints that can be loaded for evaluation or inference.

    Args:
        checkpoint_type: Type of checkpoint to merge - "regular" for the main
                        model weights, "ema" for exponential moving average weights

    Example:
        >>> from pathlib import Path
        >>> from lmms_engine.merger import FSDP2Merger
        >>> merger = FSDP2Merger(checkpoint_type="regular")
        >>> merger.merge(Path("checkpoint-1000"))
    """

    def __init__(self, checkpoint_type: CheckpointType = "regular") -> None:
        self.checkpoint_type = checkpoint_type
        self._state_dict_dirname = STATE_DICT_DIRNAME_MAP[checkpoint_type]

    def load_shards(self, checkpoint_path: Path) -> list[dict]:
        """Load all FSDP2 shards from a checkpoint directory.

        Args:
            checkpoint_path: Path to the checkpoint directory

        Returns:
            List of state dicts, one per shard

        Raises:
            ValueError: If shard directory or files are not found
        """
        shard_state_dict = checkpoint_path / self._state_dict_dirname

        if not shard_state_dict.exists():
            raise ValueError(f"Shard directory not found: {shard_state_dict}")

        shard_files = list(shard_state_dict.glob("*.pt"))
        if not shard_files:
            raise ValueError(f"No shard files found in {shard_state_dict}")

        total_shards = len(shard_files)
        model_state_dict_lst = [None] * total_shards

        def process_one_shard(rank: int, model_state_dict_lst: list) -> dict:
            model_path = shard_state_dict / f"model_world_size_{total_shards}_rank_{rank}.pt"
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state_dict_lst[rank] = state_dict
            return state_dict

        with ThreadPoolExecutor(max_workers=min(total_shards, os.cpu_count())) as executor:
            futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
            for future in tqdm(futures, desc="Loading shards"):
                future.result()

        return model_state_dict_lst

    def consolidate(self, shard_state_dicts: list[dict]) -> dict:
        """Consolidate sharded FSDP2 state dicts into a single full state dict.

        Args:
            shard_state_dicts: List of state dicts from each shard

        Returns:
            Full consolidated state dict
        """
        state_dict = {}

        # Gather all tensor shards by key
        for key in set(shard_state_dicts[0].keys()):
            state_dict[key] = []
            for model_state_shard in shard_state_dicts:
                tensor = model_state_shard.pop(key)
                # Non-sharded tensors (e.g. buffers like inv_freq) are plain Tensors,
                # while FSDP-sharded parameters are DTensors with _local_tensor.
                local = tensor._local_tensor if hasattr(tensor, "_local_tensor") else tensor
                state_dict[key].append(local.bfloat16())

        # Merge tensors along dim=0 (data parallel dimension)
        for key in sorted(state_dict):
            if not isinstance(state_dict[key], list):
                continue
            # Non-sharded tensors are duplicated across ranks; just take the first one
            if all(t.shape == state_dict[key][0].shape and torch.equal(t, state_dict[key][0]) for t in state_dict[key][1:]):
                state_dict[key] = state_dict[key][0]
            else:
                state_dict[key] = torch.cat(state_dict[key], dim=0)

        return state_dict

    def _resolve_checkpoint_path(self, path: Path) -> Path:
        """Resolve checkpoint path, handling parent directories with multiple checkpoints.

        If path is a parent directory containing checkpoint-* subdirectories,
        returns the latest checkpoint. Otherwise returns the path as-is.

        Args:
            path: Input path (may be checkpoint directory or parent directory)

        Returns:
            Resolved checkpoint directory path

        Raises:
            ValueError: If no checkpoints found
        """
        # Check if path is already a checkpoint directory
        shard_path = path / self._state_dict_dirname
        if shard_path.exists():
            return path

        # Check if path contains checkpoint subdirectories
        checkpoint_folders = list(path.glob("checkpoint-*"))
        if not checkpoint_folders:
            raise ValueError(f"No checkpoint directory or checkpoint-* subdirectories found in {path}")

        # Sort by checkpoint number and use the latest
        checkpoint_folders.sort(key=lambda x: int(x.name.split("-")[-1]))
        latest_checkpoint = checkpoint_folders[-1]
        return latest_checkpoint

    def merge(
        self,
        checkpoint_path: Path,
        output_path: Path | None = None,
        model_cls: type | None = None,
        config: object | None = None,
    ) -> Path:
        """Merge FSDP2 sharded checkpoint into a single consolidated checkpoint.

        Args:
            checkpoint_path: Path to sharded checkpoint directory or parent directory
                           containing checkpoint-* subdirectories
            output_path: Where to save merged checkpoint. If None, saves to checkpoint_path directly
            model_cls: Model class to instantiate. If None, infers from checkpoint_path
            config: Model config. If None, loads from checkpoint_path

        Returns:
            Path to the merged checkpoint directory

        Raises:
            ValueError: If checkpoint type directory is not found
        """
        # Resolve checkpoint path (handles parent directories with checkpoint-* subdirs)
        original_checkpoint_path = checkpoint_path
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)

        if output_path is None:
            output_path = original_checkpoint_path

        shard_path = checkpoint_path / self._state_dict_dirname
        logger.info(f"Selecting Checkpoint: {checkpoint_path} with state dict dirname: {self._state_dict_dirname}")
        if not shard_path.exists():
            raise ValueError(f"Checkpoint type '{self.checkpoint_type}' not found at {shard_path}")

        # Infer model class and config if not provided
        if model_cls is None:
            model_cls = create_model_from_pretrained(checkpoint_path)
        if config is None:
            config = AutoConfig.from_pretrained(checkpoint_path)

        # Load and consolidate shards
        model_state_dict_lst = self.load_shards(checkpoint_path)
        full_state_dict = self.consolidate(model_state_dict_lst)

        # Create model and load consolidated state dict
        with init_empty_weights():
            model = model_cls.from_config(config)
        model.load_state_dict(full_state_dict, assign=True)
        processor = AutoProcessor.from_pretrained(checkpoint_path)
        processor.save_pretrained(output_path)
        config.save_pretrained(output_path)
        # Save merged checkpoint
        model.save_pretrained(output_path)

        # Copy over any extra config files that AutoProcessor may not handle
        # (e.g. processor_config.json for custom processors)
        for extra_file in ["processor_config.json"]:
            src = checkpoint_path / extra_file
            if src.exists():
                shutil.copy2(src, output_path / extra_file)

        return output_path
