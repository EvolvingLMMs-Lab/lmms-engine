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

        Uses each tensor's ``DTensor.placements`` to decide whether shards are
        sharded (concatenate along the sharding dim) or replicated (take one
        copy). Falls back to value equality for plain tensors that don't carry
        placement metadata.

        Args:
            shard_state_dicts: List of state dicts from each shard

        Returns:
            Full consolidated state dict
        """
        state_dict: dict = {}

        # Gather all tensor shards by key, remembering placements / global shape
        # for proper consolidation. We can't use byte-level equality because a
        # parameter that happens to be uniform after init (e.g. RMSNorm.weight
        # initialized to 1.0) is genuinely sharded but every shard has the
        # same values, so equality would silently drop 7/8 of its dim.
        placements_per_key: dict = {}
        global_shape_per_key: dict = {}
        for key in set(shard_state_dicts[0].keys()):
            shards: list[torch.Tensor] = []
            placements = None
            global_shape = None
            for model_state_shard in shard_state_dicts:
                tensor = model_state_shard.pop(key)
                if hasattr(tensor, "_local_tensor"):
                    if placements is None:
                        placements = tensor.placements
                        global_shape = tuple(tensor.shape)
                    shards.append(tensor._local_tensor.bfloat16())
                else:
                    # Plain tensor (e.g. inv_freq buffer): replicated implicitly.
                    shards.append(tensor.bfloat16())
            state_dict[key] = shards
            placements_per_key[key] = placements
            global_shape_per_key[key] = global_shape

        # Merge tensors using placements when available, otherwise fall back to
        # value equality for plain tensors.
        for key in sorted(state_dict):
            shards = state_dict[key]
            placements = placements_per_key[key]
            if placements is None:
                # Plain tensor (no DTensor metadata): replicated across ranks,
                # all shards should be equal — take one.
                state_dict[key] = shards[0]
                continue

            # Single placement (FSDP1D): handle Shard / Replicate / Partial.
            if len(placements) == 1:
                p = placements[0]
                if p.is_replicate():
                    state_dict[key] = shards[0]
                elif p.is_shard():
                    state_dict[key] = torch.cat(shards, dim=p.dim)
                else:
                    raise NotImplementedError(
                        f"Unsupported placement {p} for key '{key}' (only Shard / Replicate are handled)."
                    )
            else:
                # Multi-axis (e.g. HSDP / 2D mesh): not currently produced by
                # the trainer's FSDP2 setup. Fail loudly rather than silently
                # mis-consolidating.
                raise NotImplementedError(f"Multi-placement DTensor not supported: key='{key}' placements={placements}")

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

    def maybe_tie_weights(self, model: torch.nn.Module, config: object, state_dict: dict) -> None:
        """Re-tie weights if the model declares weight tying.

        FSDP saves tied parameters (e.g. ``lm_head`` <-> ``embed_tokens``) as
        independent shards, so after ``load_state_dict(..., assign=True)`` they
        become separate tensors and ``save_pretrained`` would write both.

        Only re-ties when the model declares tying AND the saved tensors
        actually agree, to avoid silently dropping divergent weights.
        """
        tied_keys_map = getattr(model, "_tied_weights_keys", None)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False) or getattr(
            getattr(config, "text_config", None), "tie_word_embeddings", False
        )
        if not (tied_keys_map and tie_word_embeddings):
            return

        if isinstance(tied_keys_map, dict):
            for tied_key, source_key in tied_keys_map.items():
                t1 = state_dict.get(tied_key)
                t2 = state_dict.get(source_key)
                if t1 is not None and t2 is not None and not torch.equal(t1, t2):
                    logger.warning(f"Tied weights mismatch: '{tied_key}' != '{source_key}'. Skipping tie_weights().")
                    return

        logger.info("Re-tying weights (tie_word_embeddings=True).")
        model.tie_weights()

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
        self.maybe_tie_weights(model, config, full_state_dict)
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
