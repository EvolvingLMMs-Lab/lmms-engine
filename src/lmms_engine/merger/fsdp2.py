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

        Uses each tensor's ``DTensor.placements`` and ``device_mesh`` to decide
        how to merge shards:

        * Single Shard / Replicate placement (plain FSDP2): concatenate along
          the sharding dim or take one copy.
        * Multi-placement DTensor (FSDP2 + Expert Parallel): each placement is
          one axis of the device mesh. We materialize the 2D mesh layout
          implied by ``mesh.shape`` + the global rank ordering (``stride``),
          then cat along inner mesh axes first and outer mesh axes last so the
          local-tensor index arithmetic matches DTensor semantics.

        For plain torch.Tensor entries (e.g. ``inv_freq`` buffers) we just
        take the first shard — they're replicated across ranks.

        Args:
            shard_state_dicts: List of state dicts from each shard

        Returns:
            Full consolidated state dict
        """
        state_dict: dict = {}

        # Gather all tensor shards by key, remembering placements / mesh shape
        # / mesh stride / global shape for proper consolidation. We can't use
        # byte-level equality because a parameter that happens to be uniform
        # after init (e.g. RMSNorm.weight initialized to 1.0) is genuinely
        # sharded but every shard has the same values, so equality would
        # silently drop 7/8 of its dim.
        placements_per_key: dict = {}
        mesh_shape_per_key: dict = {}
        mesh_stride_per_key: dict = {}
        global_shape_per_key: dict = {}
        for key in set(shard_state_dicts[0].keys()):
            shards: list[torch.Tensor] = []
            placements = None
            mesh_shape = None
            mesh_stride = None
            global_shape = None
            for model_state_shard in shard_state_dicts:
                tensor = model_state_shard.pop(key)
                if hasattr(tensor, "_local_tensor"):
                    if placements is None:
                        placements = tensor.placements
                        mesh = tensor.device_mesh
                        mesh_shape = tuple(mesh.shape)
                        global_shape = tuple(tensor.shape)
                        # mesh.mesh accessor requires an initialized PG (we
                        # don't have one when merging offline). DeviceMesh
                        # uses C-order (row-major) rank layout by default:
                        # last axis stride=1, then each preceding axis's
                        # stride = product of all following axes' sizes.
                        mesh_stride = tuple(
                            int(torch.tensor(mesh_shape[i + 1 :]).prod().item()) if i < len(mesh_shape) - 1 else 1
                            for i in range(len(mesh_shape))
                        )
                    shards.append(tensor._local_tensor.bfloat16())
                else:
                    # Plain tensor (e.g. inv_freq buffer): replicated implicitly.
                    shards.append(tensor.bfloat16())
            state_dict[key] = shards
            placements_per_key[key] = placements
            mesh_shape_per_key[key] = mesh_shape
            mesh_stride_per_key[key] = mesh_stride
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

            if len(placements) == 1:
                # Single placement (FSDP1D): handle Shard / Replicate / Partial.
                p = placements[0]
                if p.is_replicate():
                    state_dict[key] = shards[0]
                elif p.is_shard():
                    # When the mesh axis has size 1 (e.g. dp_shard_mod_ep=1
                    # for non-expert params in an EP-only config), each rank
                    # holds a full copy even though placement says Shard(dim).
                    # Detect this by comparing the global shape on the
                    # sharded dim with one local shard's: equal = no actual
                    # split, treat as replicate.
                    if shards[0].shape[p.dim] == global_shape_per_key[key][p.dim]:
                        state_dict[key] = shards[0]
                    else:
                        state_dict[key] = torch.cat(shards, dim=p.dim)
                else:
                    raise NotImplementedError(
                        f"Unsupported placement {p} for key '{key}' (only Shard / Replicate are handled)."
                    )
            else:
                # Multi-axis DTensor (FSDP2 + EP). Re-stitch by walking mesh
                # axes from inner-most (largest stride dim of the global rank
                # index, but mesh.mesh.stride() gives that ordering directly)
                # outward — at each step we group consecutive shards by the
                # current axis and cat along that placement's dim.
                state_dict[key] = self._consolidate_multi_axis(
                    shards,
                    placements,
                    mesh_shape_per_key[key],
                    mesh_stride_per_key[key],
                    key,
                )

        return state_dict

    @staticmethod
    def _consolidate_multi_axis(
        shards: list[torch.Tensor],
        placements: tuple,
        mesh_shape: tuple[int, ...],
        mesh_stride: tuple[int, ...],
        key: str,
    ) -> torch.Tensor:
        """Reduce a list of shards from a multi-axis DTensor into one tensor.

        Strategy: arrange shards into a nested list shaped like ``mesh_shape``
        using ``mesh_stride`` to map global rank -> multi-index, then
        recursively cat from inner-most axis outward. Each axis's Placement
        tells us which tensor dim to cat along (Shard.dim) or whether it's a
        no-op (Replicate / Partial).
        """
        from torch.distributed.tensor.placement_types import Replicate, Shard

        assert len(placements) == len(mesh_shape), (
            f"placements vs mesh_shape rank mismatch for key '{key}': "
            f"placements={placements} mesh_shape={mesh_shape}"
        )
        world_size = 1
        for s in mesh_shape:
            world_size *= s
        assert len(shards) == world_size, (
            f"shard count mismatch for key '{key}': got {len(shards)} shards, " f"mesh implies {world_size}"
        )

        def rank_to_index(rank: int) -> tuple[int, ...]:
            """Map a flat global rank to a multi-index on the device mesh.

            mesh_stride[i] is the stride for axis i: rank //= stride[i], then
            % shape[i] gives that axis's coordinate.
            """
            return tuple((rank // mesh_stride[i]) % mesh_shape[i] for i in range(len(mesh_shape)))

        # Build a nested-list grid indexed by mesh coords
        def make_grid(shape):
            if not shape:
                return None
            head, *tail = shape
            return [make_grid(tail) for _ in range(head)]

        grid = make_grid(mesh_shape)

        def grid_set(grid, idx, val):
            for i in idx[:-1]:
                grid = grid[i]
            grid[idx[-1]] = val

        def grid_get(grid, idx):
            for i in idx:
                grid = grid[i]
            return grid

        for rank, shard in enumerate(shards):
            idx = rank_to_index(rank)
            grid_set(grid, idx, shard)

        # Fold axes from inner-most (last) to outer-most (first). At each
        # axis, we cat together the inner sublists per placement[i].
        def fold(subgrid, axis: int) -> torch.Tensor:
            if axis == len(mesh_shape) - 1:
                # Leaf: subgrid is a list of tensors, one per coord on this axis.
                tensors = subgrid
            else:
                tensors = [fold(child, axis + 1) for child in subgrid]
            p = placements[axis]
            if isinstance(p, Replicate):
                return tensors[0]
            if isinstance(p, Shard):
                return torch.cat(tensors, dim=p.dim)
            raise NotImplementedError(f"Unsupported placement {p} on mesh axis {axis} for key '{key}'.")

        return fold(grid, axis=0)

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
        model_general_type: str | None = None,
    ) -> Path:
        """Merge FSDP2 sharded checkpoint into a single consolidated checkpoint.

        Args:
            checkpoint_path: Path to sharded checkpoint directory or parent directory
                           containing checkpoint-* subdirectories
            output_path: Where to save merged checkpoint. If None, saves to checkpoint_path directly
            model_cls: Model class to instantiate. If None, infers from checkpoint_path
            config: Model config. If None, loads from checkpoint_path
            model_general_type: Override AutoModel class (causal_lm /
                image_text_to_text / masked_lm / general). Forwarded to
                ``create_model_from_pretrained`` for configs registered under
                multiple AutoModel mappings.

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
            model_cls = create_model_from_pretrained(checkpoint_path, model_general_type=model_general_type)
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
