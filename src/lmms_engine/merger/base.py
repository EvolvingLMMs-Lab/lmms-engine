"""Base class for checkpoint mergers."""

from abc import ABC, abstractmethod
from pathlib import Path


class CheckpointMerger(ABC):
    """Abstract base class for checkpoint mergers.

    This class defines the interface for merging sharded checkpoints from
    distributed training into single consolidated checkpoints.
    """

    @abstractmethod
    def merge(self, checkpoint_path: Path, output_path: Path | None = None, **kwargs) -> Path:
        """Merge checkpoint and return output path.

        Args:
            checkpoint_path: Path to the sharded checkpoint directory
            output_path: Where to save merged checkpoint. If None, saves to checkpoint_path
            **kwargs: Additional arguments specific to merger implementation

        Returns:
            Path to the merged checkpoint directory
        """
        pass
