"""CLI entry point for checkpoint merger.

Usage:
    python -m lmms_engine.merger --checkpoint_path checkpoint-1000
    python -m lmms_engine.merger --checkpoint_path /path/to/output --checkpoint_type ema
    # Can also pass parent directory - will use the latest checkpoint automatically
"""

import argparse
from pathlib import Path

from lmms_engine.merger import FSDP2Merger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge FSDP sharded checkpoints into a single checkpoint")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the sharded checkpoint directory (e.g., checkpoint-1000)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save merged checkpoint. If not specified, saves to checkpoint_path directly",
    )

    parser.add_argument(
        "--checkpoint_type",
        type=str,
        default="regular",
        choices=["regular", "ema"],
        help="Type of checkpoint to merge: 'regular' for main model weights, 'ema' for EMA weights",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    output_path = Path(args.output_path) if args.output_path else None

    print(f"Merging {args.checkpoint_type} checkpoint from {checkpoint_path}")
    merger = FSDP2Merger(checkpoint_type=args.checkpoint_type)
    result_path = merger.merge(checkpoint_path, output_path=output_path)

    print(f"Merged checkpoint saved to: {result_path}")


if __name__ == "__main__":
    main()
