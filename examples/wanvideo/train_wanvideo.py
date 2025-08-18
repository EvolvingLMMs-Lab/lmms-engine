#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 WanVideo team. All rights reserved.
"""
Training script for WanVideo models using lmms-engine-mini.

Usage:
    # Single GPU training
    python train_wanvideo.py --config configs/wan2.1_t2v_1.3b.yaml
    
    # Multi-GPU training with torchrun
    torchrun --nproc_per_node=8 train_wanvideo.py --config configs/wan2.1_t2v_14b.yaml
    
    # Multi-GPU training with accelerate
    accelerate launch --config_file accelerate_config.yaml train_wanvideo.py --config configs/wan2.1_t2v_14b.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to import lmms_engine
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lmms_engine.launch.cli import main as launch_training

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train WanVideo models using lmms-engine-mini"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed configuration file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Prepare arguments for lmms-engine-mini CLI
    cli_args = [
        "--config", args.config,
    ]
    
    if args.local_rank >= 0:
        cli_args.extend(["--local_rank", str(args.local_rank)])
    
    if args.deepspeed:
        cli_args.extend(["--deepspeed", args.deepspeed])
    
    if args.resume:
        cli_args.append("--resume")
    
    if args.eval_only:
        cli_args.append("--eval_only")
    
    # Launch training using lmms-engine-mini
    logger.info(f"Starting WanVideo training with config: {args.config}")
    sys.argv = ["train_wanvideo.py"] + cli_args
    launch_training()


if __name__ == "__main__":
    main()
