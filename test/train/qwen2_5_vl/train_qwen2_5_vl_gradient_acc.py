#!/usr/bin/env python3
"""
Training script for Qwen2.5-VL model.
This script is designed to be launched by torchrun for multi-GPU training.
"""

import argparse
import os
import sys

from lmms_engine.launch.cli import create_train_task


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for training")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Master address")
    parser.add_argument("--master_port", type=str, default="8000", help="Master port")

    args = parser.parse_args()

    # Training configuration
    cfg = {
        "trainer_type": "fsdp2_trainer",
        "dataset_config": {
            "dataset_type": "vision",
            "dataset_format": "yaml",
            "datasets": [
                {
                    "path": "data/open_thoughts_debug",
                    "data_folder": "",
                    "data_type": "arrow",
                }
            ],
            "processor_config": {
                "processor_name": "Qwen/Qwen2.5-VL-3B-Instruct",
                "processor_type": "qwen2_5_vl",
            },
            "packing": False,
            "video_backend": "qwen_vl_utils",
        },
        "model_config": {
            "load_from_pretrained_path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "attn_implementation": "flash_attention_2",
        },
        "trainer_args": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": True,
            "num_train_epochs": 1,
            "max_steps": 10,
            "report_to": "none",
            "output_dir": args.output_dir,
            "warmup_ratio": 0.0,
            "eval_strategy": "no",
            "dataloader_num_workers": 8,
            "bf16": True,
            "lr_scheduler_type": "cosine",
            "use_liger_kernel": True,
            "use_rmpad": True,
            "fsdp2": True,
            "group_by_length": True,
            "fsdp_config": {
                "transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer"],
                "reshard_after_forward": False,
            },
            "sp_ulysses_degree": 1,
            "print_batch_input_steps": -1,
        },
    }

    # Create and run training task
    train_task = create_train_task(cfg)
    train_task.build()
    train_task.run()


if __name__ == "__main__":
    main()
