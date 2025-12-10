#!/usr/bin/env python3
"""
Training script for LLaVA-OneVision 1.5 model.
Designed to be launched by torchrun for multi-GPU training.
"""

import argparse

from lmms_engine.launch.cli import create_train_task


def main():
    parser = argparse.ArgumentParser(description="Train LLaVA-OneVision 1.5 model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for training")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Master address")
    parser.add_argument("--master_port", type=str, default="8000", help="Master port")

    args = parser.parse_args()

    cfg = {
        "trainer_type": "fsdp2_trainer",
        "dataset_config": {
            "dataset_type": "vision_iterable",
            "dataset_format": "yaml",
            "datasets": [
                {
                    "path": "data/lmms_engine_test/text_example/open_thoughts_5k.parquet",
                    "data_folder": "",
                    "data_type": "parquet",
                }
            ],
            "processor_config": {
                "processor_name": "Jinghao-Guo/llavaov1.5-4B-instruct-converted-qwen",
                "processor_type": "llava",
            },
            "packing": False,
            "shuffle": False,
            "video_backend": "qwen_vl_utils",
        },
        "trainer_args": {
            "per_device_train_batch_size": 1,
            "gradient_checkpointing": True,
            "num_train_epochs": 1,
            "max_steps": 1,
            "report_to": "none",
            "output_dir": args.output_dir,
            "warmup_ratio": 0.0,
            "eval_strategy": "no",
            "dataloader_num_workers": 1,
            "bf16": True,
            "lr_scheduler_type": "cosine",
            "use_liger_kernel": True,
            "use_rmpad": True,
            "fsdp2": True,
            "group_by_length": True,
            "fsdp_config": {
                "transformer_layer_cls_to_wrap": [
                    "Qwen3DecoderLayer",
                    "RiceBlock",
                ],
                "reshard_after_forward": False,
            },
            "sp_ulysses_degree": 1,
            "print_batch_input_steps": -1,
        },
        "model_config": {
            "load_from_pretrained_path": "Jinghao-Guo/llavaov1.5-4B-instruct-converted-qwen",
            "attn_implementation": "flash_attention_2",
        },
        "extra_kwargs": {},
    }

    train_task = create_train_task(cfg)
    train_task.build()
    train_task.run()


if __name__ == "__main__":
    main()
