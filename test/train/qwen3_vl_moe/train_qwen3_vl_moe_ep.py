import argparse
import os
import sys

from lmms_engine.launch.cli import create_train_task


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3 VL MoE model with Expert Parallelism")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for training",
    )
    parser.add_argument(
        "--ep_degree",
        type=int,
        default=None,
        help="Expert parallelism degree",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum number of training steps for testing",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=None,
        help="Number of processes per node (auto-set from ep_degree if not specified)",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Rank of this node",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="127.0.0.1",
        help="Master address",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="8000",
        help="Master port",
    )

    args, unknown = parser.parse_known_args()

    if args.ep_degree is None:
        if args.nproc_per_node is not None:
            args.ep_degree = args.nproc_per_node
        else:
            args.ep_degree = 2

    cfg = {
        "trainer_type": "fsdp2_trainer",
        "dataset_config": {
            "dataset_type": "qwen3_vl_iterable",
            "dataset_format": "yaml",
            "datasets": [
                {
                    "path": "data/lmms_engine_test/text_example/open_thoughts_5k.parquet",
                    "data_folder": "",
                    "data_type": "parquet",
                }
            ],
            "processor_config": {
                "processor_name": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                "processor_type": "qwen3_vl",
            },
            "packing": False,
            "video_backend": "qwen_vl_utils",
        },
        "model_config": {
            "load_from_pretrained_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16",
            "monkey_patch_kwargs": {
                "patch_type": ["liger"],
                "fused_linear_cross_entropy": True,
                "rms_norm": True,
                "layer_norm": True,
                "swiglu": True,
            },
        },
        "trainer_args": {
            "per_device_train_batch_size": 1,
            "gradient_checkpointing": True,
            "num_train_epochs": 1,
            "max_steps": args.max_steps,
            "learning_rate": 1e-5,
            "report_to": "none",
            "output_dir": args.output_dir,
            "warmup_ratio": 0.0,
            "eval_strategy": "no",
            "save_strategy": "no",
            "dataloader_num_workers": 8,
            "bf16": True,
            "lr_scheduler_type": "cosine",
            "use_liger_kernel": True,  # Enable Liger kernel optimizations
            "use_rmpad": True,  # Enable RMPad for efficient padding
            "fsdp2": True,  # Use FSDP2 for distributed training
            "group_by_length": True,
            "use_muon": True,  # Test Muon optimizer with MoE
            "weight_decay": 0.01,
            "fsdp_config": {
                "transformer_layer_cls_to_wrap": [
                    "Qwen3VLMoeTextDecoderLayer",  # Text decoder layers with MoE
                    "Qwen3VLMoeVisionBlock",  # Vision encoder blocks
                ],
                "reshard_after_forward": False,
            },
            # Expert Parallelism configuration
            "ep_degree": args.ep_degree,  # Number of GPUs to distribute experts across
            "sp_ulysses_degree": 1,  # No sequence parallelism
        },
    }

    print(f"\n{'='*70}")
    print(f"Qwen3 VL MoE Expert Parallelism Test")
    print(f"{'='*70}")
    print(f"EP Degree: {args.ep_degree}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Batch Size per Device: 1")
    print(f"Model: Qwen/Qwen3-VL-30B-A3B-Instruct")
    print(f"Optimizer: Muon (Testing MoE compatibility)")
    print(f"Liger Kernel: Enabled")
    print(f"RMPad: Enabled")
    print(f"FSDP2: Enabled")
    print(f"Expert Parallelism: Enabled (degree={args.ep_degree})")
    print(f"Sequence Parallelism: Disabled")
    print(f"{'='*70}\n")

    # Create and run training task
    train_task = create_train_task(cfg)
    train_task.build()
    train_task.run()

    print(f"\n{'='*70}")
    print(f"EP Test Completed Successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
