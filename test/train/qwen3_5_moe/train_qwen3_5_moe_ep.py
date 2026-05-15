#!/usr/bin/env python
"""CI/CD smoke test: tiny qwen3_5_moe with Expert Parallelism.

Builds the model via `load_from_config` (random init) — no checkpoint
download. Top-level model is Qwen3_5MoeForConditionalGeneration with
model_type='qwen3_5_moe', so our liger/rmpad monkey patches and EP
parallelize fn dispatch correctly.

Usage:
    torchrun --nproc_per_node=8 test/train/qwen3_5_moe/train_qwen3_5_moe_ep.py \\
        --output_dir ./output/qwen3_5_moe_ep4 --ep_degree 4
"""
import argparse

from lmms_engine.launch.cli import create_train_task


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3.5 MoE model with Expert Parallelism")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for training")
    parser.add_argument("--ep_degree", type=int, default=2, choices=[2, 4, 8], help="Expert parallelism degree")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of training steps")
    parser.add_argument("--processor_name", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--nproc_per_node", type=int, default=None)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="8000")

    args, unknown = parser.parse_known_args()

    text_hidden_size = 256

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
                "processor_name": args.processor_name,
                "processor_type": "qwen3_vl",
            },
            "packing": False,
            "video_backend": "qwen_vl_utils",
        },
        "model_config": {
            "load_from_config": {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "hidden_size": text_hidden_size,
                    "intermediate_size": 512,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,
                    "num_experts": 8,
                    "num_experts_per_tok": 2,
                    "shared_expert_intermediate_size": 256,
                    "layer_types": [
                        "linear_attention",
                        "full_attention",
                        "linear_attention",
                        "full_attention",
                    ],
                    "head_dim": 32,
                    # match Qwen/Qwen3.6-35B-A3B tokenizer vocab (incl. image/video special tokens)
                    "vocab_size": 248320,
                },
                "vision_config": {
                    "depth": 2,
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "num_heads": 4,
                    "out_hidden_size": text_hidden_size,
                    "num_position_embeddings": 64,
                },
            },
            "attn_implementation": "flash_attention_2",
            "model_general_type": "image_text_to_text",
            "monkey_patch_kwargs": {
                "patch_type": ["liger", "rmpad"],
                "fused_linear_cross_entropy": True,
                "rms_norm": True,
                "swiglu": True,
            },
        },
        "trainer_args": {
            "per_device_train_batch_size": 1,
            "gradient_checkpointing": True,
            "num_train_epochs": 1,
            "max_steps": args.max_steps,
            "report_to": "none",
            "output_dir": args.output_dir,
            "warmup_ratio": 0.0,
            "eval_strategy": "no",
            "save_strategy": "no",
            "dataloader_num_workers": 2,
            "bf16": True,
            "lr_scheduler_type": "cosine",
            "use_liger_kernel": True,
            "use_rmpad": True,
            "fsdp2": True,
            "group_by_length": True,
            "fsdp_config": {
                "transformer_layer_cls_to_wrap": ["Qwen3_5MoeDecoderLayer"],
                "reshard_after_forward": False,
            },
            "ep_degree": args.ep_degree,
            "sp_ulysses_degree": 1,
        },
    }

    print(f"\n{'='*70}\nqwen3_5_moe EP test  ep={args.ep_degree}  steps={args.max_steps}\n{'='*70}\n")
    train_task = create_train_task(cfg)
    train_task.build()
    train_task.run()
    print(f"\n{'='*70}\nEP test completed successfully\n{'='*70}\n")


if __name__ == "__main__":
    main()
