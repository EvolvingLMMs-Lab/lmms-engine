#!/usr/bin/env python3
"""Training script for MiniCPM-o (simplex).

Self-contained test: generates a tiny image + audio dataset on the fly and
trains a randomly-initialized tiny ``minicpmo`` model for a few steps.  The
processor loads the tokenizer of the public checkpoint (a few MB).

This script is designed to be launched by torchrun for multi-GPU training.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np


def build_tiny_dataset(data_dir):
    """Write a tiny simplex dataset with one image and one audio sample."""
    from PIL import Image
    from soundfile import write as write_wav

    image_path = os.path.join(data_dir, "tiny.jpg")
    Image.fromarray((np.random.RandomState(0).rand(224, 224, 3) * 255).astype(np.uint8)).save(image_path)

    audio_path = os.path.join(data_dir, "tiny.wav")
    t = np.arange(16000) / 16000.0
    write_wav(audio_path, (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000)

    samples = []
    for i in range(4):
        samples.append(
            {
                "id": i,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "tiny.jpg"}},
                            {"type": "audio_url", "audio_url": {"url": "tiny.wav"}},
                            {"type": "text", "text": f"Describe sample {i}."},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": f"It is sample {i}."}]},
                ],
            }
        )

    data_path = os.path.join(data_dir, "data.json")
    with open(data_path, "w") as f:
        json.dump(samples, f)
    return data_path, data_dir


def main():
    parser = argparse.ArgumentParser(description="Train a tiny MiniCPM-o model (simplex).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", default="127.0.0.1")
    parser.add_argument("--master_port", default="8000")
    parser.add_argument("--dataset", default=None, help="Optional existing dataset JSON.")
    parser.add_argument("--dataset_folder", default=None)
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5", help="Processor/tokenizer source.")
    parser.add_argument("--trainer", default="fsdp2_trainer", help="Trainer type (fsdp2_trainer / hf_trainer).")
    parser.add_argument("--cpu", action="store_true", help="CPU debug mode (fp32, no fsdp).")
    args = parser.parse_args()

    from lmms_engine.launch.cli import create_train_task

    if args.dataset:
        data_path, data_folder = args.dataset, args.dataset_folder or ""
    else:
        data_path, data_folder = build_tiny_dataset(tempfile.mkdtemp(prefix="minicpmo_test_"))

    # A tiny randomly-initialized MiniCPM-o (no TTS head) keeps the CI test
    # fast while exercising registration, the data pipeline, the collator,
    # the forward and the loss.
    tiny_llm_config = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": True,
    }
    tiny_vision_config = {
        "hidden_size": 64,
        "image_size": 56,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "patch_size": 14,
    }
    tiny_audio_config = {
        "d_model": 64,
        "encoder_layers": 2,
        "encoder_attention_heads": 4,
        "encoder_ffn_dim": 256,
        "num_mel_bins": 80,
        "max_source_positions": 1500,
    }

    cfg = {
        "trainer_type": args.trainer,
        "dataset_config": {
            "dataset_type": "minicpmo_iterable",
            "dataset_format": "json",
            "dataset_path": data_path,
            "datasets": None,
            "processor_config": {
                "processor_name": args.model,
                "processor_type": "minicpmo",
                "extra_kwargs": {"max_slice_nums": 1, "use_audio_in_video": True},
            },
            "shuffle": False,
            "eval_dataset_path": None,
            "object_storage": "none",
            "bucket_name": None,
            "packing": False,
            "packing_strategy": "first_fit",
            "packing_length": 512,
            "filter_overlong": True,
            "filter_overlong_workers": 1,
            "max_length": None,
            "video_sampling_strategy": "fps",
            "video_max_pixels": 200704,
            "video_max_frames": 8,
            "frame_num": 4,
            "fps": 1.0,
            "video_backend": "decord",  # minicpmo decodes videos with PyAV internally
        },
        "model_config": {
            "load_from_pretrained_path": None,
            "load_from_config": {
                "model_type": "minicpmo",
                "config": {
                    "query_num": 32,
                    "image_size": 56,
                    "vision_batch_size": 4,
                    "vision_config": tiny_vision_config,
                    "audio_config": tiny_audio_config,
                    "slice_config": {"max_slice_nums": 1},
                    "init_tts": False,
                    **tiny_llm_config,
                },
            },
            "attn_implementation": "eager" if args.cpu else "sdpa",
            "torch_dtype": "float32" if args.cpu else "bfloat16",
            "overwrite_config": None,
            "monkey_patch_kwargs": None,
        },
        "trainer_args": {
            "output_dir": args.output_dir,
            "do_train": True,
            "do_eval": False,
            "eval_strategy": "no",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "num_train_epochs": 1,
            "max_steps": 5,
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "bf16": not args.cpu,
            "seed": 42,
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
            "gradient_checkpointing": True,
            "fsdp2": args.trainer == "fsdp2_trainer",
            "sp_ulysses_degree": 1,
            "reduce_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "fsdp_config": (
                {"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"], "reshard_after_forward": False}
                if args.trainer == "fsdp2_trainer"
                else None
            ),
            "freeze_modules": None,
            "use_rmpad": False,
            "use_liger_kernel": False,
        },
    }

    train_task = create_train_task(cfg)
    train_task.build()
    train_task.run()


if __name__ == "__main__":
    main()
