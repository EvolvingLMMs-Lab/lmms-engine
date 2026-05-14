#!/bin/bash

################################################################################
# LLaVA-OneVision2 (8B-Instruct) Training with FSDP2
################################################################################
#
# DESCRIPTION:
#   Train the LMMs-Lab LLaVA-OneVision2 checkpoint with FSDP2, sequence
#   packing (rmpad), and Liger fused linear cross-entropy.
#
# KEY NOTES:
#   - OV2 ships its modeling + processor code via auto_map. We forward
#     trust_remote_code through AutoConfig / AutoModelFor*ImageTextToText
#     so the remote code path is honored. The yaml sets:
#       model_config.extra_kwargs.trust_remote_code: true
#   - Inner LM is stock Qwen3, so most liger / rmpad work is delegated to
#     the qwen3 monkey patch. OV2-specific bits (outer model.forward,
#     vision LayerNorm, video token expansion) live under
#     ``src/lmms_engine/models/llava_onevision2``.
#   - Video frames go through the same image path as multi-image inputs;
#     the data processor rewrites <video_pad> into per-frame
#     ``<X.X seconds><vision_start><image_pad>*n<vision_end>`` blocks.
#
# REQUIREMENTS:
#   - 8x GPUs (A100/H100 with 80GB recommended)
#   - flash-attn: pip install flash-attn --no-build-isolation
#   - liger-kernel: pip install liger-kernel
#
# DATASET:
#   OpenAI chat format (JSONL/Arrow/Parquet); see docs/user_guide/data_prep.md.
#
################################################################################

NGPUS=8

# Auto-accept trust_remote_code prompts triggered by transitive HF auto_*
# loads (the explicit kwarg we pass should already cover the main path).
export HF_HUB_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=1

torchrun --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12356 \
  -m lmms_engine.launch.cli \
  config_yaml=examples/llava_onevision2/example.yaml
