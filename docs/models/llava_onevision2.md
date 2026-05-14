# LLaVA-OneVision2 Training

## Overview

LLaVA-OneVision2 (OV2) is the LMMs-Lab successor to LLaVA-OneVision 1.5. The
8B-Instruct checkpoint pairs a custom OneVision vision encoder (SigLIP-like
ViT with 3D RoPE and a patch-merger) with a stock **Qwen3-8B** language
model. Modeling code is shipped via Hugging Face ``auto_map`` and is loaded
at runtime through ``trust_remote_code``.

## Supported Features

| Feature | Support |
|---------|---------|
| **FSDP2** | ✅ |
| **FlashAttention 2** | ✅ |
| **Liger Kernel** | ✅ |
| **RMPAD (sequence packing)** | ✅ |
| **Packing** | ✅ |
| **Ulysses Sequence Parallel** | ✅ (via Qwen3 inner LM) |

## Quick Start

- **Example Config**: [examples/llava_onevision2/example.yaml](../../examples/llava_onevision2/example.yaml)
- **Run Script**: [examples/llava_onevision2/run.sh](../../examples/llava_onevision2/run.sh)

```bash
bash examples/llava_onevision2/run.sh
```

## How Monkey Patching Works

Because OV2's modeling classes are loaded dynamically (no shared import
path), patches are applied at the **model instance** level. Two patch_types
are registered for ``model_type == "llava_onevision2"``:

* ``"liger"`` – Liger kernels: RoPE, RMSNorm, SwiGLU MLP (inner Qwen3 LM),
  LayerNorm (OV2 vision encoder), plus a fused linear cross-entropy bound
  onto OV2's ``ForConditionalGeneration.forward``.
* ``"rmpad"`` – Sequence-packing (unpadded) attention path: class-level
  patches to inner Qwen3 attention/decoder/model forwards so they consume
  ``cu_seq_lens``/``indices``, and an outer ``model_forward`` that wires
  rmpad metadata through to ``causal_lm_forward``.

The runner appends them in order based on ``trainer_args``:

| `use_liger_kernel` | `use_rmpad` | Resulting behaviour |
|---|---|---|
| ✅ | ✅ | rmpad + fused LCE (historical default) |
| ✅ | ❌ | fused LCE, no unpadding |
| ❌ | ✅ | unpadded attention, standard CE |
| ❌ | ❌ | stock HF forward |

## Key Configuration

```yaml
model_config:
  load_from_pretrained_path: lmms-lab-ov2/LLaVA-OneVision2-8B-Instruct
  attn_implementation: flash_attention_2
  torch_dtype: bfloat16
  model_type: llava_onevision2
  extra_kwargs:
    trust_remote_code: true        # required: OV2 ships modeling via auto_map

dataset_config:
  dataset_type: qwen3_vl_iterable
  processor_config:
    processor_name: lmms-lab-ov2/LLaVA-OneVision2-8B-Instruct
    processor_type: llava_onevision2
  packing: true
  packing_length: 8192

trainer_args:
  use_liger_kernel: true
  use_rmpad: true
  fsdp2: true
  fsdp_config:
    transformer_layer_cls_to_wrap:
      - Qwen3DecoderLayer            # inner LM (stock Qwen3)
      - OneVisionEncoderEncoderLayer # OV2 vision tower
```

## Data Processor

``LlavaOnevision2DataProcessor`` inherits from ``Qwen3_VLDataProcessor``
and:

1. Uses the OV2 ``AutoProcessor`` (image_processor + video_processor)
   loaded with ``trust_remote_code=True``.
2. Rewrites each chat-template ``<vision_start><video_pad><vision_end>``
   into a sequence of per-frame blocks
   ``<X.X seconds><vision_start><image_pad>*n<vision_end>`` and aliases the
   video patch tensors into the image path (every frame becomes a
   ``[1, H, W]`` row of ``image_grid_thw``).
3. Computes the block-layout ``patch_positions`` tensor required by the
   OV2 vision tower's 3D RoPE.
4. Normalizes per-frame numpy arrays from ``qwen_vl_utils`` (CHW float) to
   HWC uint8 so OV2's video processor can PIL-convert them.

## Implementation Pointers

* Monkey patches: ``src/lmms_engine/models/llava_onevision2/monkey_patch.py``
* OV2 forward replacements: ``src/lmms_engine/models/llava_onevision2/llava_onevision2_ops.py``
* Shared LM loss helper (LCE / CE, rmpad shift, Ulysses SP):
  ``src/lmms_engine/models/common_ops/loss.py``
* Data processor: ``src/lmms_engine/datasets/processor/llava_onevision2_processor.py``
