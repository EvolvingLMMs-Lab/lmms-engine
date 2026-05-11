# Qwen3.5 Training

## Overview

Qwen3.5 is a hybrid Transformer + linear-attention (GatedDeltaNet) language model
with vision support unified in a single model class. Decoder layers alternate
between standard softmax attention and a linear-attention block backed by
[`fla.ops.gated_delta_rule`](https://github.com/fla-org/flash-linear-attention).

## Supported Features

| Feature | Support |
|---------|---------|
| **FSDP2** | ✅ |
| **USP / Ulysses SP** | ⚠️ broken on linear-attention layers — see below |
| **Liger Kernel** | ✅ (Liger RoPE intentionally disabled — see below) |
| **Packing (rmpad)** | ✅ |
| **VL stack (`Qwen3_5ForConditionalGeneration`)** | ✅ — requires `model_general_type` override (see below) |
| **ViT frame parallel** | ✅ — see [`vit_frame_parallel`](../user_guide/vit_frame_parallel.md) |
| **causal_conv1d fast path** | ✅ optional, ~4× warmup speedup |

## Choosing the model class (`model_general_type`)

The Qwen3.5 `AutoConfig` registers under **both** auto-classes:

- `AutoModelForCausalLM` → `Qwen3_5ForCausalLM` (text-only, `model_type="qwen3_5_text"`)
- `AutoModelForImageTextToText` → `Qwen3_5ForConditionalGeneration` (VL, `model_type="qwen3_5"`)

By default `from_pretrained` resolves to the causal-LM class, which means you
would silently train only the text tower. For VL training, **explicitly set
`model_general_type: image_text_to_text`** so the engine routes the load
through the right auto-class:

```yaml
model_config:
  load_from_pretrained_path: Qwen/Qwen3.5-4B
  attn_implementation: flash_attention_2
  model_general_type: image_text_to_text   # force the VL stack
```

For text-only training (or starting from the text-only checkpoint), set
`model_general_type: causal_lm` or omit the field.

## Liger Kernel notes

Liger is fully supported except for `rope`, which is **not yet available for
Qwen3.5** due to the hybrid attention layout (Gated DeltaNet + Gated
Attention). The patch raises `NotImplementedError` if `rope=True` is forced.
RMSNorm, SwiGLU, and fused linear cross-entropy are all on by default.

## Packed Linear Attention

When `use_rmpad=true`, lmms-engine monkey-patches `Qwen3_5GatedDeltaNet.forward`
with a packed-aware version (`linear_attn_forward` in
`src/lmms_engine/models/qwen3_5/qwen3_5_ops.py`) that:

1. Forwards `cu_seqlens` to `chunk_gated_delta_rule` so the recurrent state
   resets at every sample boundary. Without this the state leaks across the
   whole packed batch and training is silently wrong.
2. Forwards `seq_idx` to `causal_conv1d_fn` so the input depthwise conv does
   not bleed across sample boundaries.

### Required: `fla` (flash-linear-attention)

```bash
pip install flash-linear-attention
```

Required for the packed path; raises at runtime if missing.

### Optional but recommended: `causal_conv1d`

```bash
pip install causal-conv1d --no-build-isolation
```

If absent, the conv falls back to `nn.Conv1d`. This is functionally fine but:

- Up to `conv_kernel_size - 1` (= 3) tokens per sample boundary receive a
  small amount of cross-sample information through the conv receptive field.
  The recurrent attention itself remains correct.
- `nn.Conv1d` on long packed sequences has a ~150s first-step warmup tax in
  practice (cuDNN algorithm pick + autotune); the fused kernel avoids it.

In a 10-step rmpad smoke test on Qwen3.5-0.8B / 2× A6000:

| Configuration | Total time | First step |
|---|---|---|
| Without `causal_conv1d` | 197s | 156s |
| With `causal_conv1d` | 43s | 7s |

Loss / grad-norm trajectories match to ~1e-3 between the two configurations.

## Sequence Parallelism is currently broken

The Ulysses SP path in `qwen3_5_ops.py` only modifies the full-attention
layers; the linear-attention branch in `decoder_layer_forward` runs without
any SP-aware reshaping. As a result, with `sp_ulysses_degree > 1`:

- Each SP rank receives only `seq / sp_size` tokens of the packed sequence.
- The GatedDeltaNet layer treats this slice as a complete sequence and runs
  its recurrent state from zero on that fragment — **the recurrent state
  no longer accumulates across the full sequence**, which is mathematically
  inconsistent with the dense / no-SP forward.

Ulysses' all-to-all trick works for softmax attention because heads are
independent, so swapping (seq-shard, all-heads) ↔ (full-seq, head-shard)
recovers the exact computation. Linear attention's recurrent state has no
analogous decomposition along the head axis; the seq dimension is intrinsically
serial.

**Do not enable `sp_ulysses_degree > 1` on Qwen3.5 until this is fixed.** The
fix likely needs either (a) gather-then-scatter the full sequence around each
linear-attention layer, or (b) integrate `fla`'s `cp_context` ring-style CP.
Tracking work separately.

## Quick Start

Example training config (FSDP2, packed, VL stack, no SP):

```yaml
trainer_type: fsdp2_trainer

dataset_config:
  dataset_type: qwen3_vl_iterable
  processor_config:
    processor_name: "Qwen/Qwen3.5-4B"
    processor_type: qwen3_vl       # qwen3_vl processor is reused for qwen3.5
  packing: true
  packing_length: 32768

model_config:
  load_from_pretrained_path: "Qwen/Qwen3.5-4B"
  attn_implementation: flash_attention_2
  model_general_type: image_text_to_text   # force the VL stack
  monkey_patch_kwargs:
    patch_type: ["liger"]          # add "vit_frame_parallel" if frame counts are skewed

trainer_args:
  use_rmpad: true
  use_liger_kernel: true
  fsdp2: true
  fsdp_config:
    transformer_layer_cls_to_wrap: ["Qwen3_5DecoderLayer"]
  sp_ulysses_degree: 1            # MUST stay 1; see "SP is broken" above
  bf16: true
```

A reference smoke-test script lives at
[`test/train/qwen3_5/train_qwen3_5.py`](../../test/train/qwen3_5/train_qwen3_5.py).
