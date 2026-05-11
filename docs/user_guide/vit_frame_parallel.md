# ViT Frame Parallel

For multimodal training with **variable per-sample frame counts** (typical of
mixed image + video batches), the per-rank ViT forward is the dominant
imbalance source: one rank may end up with a 64-frame video while another rank
processes a single image, and FSDP stalls all ranks waiting on the slowest.

ViT frame parallel redistributes the *frames themselves* across the DP group
before the ViT forward, then routes each frame's patch embeddings back to the
rank that owns the corresponding text sample.

## Mental Model

For each forward step:

1. Every rank announces how many frames / patches it would normally process.
2. An **LPT** (longest-processing-time first) bin-packing pass assigns frames
   to ranks so that the per-rank patch count is balanced.
3. **`input_dispatch`** ships frames to their assigned compute rank via a
   single `all_to_all_single` (autograd-aware), so each rank now runs the ViT
   on a roughly equal number of patches.
4. The ViT forward runs unchanged.
5. **`output_dispatch`** ships the ViT outputs (`last_hidden_state` and
   `pooler_output`) back to the rank that owns the original sample, again
   via a single `all_to_all_single`.

The wrap is implemented as a thin plumbing layer
(`lmms_engine.parallel.vit_parallel.frame_parallel.wrap_vit_forward`) that
takes three callables — `input_dispatch`, `orig_forward`, `output_dispatch` —
so the model-specific dispatch logic stays in the corresponding model
directory.

## When It Helps

| Scenario | Net effect |
|---|---|
| Single node, NVLink, video + image mix | Almost always a win — all-to-all is cheap on NVLink, idle time savings are large. |
| Multi-node, IB, large variance in frame counts | Usually a clear win. Communication is on the order of 100 ms per step; idle time saved is often 1–3 s. |
| Multi-node, very uniform frame counts (e.g. fixed 8 frames everywhere) | Marginal or net negative. Imbalance is small; the extra communication may outweigh the savings. |
| `dp_world_size <= 1` | No-op. The patch logs and returns immediately. |

If your batches are already balanced (every sample has roughly the same
number of patches), the all-to-all overhead is pure cost. Inspect step-time
variance across ranks before enabling on uniform workloads.

## Enabling It

Add `vit_frame_parallel` to `model_config.monkey_patch_kwargs.patch_type`:

```yaml
model_config:
  load_from_pretrained_path: Qwen/Qwen3.5-4B
  attn_implementation: flash_attention_2
  monkey_patch_kwargs:
    patch_type: ["liger", "vit_frame_parallel"]
```

The patch must be registered for the target model. Currently registered:

- `qwen3_5` (`Qwen3_5VisionModel.forward`)

To add support for a new model, drop an `<model>_vit_ops.py` that exports
`input_dispatch(self, ...) -> (dispatched_inputs, ctx)` and
`output_dispatch(ctx, ...) -> outputs`, and register a patch under
`@MONKEY_PATCHER.register("<model>", "vit_frame_parallel")` that calls
`wrap_vit_forward`. See `src/lmms_engine/models/qwen3_5/qwen3_5_vit_ops.py`
and `src/lmms_engine/models/qwen3_5/monkey_patch.py` for the reference.

## Implementation Notes

- The LPT planner lives in `lmms_engine.parallel.vit_parallel.balance` and is
  a pure algorithm (no rank / comm knowledge), making it easy to unit-test
  against synthetic load distributions.
- Frames are physically reordered on the source rank (via an `argsort`-based
  permutation) before the `all_to_all_single` so that each destination's
  contiguous slice corresponds to its assigned frames. The reverse permutation
  is applied in `output_dispatch`.
- Both ViT outputs (`last_hidden_state` at patch scale and `pooler_output` at
  patch / merge² scale) are shipped back. The LLM consumes `pooler_output`;
  `last_hidden_state` is shipped for completeness (loss-free with respect to
  downstream usage).
- The wrap reuses `pgm.process_group_manager.dp_group` so it composes
  naturally with FSDP2 sharding.
