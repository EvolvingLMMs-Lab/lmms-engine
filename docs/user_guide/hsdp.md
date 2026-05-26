# HSDP (Hybrid Sharded Data Parallel)

## What it does

By default, FSDP2 in this repo shards model parameters across **all** ranks in
the world. Every forward pass triggers an all-gather over the entire world to
reassemble parameters, and every backward pass triggers a reduce-scatter over
the entire world to average gradients.

When training spans multiple nodes, those collectives have to traverse the
inter-node fabric (InfiniBand / RoCE / etc.), which is typically much slower
than the intra-node NVLink/NVSwitch.

HSDP splits the world into **shard groups** of a fixed size. Inside a shard
group, parameters are sharded as usual (cheap intra-node collectives). Across
shard groups, parameters are **replicated** — only the per-step gradient
reduce-scatter has to cross the slow link.

The typical mapping is one shard group per node: parameter all-gather stays on
NVLink, only gradient all-reduce crosses the inter-node link.

## Configuration

Set `hsdp_shard_size` inside `fsdp_config`:

```yaml
trainer_args:
  fsdp2: true
  fsdp_config:
    transformer_layer_cls_to_wrap: ["Qwen3VLTextDecoderLayer"]
    reshard_after_forward: true
    hsdp_shard_size: 8   # 8 GPUs per shard group
```

Semantics of `hsdp_shard_size`:

| Value | Meaning |
|---|---|
| `0` / unset | HSDP disabled. Default full-world FSDP (current behavior). |
| `1` | Rejected — would be pure DDP, which this trainer does not target. |
| `> 1` | Enabled. Each shard group has this many ranks; the world is split into `world_size / hsdp_shard_size` replicate groups. |

Constraints:

- `world_size` must be divisible by `hsdp_shard_size`.
- Special case `hsdp_shard_size == world_size` is **equivalent to plain FSDP**
  (one shard group, no replicate axis); prefer leaving it unset in that case.

## When to enable

HSDP is a **communication / memory trade-off**:

| `hsdp_shard_size` | Per-GPU parameter memory | Cross-group communication |
|---|---|---|
| small | larger (fewer shards) | less |
| large | smaller (more shards) | more |

Rules of thumb:

- **Multi-node, model fits per node** → set `hsdp_shard_size = ranks_per_node`.
  This is the headline HSDP win.
- **Single-node training** → don't enable HSDP. There's no slow link to avoid;
  full-world FSDP already keeps everything on NVLink.
- **Model does NOT fit per node** → keep HSDP disabled (or shard across
  multiple nodes via a larger `hsdp_shard_size`). HSDP cannot make a model fit
  that plain FSDP couldn't.

## Limitations (current implementation)

The first version supports HSDP only on the default FSDP2 path. The following
combinations are explicitly rejected with an assertion:

- **Tensor / Context / Expert parallel** (`tp_degree > 1`, `sp_ulysses_degree
  > 1`, `ep_degree > 1`): the per-model `parallelize` path does not yet plumb
  the 2D mesh through. Use `hsdp_shard_size = 0` with those.
- **EP**: same reason; not supported in this version.

The checkpoint merger (`python -m lmms_engine.merger`) already handles
multi-placement DTensors and is expected to work for HSDP checkpoints, but has
not been exhaustively validated end-to-end. If you hit issues merging an HSDP
checkpoint, please open an issue.

## How it works under the hood

When `hsdp_shard_size > 1`, the process group manager builds an independent 2D
device mesh:

```python
init_device_mesh(
    "cuda",
    (replicate_size, hsdp_shard_size),
    mesh_dim_names=("hsdp_replicate", "hsdp_shard"),
)
```

This mesh is exposed via `pgm.process_group_manager.fsdp_mesh` and passed to
every `fully_shard` call. FSDP2 dispatches on mesh rank:

- 1D mesh → plain FSDP (full-world shard).
- 2D mesh → HSDP (replicate on outer axis, shard on inner axis).

Resulting DTensor placements are `(Replicate(), Shard(0))` — replicated across
shard groups, sharded within each group.
