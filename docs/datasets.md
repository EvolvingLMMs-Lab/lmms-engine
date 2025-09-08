## Datasets and Packing: Naive vs Streaming

This guide shows how to use the dataset implementations and explains the difference between naive (precomputed) packing and streaming packing.

### Overview

- `MultiModalDataset` (naive): indexable dataset that optionally precomputes packing groups before training.
- `MultiModalIterableDataset` (streaming): iterable dataset that forms packed batches on the fly while iterating.

Both use `DatasetConfig` to control data format, packing, and filtering behavior.

### Quick start: choose a dataset

```python
from lmms_engine.datasets.config import DatasetConfig
from lmms_engine.datasets.naive.multimodal_dataset import MultiModalDataset
from lmms_engine.datasets.iterable.multimodal_iterable_dataset import MultiModalIterableDataset

# Shared config fields (examples)
cfg = DatasetConfig(
    dataset_type="vision",                 # or "vision_audio"
    dataset_format="hf_dataset",          # json | jsonl | yaml | hf_dataset | arrow | parquet
    dataset_path="your/hub_or_path",      # or use datasets=[...] for yaml inline
    processor_config={"processor_type": "your_processor"},
    shuffle=True,
    # Packing
    packing=True,
    packing_strategy="first_fit",         # naive only: first_fit | window_XX
    packing_length=32000,
    filter_overlong=True,                  # drop samples > packing_length
)

# Pick ONE dataset style
dataset = MultiModalDataset(cfg)                 # Naive (map-style)
# dataset = MultiModalIterableDataset(cfg)       # Streaming (iterable)

dataset.build()
collator = dataset.get_collator()

# Pass to the FSDP2 trainer
# trainer = FSDP2SFTTrainer(model, args, train_dataset=dataset, data_collator=collator)
# trainer.train()
```

### Naive packing (precompute, map-style)

- Loads all samples, optionally shuffles, estimates per-sample lengths, and precomputes packing groups upfront.
- Implements packing via `_pack_by_first_fit` or `_pack_by_window` using `config.packing_length`.
- `__len__` returns number of packs when packing is enabled; `__getitem__` returns a list of samples for a given pack.
- `filter_overlong=True` removes samples whose estimated length exceeds `packing_length` before packing.

Best when:
- You can afford a preprocessing pass and want deterministic, precomputed packs.
- You need full control over packing strategies (e.g., windowed packing).

Trade-offs:
- Startup time and memory overhead for length estimation and pack computation.
- Not ideal for true streaming or extremely large datasets where scanning upfront is expensive.

### Streaming packing (on-the-fly, iterable)

- Requires `HFDataset` data source (enforced). Performs rank sharding and worker splitting at iteration time.
- Packs while iterating: keeps a buffer of samples and appends until the sum of `input_ids.shape[0]` would exceed `packing_length`, then yields the buffer and starts a new one.
- Behavior:
  - If `filter_overlong=True`, drops any single sample longer than `packing_length`.
  - If `filter_overlong=False` and a sample is longer than `packing_length`, yields it alone.
  - Flushes any remaining buffer at the end of the epoch.
- When `packing=False`, yields one processed sample at a time.

Best when:
- You want low-latency startup and true streaming behavior.
- Dataset is large and/or produced dynamically.
- You want to optimize your performance but don't want preprocess your data

Trade-offs:
- Packing is greedy per stream; no global optimality.
- Step count per rank/worker depends on sharding and filtering.
- Can not use certain lr scheduler since we do not know the total steps

### Distributed behavior differences

- Naive (map-style):
  - Trainer uses `DistributedSampler` or `DistributedLengthGroupedSampler` (`group_by_length=True`).
  - Steps per epoch are known (length is defined).

- Streaming (iterable):
  - Dataset does rank sharding (`HFDataset.shard`) and splits by worker with `torch.utils.data.get_worker_info()`.
  - Trainer does not attach a sampler for iterable datasets; steps per epoch are unknown.
  - Ensure your source `HFDataset` length divides reasonably across ranks to avoid imbalanced work.

### Configuration tips

- `packing_length`: max summed token length per pack (streaming uses actual `input_ids.shape[0]`).
- `filter_overlong`: set True to drop outliers (> `packing_length`) to keep batches consistent.
- Naive-only `packing_strategy`: `first_fit` or `window_{size}` for different packing heuristics.
- For streaming, prefer `hf_dataset` input for efficient sharding.

### Minimal YAML example

```yaml
dataset:
  dataset_type: vision
  dataset_format: hf_dataset
  dataset_path: your/hub_or_path
  shuffle: true
  packing: true
  packing_length: 32000
  filter_overlong: true
  processor_config:
    processor_type: your_processor
```

### FAQs

- Why can collectives hang in distributed runs?
  - Reduce operations must use tensors with identical shapes across ranks. In the trainer, aggregate scalar stats (e.g., sum/min/max of per-batch lengths) instead of reducing variable-length vectors.

- Which should I use?
  - Use naive packing for deterministic, globally planned packs when upfront preprocessing is acceptable. Use streaming packing for large-scale or online data where you want immediate iteration and on-the-fly grouping.


