# Datasets and Packing: Naive vs Streaming

This guide explains the two dataset implementations in LMMS Engine and helps you choose the right approach for your training needs.

## Overview

LMMS Engine provides two distinct dataset implementations:

| Dataset Type | Class | Description | Best For |
|-------------|-------|-------------|----------|
| **Naive (Map-style)** | `MultiModalDataset` | Precomputes packing groups before training | Small to medium datasets, deterministic packing |
| **Streaming (Iterable)** | `MultiModalIterableDataset` | Packs sequences on-the-fly during iteration | Large datasets, low memory usage, dynamic data |

Both implementations share the same `DatasetConfig` interface for seamless switching between approaches.

## Quick Start

### Basic Usage

```python
from lmms_engine.datasets import DatasetConfig, MultiModalDataset, MultiModalIterableDataset
from lmms_engine.train import FSDP2SFTTrainer

# Configure your dataset
config = DatasetConfig(
    # Core settings
    dataset_type="vision",                    # Type: vision | vision_audio | fineweb_edu
    dataset_format="hf_dataset",              # Format: json | jsonl | yaml | hf_dataset | arrow | parquet
    dataset_path="your/dataset/path",         # Path to dataset or HF Hub ID
    
    # Processing
    processor_config={"processor_type": "your_processor"},
    shuffle=True,
    
    # Packing configuration
    packing=True,                              # Enable sequence packing
    packing_length=32000,                      # Maximum tokens per packed sequence
    filter_overlong=True,                      # Drop sequences > packing_length
    packing_strategy="first_fit",              # Naive only: first_fit | window_XX
)

# Choose your dataset implementation
# Option 1: Naive (precomputed packing)
dataset = MultiModalDataset(config)

# Option 2: Streaming (on-the-fly packing)  
dataset = MultiModalIterableDataset(config)

# Build and use
dataset.build()
collator = dataset.get_collator()

# Train with FSDP2
trainer = FSDP2SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator
)
trainer.train()
```

## Dataset Implementation Details

### Naive Dataset (Precomputed Packing)

The `MultiModalDataset` loads all data into memory and precomputes optimal packing arrangements before training begins.

#### How it works:
1. **Load**: Reads entire dataset into memory
2. **Estimate**: Calculates token length for each sample  
3. **Pack**: Groups samples using packing algorithm (`first_fit` or `window`)
4. **Serve**: Returns precomputed packs during training

#### Characteristics:
- ✅ **Deterministic**: Same packing arrangement every epoch
- ✅ **Optimal packing**: Can use sophisticated algorithms for better utilization
- ✅ **Known length**: Exact number of steps per epoch is known
- ❌ **Memory intensive**: Requires loading full dataset upfront
- ❌ **Slower startup**: Preprocessing adds initialization time

#### When to use:
- Dataset fits comfortably in memory (< 100GB)
- You need reproducible training runs
- Packing efficiency is critical
- You're debugging or experimenting

### Streaming Dataset (On-the-fly Packing)

The `MultiModalIterableDataset` streams data and packs sequences dynamically during iteration.

#### How it works:
1. **Stream**: Loads data samples one at a time
2. **Buffer**: Accumulates samples in a buffer
3. **Pack**: When buffer + next sample > `packing_length`, yields buffer
4. **Flush**: Yields remaining buffer at epoch end

#### Characteristics:
- ✅ **Memory efficient**: Only loads current batch
- ✅ **Fast startup**: No preprocessing required
- ✅ **Scales infinitely**: Works with any dataset size
- ❌ **Non-deterministic**: Different packing each epoch
- ❌ **Unknown length**: Can't calculate exact steps per epoch
- ❌ **Suboptimal packing**: Greedy algorithm may waste tokens

#### When to use:
- Large datasets (> 100GB)
- Limited memory environments
- Continuous/streaming data sources
- Production training at scale

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


