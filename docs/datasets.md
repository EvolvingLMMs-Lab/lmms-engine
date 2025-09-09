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

## Distributed Training Behavior

### Naive Dataset
- Uses `DistributedSampler` or `DistributedLengthGroupedSampler` 
- Each rank gets deterministic subset of packs
- Steps per epoch = `total_packs / world_size`
- Supports `group_by_length` for improved training efficiency

### Streaming Dataset  
- Performs rank sharding via `HFDataset.shard()`
- Worker splitting via `torch.utils.data.get_worker_info()`
- Dynamic step count per rank (depends on data distribution)
- No sampler attachment (handled internally)

⚠️ **Important**: Ensure dataset length divides evenly across ranks to avoid imbalanced workloads.

## Configuration Reference

### Core Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `packing` | bool | Enable sequence packing | `False` |
| `packing_length` | int | Maximum tokens per packed sequence | `32000` |
| `filter_overlong` | bool | Drop sequences exceeding `packing_length` | `True` |
| `packing_strategy` | str | **Naive only**: `first_fit` or `window_XX` | `first_fit` |
| `shuffle` | bool | Shuffle dataset before packing | `True` |

### Packing Strategies (Naive Only)

- **`first_fit`**: Greedily pack sequences into first available space
- **`window_XX`**: Group sequences within sliding windows of size XX

### Configuration Examples

#### YAML Configuration
```yaml
dataset:
  dataset_type: vision
  dataset_format: hf_dataset
  dataset_path: your/dataset/path
  shuffle: true
  packing: true
  packing_length: 32000
  filter_overlong: true
  processor_config:
    processor_type: your_processor
```

#### Python Configuration
```python
config = DatasetConfig(
    dataset_type="vision",
    dataset_format="hf_dataset",
    dataset_path="your/dataset/path",
    packing=True,
    packing_length=32000,
    filter_overlong=True,
    processor_config={"processor_type": "your_processor"}
)
```

## Performance Tips

### Optimizing Packing Efficiency

1. **Choose appropriate `packing_length`**:
   - Too small: Underutilized sequences
   - Too large: May exceed memory limits
   - Recommended: Start with model's max sequence length

2. **Monitor packing metrics**:
   ```python
   # Trainer logs these automatically
   - perf/global_seq_len_avg  # Average packed sequence length
   - perf/global_seq_len_min  # Minimum across ranks
   - perf/global_seq_len_max  # Maximum across ranks
   ```

3. **Handle outliers**:
   - Set `filter_overlong=True` to drop anomalously long sequences
   - Prevents memory spikes and improves batch consistency

### Memory Management

#### For Naive Dataset:
```python
# Estimate memory usage
estimated_memory = num_samples * avg_sample_size * 1.2  # 20% overhead
```

#### For Streaming Dataset:
```python
# Memory usage is constant
max_memory = batch_size * packing_length * token_size
```

## Troubleshooting

### Common Issues

#### 1. Distributed Training Hangs
**Problem**: Collective operations hang during training.

**Solution**: Ensure all ranks have identical tensor shapes:
```python
# Bad: Different shapes across ranks
loss = torch.tensor([loss1, loss2, ...])  

# Good: Scalar aggregation
loss = torch.tensor(loss.item())
torch.distributed.all_reduce(loss, op=ReduceOp.AVG)
```

#### 2. Imbalanced Workload
**Problem**: Some ranks finish before others.

**Solution**: 
- Ensure dataset size is divisible by world_size
- Use streaming dataset for better load balancing
- Enable `shuffle=True` to randomize distribution

#### 3. OOM with Naive Dataset
**Problem**: Out of memory during dataset loading.

**Solutions**:
- Switch to streaming dataset
- Reduce `packing_length`
- Enable `filter_overlong=True`
- Use data sharding:
  ```python
  dataset = load_dataset("path", split=f"train[{rank}:{rank+1}:{world_size}]")
  ```

## Decision Matrix

| Criterion | Naive Dataset | Streaming Dataset |
|-----------|--------------|-------------------|
| Dataset Size | < 100GB | Any size |
| Memory Usage | High | Low |
| Startup Time | Slow | Fast |
| Packing Quality | Optimal | Good |
| Reproducibility | Yes | No |
| Step Count Known | Yes | No |
| LR Schedulers | All supported | Limited |
| Best For | Research, debugging | Production, scale |

## Migration Guide

### From Naive to Streaming
```python
# Before (Naive)
from lmms_engine.datasets import MultiModalDataset
dataset = MultiModalDataset(config)

# After (Streaming)
from lmms_engine.datasets import MultiModalIterableDataset
dataset = MultiModalIterableDataset(config)
# Note: Ensure dataset_format="hf_dataset"
```

### From Streaming to Naive
```python
# Before (Streaming)
from lmms_engine.datasets import MultiModalIterableDataset
dataset = MultiModalIterableDataset(config)

# After (Naive)
from lmms_engine.datasets import MultiModalDataset
dataset = MultiModalDataset(config)
# Note: May need to adjust memory allocation
```


