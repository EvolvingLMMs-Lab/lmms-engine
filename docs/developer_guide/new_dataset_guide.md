# Creating New Datasets

This guide walks you through creating a new dataset for the LMMS Engine. We support both **map-style (naive)** and **iterable** datasets to handle different data loading scenarios.

## Architecture Overview

The LMMS Engine provides a flexible dataset framework with the following hierarchy:

```
BaseDataset / BaseIterableDataset
    ↓
MultiModalDataset / MultiModalIterableDataset
    ↓
(MultiModalDataLoadingMixin)
    ↓
YourCustomDataset
```

### Key Components

- **Base Classes**: Abstract base classes that define the dataset interface
- **MultiModal Classes**: Implement common functionality for handling images, audio, and video
- **MultiModalDataLoadingMixin**: Provides reusable methods for loading different media types
- **Your Custom Dataset**: Inherits from one of the multimodal classes and implements data format-specific logic

## Choosing Between Map-Style and Iterable Datasets

### Map-Style Datasets (Recommended for Most Use Cases)

**Use `MultiModalDataset` when:**
- Your dataset fits entirely in memory or has a fixed, known size
- You need random access to samples (important for training with shuffling)
- Your data is relatively static and doesn't stream continuously
- You want simpler implementation with fewer moving parts

**Advantages:**
- Simpler implementation with standard indexing (`__getitem__`)
- Better compatibility with distributed training
- Can easily apply packing strategies for efficient batching
- Supports data shuffling and filtering during dataset building

**Example:**
```python
dataset = VisionSFTDataset(config)
sample = dataset[42]  # Direct random access
```

### Iterable Datasets (For Streaming Data)

**Use `MultiModalIterableDataset` when:**
- Your dataset is very large or streams data continuously
- You're using it with a real-time data source
- You prefer yielding samples via iteration rather than indexing
- Your data format naturally supports streaming

**Advantages:**
- Handles large datasets without loading everything into memory
- Native streaming support for distributed training
- Better for continuous data pipelines
- Can dynamically fetch and process data on-the-fly

**Example:**
```python
dataset = VisionSFTIterableDataset(config)
for sample in dataset:
    process(sample)
```

## Quick Start: Creating a Map-Style Dataset

Here's the simplest approach - inherit from `MultiModalDataset`:

### Step 1: Create Your Dataset Class

```python
from typing import Dict
import torch
from PIL import Image

from lmms_engine.datasets.naive.multimodal_dataset import MultiModalDataset
from lmms_engine.datasets.collator import VisionCollator
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("my_dataset")
class MyCustomDataset(MultiModalDataset):
    """Custom dataset for handling my specific data format."""
    
    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        """
        Load and process data from JSON format.
        
        Args:
            data: Dictionary containing a 'messages' key with conversation data
            data_folder: Optional base folder for relative file paths
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask', etc.
        """
        messages = data["messages"]
        images_list = []
        videos = []
        kwargs = {}
        
        # Extract media from messages
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "video_url":
                    frames, sample_fps = self.load_videos(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    videos.append(frames)
                    kwargs["fps"] = sample_fps
        
        # Convert to HuggingFace format
        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        
        # Load images
        if data_folder is not None:
            images = [
                Image.open(os.path.join(data_folder, img)) 
                for img in images_list
            ]
        else:
            images = [Image.open(img) for img in images_list]
        
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None
        
        # Process through the configured processor
        inputs = self.processor.process(
            images=images, 
            hf_messages=hf_messages, 
            videos=videos, 
            **kwargs
        )
        return inputs
    
    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        """Load from HuggingFace dataset format."""
        messages = data["messages"]
        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        
        # Handle single or multiple images
        if isinstance(data["image"], list):
            images = data["image"]
        else:
            images = [data["image"]]
        
        inputs = self.processor.process(
            images=images, 
            hf_messages=hf_messages
        )
        return inputs
    
    def get_collator(self):
        """Return the appropriate collator for batching."""
        return VisionCollator(self.processor)
```

### Step 2: Register Your Dataset

The `@register_dataset("my_dataset")` decorator automatically registers your dataset. You can then use it in your config:

```yaml
dataset_type: my_dataset
```

## Creating an Iterable Dataset

For iterable datasets, inherit from `MultiModalIterableDataset`:

```python
from lmms_engine.datasets.iterable.multimodal_iterable_dataset import (
    MultiModalIterableDataset,
)


@register_dataset("my_iterable_dataset")
class MyIterableDataset(MultiModalIterableDataset):
    """Streaming dataset for continuous data pipelines."""
    
    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        # Same implementation as map-style
        # The base class handles streaming logic
        pass
    
    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        # Same implementation as map-style
        pass
    
    def get_collator(self):
        return VisionCollator(self.processor)
```

The base `MultiModalIterableDataset` handles the `__iter__` method for you, calling your `load_from_json` or `load_from_hf` methods as it iterates through the data.

## Required Methods to Implement

Every custom dataset must implement these methods:

### 1. `load_from_json(data, data_folder=None)`

Transforms raw JSON data into processor-ready format:
- Extract media paths/URLs
- Load images using `self.load_image()`
- Load videos using `self.load_videos()`
- Load audio using `self.load_audio()`
- Convert to HuggingFace message format
- Return processed dictionary with tensor outputs

### 2. `load_from_hf(data)`

Handles HuggingFace dataset format:
- Extract data fields
- Process media similarly to JSON
- Return processed dictionary

### 3. `get_collator()`

Returns a collator instance for batching:
- Import appropriate collator (e.g., `VisionCollator`, `AudioCollator`)
- Pass your processor instance
- Example: `return VisionCollator(self.processor)`

### 4. `_build_from_config()` (Optional)

Override if you need custom initialization logic. The base class already handles:
- Loading various data formats (JSON, JSONL, Arrow, Parquet, HuggingFace, YAML)
- Shuffling
- Token estimation
- Packing (for map-style)

## Available Media Loading Methods

The `MultiModalDataLoadingMixin` provides these methods:

### Loading Images

```python
image = self.load_image(image_path, data_folder=None)
# Returns: PIL.Image
```

### Loading Audio

```python
audio = self.load_audio(audio_path, sr=16000, data_folder=None)
# Returns: numpy.ndarray (1D)
```

### Loading Videos

```python
frames, sample_fps = self.load_videos(
    video_path, 
    data_folder=None, 
    fps=1
)
# Returns: (numpy.ndarray, float)
```

## Supported Data Formats

Map-style and iterable datasets support loading from:

- **JSON**: List of data dictionaries with 'messages'
- **JSONL**: Line-delimited JSON
- **Arrow**: Hugging Face arrow format
- **Parquet**: Parquet format
- **HuggingFace**: Direct HF dataset loading
- **YAML**: Inline datasets or external YAML files

## Object Storage Support

Both dataset types support cloud storage backends:

- **GCS (Google Cloud Storage)**: Set `object_storage: "gcs"`
- **Azure Blob Storage**: Set `object_storage: "azure"`
- **Local filesystem**: Set `object_storage: "none"` (default)

## Configuration

Your dataset configuration typically looks like:

```yaml
dataset_type: my_dataset
dataset_format: json  # json, jsonl, arrow, parquet, hf_dataset, yaml
dataset_path: /path/to/data.json
shuffle: true
filter_overlong: true
max_length: 2048
packing: false
processor_config:
  processor_type: qwen_processor  # Or your processor type
```

## Best Practices

1. **Start with `MultiModalDataset`**: Unless you have a specific need for streaming, use map-style datasets. They're simpler and more compatible with standard training loops.

2. **Reuse MultiModalDataset**: Inheriting from `MultiModalDataset` gives you all the built-in data format handling and media loading methods.

3. **Implement format handlers**: Focus on `load_from_json()` and `load_from_hf()`. These are the main customization points.

4. **Use TrainUtilities.convert_open_to_hf()**: This standardizes your message format for the processor.

5. **Handle missing media gracefully**: Set images/videos to `None` if not present.

6. **Leverage the processor**: The processor handles tokenization, image resizing, etc. Pass it the standard format and let it work.

## Example: Complete Vision Dataset

```python
from typing import Dict
import os
import torch
from PIL import Image

from lmms_engine.datasets.naive.multimodal_dataset import MultiModalDataset
from lmms_engine.datasets.collator import VisionCollator
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("custom_vision")
class CustomVisionDataset(MultiModalDataset):
    
    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        messages = data["messages"]
        images_list = []
        
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
        
        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        
        if data_folder is not None:
            images = [
                Image.open(os.path.join(data_folder, img)) 
                for img in images_list
            ]
        else:
            images = [Image.open(img) for img in images_list] if images_list else None
        
        inputs = self.processor.process(
            images=images, 
            hf_messages=hf_messages
        )
        return inputs
    
    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        messages = data["messages"]
        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        images = data.get("image", None)
        
        inputs = self.processor.process(
            images=images, 
            hf_messages=hf_messages
        )
        return inputs
    
    def get_collator(self):
        return VisionCollator(self.processor)
```

## Testing Your Dataset

After implementation, test your dataset:

```python
from lmms_engine.datasets.config import DatasetConfig

config = DatasetConfig(
    dataset_type="my_dataset",
    dataset_format="json",
    dataset_path="path/to/data.json",
    processor_config={"processor_type": "qwen_processor"}
)

dataset = MyCustomDataset(config)
dataset.build()

# Test access
sample = dataset[0]
print(sample.keys())  # Should have input_ids, attention_mask, etc.

# Test collator
collator = dataset.get_collator()
batch = collator([dataset[i] for i in range(4)])
```

## Common Issues

### Issue: AttributeError for `load_from_json`
**Cause**: Method signature mismatch
**Solution**: Ensure your method signature matches: `load_from_json(self, data, data_folder=None)`

### Issue: Missing media files
**Cause**: Incorrect path construction
**Solution**: Always use `os.path.join(data_folder, path)` when data_folder is provided

### Issue: Processor returns empty tensors
**Cause**: Incorrect message format for processor
**Solution**: Use `TrainUtilities.convert_open_to_hf()` to standardize format

## See Also

- [Dataset Configuration Reference](../reference/dataset_configuration.md)
- [Video Configuration](../reference/video_configuration.md)
- [API Reference](../reference/api.md)
