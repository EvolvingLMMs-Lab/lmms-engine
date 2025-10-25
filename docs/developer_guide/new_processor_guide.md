# Creating Custom Data Processors

This guide explains how to implement custom data processors for the LMMS Engine. Data processors are responsible for transforming raw multimodal data (images, audio, videos, text) into tokenized sequences suitable for model training.

## Architecture Overview

The LMMS Engine processor hierarchy is designed for flexibility and reusability:

```
ProcessorConfig
    ↓
BaseProcessor (AeroDataProcessor)
    ↓
Specialized Processors (Qwen2_5-based, Text-only, etc.)
    ↓
YourCustomProcessor
```

### Key Components

- **ProcessorConfig**: Configuration container with processor name and type
- **Base Processors**: Define common functionality for tokenization and tensor creation
- **Specialized Processors**: Handle specific model requirements (Qwen, Llava, etc.)
- **Custom Processors**: Your implementation inheriting from appropriate base class

## When to Create a Custom Processor

Create a custom processor when you need to:

1. **Support a new model architecture** - Different models have different tokenization and formatting requirements
2. **Handle a new modality combination** - e.g., text + image + audio + video
3. **Implement custom tokenization logic** - Special token handling or chat templates
4. **Add model-specific preprocessing** - Image resizing, audio normalization, etc.

## Processor Roles

A processor typically handles:

| Task | Responsibility |
|------|-----------------|
| **Building** | Load tokenizer and media processors from HuggingFace |
| **Processing** | Transform raw data into model-compatible tensors |
| **Tokenization** | Convert text to token IDs using chat templates |
| **Token Expansion** | Expand special media tokens to appropriate lengths |
| **Label Masking** | Create training labels with attention masking |

## Architecture: Three Main Approaches

### 1. Text-Only Processor (Simplest)

Use when you only need to handle text data:

```python
from lmms_engine.mapping_func import register_processor
from lmms_engine.datasets.processor.config import ProcessorConfig
from transformers import AutoTokenizer

@register_processor("my_text_processor")
class MyTextProcessor:
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config
    
    def build(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.processor_name)
    
    def process(self, texts: List[str], **kwargs):
        return self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=2048,
            return_tensors='pt'
        )
    
    def save_pretrained(self, path: str):
        self.tokenizer.save_pretrained(path)
```

### 2. Audio-Only Processor

For audio modality:

```python
@register_processor("my_audio_processor")
class MyAudioProcessor:
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config
    
    def build(self):
        # Load audio processor from transformers
        from transformers import AutoFeatureExtractor
        self.audio_processor = AutoFeatureExtractor.from_pretrained(
            self.config.processor_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.processor_name)
    
    def process(self, audios: List[np.ndarray], sampling_rate: int, **kwargs):
        # Process audio with feature extractor
        audio_inputs = self.audio_processor(
            audios,
            sampling_rate=sampling_rate,
            return_tensors='pt'
        )
        
        # Tokenize text and integrate audio
        # ... implementation ...
        
        return {
            'input_ids': input_ids,
            'audio_values': audio_inputs['input_features'],
            'labels': labels
        }
```

### 3. Multimodal Processor (Most Common)

Handle images, audio, and/or videos - inherit from base processor:

```python
from lmms_engine.datasets.processor.aero_processor import AeroDataProcessor

@register_processor("my_multimodal_processor")
class MyMultimodalProcessor(AeroDataProcessor):
    def _build_processor(self):
        # Load model-specific processor from transformers
        from transformers import MyModelProcessor
        processor = MyModelProcessor.from_pretrained(
            self.config.processor_name
        )
        return processor
    
    def process(
        self,
        images: List[Image.Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        add_system_prompt=True,
        **kwargs,
    ):
        # Process each modality
        image_inputs = self._process_images(images)
        video_inputs = self._process_videos(videos)
        audio_inputs = self._process_audio(audios, sampling_rate)
        
        # Tokenize messages and expand media tokens
        inputs = self._tokenize_and_expand(
            hf_messages,
            image_inputs,
            video_inputs,
            audio_inputs,
            add_system_prompt
        )
        
        # Merge all inputs
        inputs.update(image_inputs)
        inputs.update(video_inputs)
        inputs.update(audio_inputs)
        
        return inputs
```

## Quick Start: Creating a Vision-Only Processor

Here's a complete minimal example for a vision processor:

### Step 1: Create the Processor Class

```python
from typing import List, Optional
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLProcessor

from lmms_engine.mapping_func import register_processor
from lmms_engine.datasets.processor.config import ProcessorConfig
from lmms_engine.datasets.processor.base_qwen2_5_processor import BaseQwen2_5_DataProcessor


@register_processor("qwen_vision_simple")
class QwenVisionSimpleProcessor(BaseQwen2_5_DataProcessor):
    """Simplified vision processor for Qwen2.5-VL model."""
    
    def _build_processor(self):
        """Build the underlying Qwen processor."""
        processor = Qwen2_5_VLProcessor.from_pretrained(
            self.config.processor_name
        )
        
        # Customize processor parameters if needed
        if hasattr(processor, 'image_processor'):
            image_max_pixels = self.config.extra_kwargs.get("image_max_pixels")
            if image_max_pixels:
                processor.image_processor.max_pixels = image_max_pixels
        
        return processor
    
    def process(
        self,
        images: List[Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt=True,
        add_generation_prompt=False,
        **kwargs,
    ):
        """Process vision data."""
        # Don't support audio in this simple example
        assert audios is None, "This processor does not support audio"
        
        # Call parent implementation which handles vision + video
        return super().process(
            images=images,
            hf_messages=hf_messages,
            audios=None,
            videos=videos,
            system_message=system_message,
            add_system_prompt=add_system_prompt,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
```

### Step 2: Register Configuration

Add to your training config:

```yaml
processor_config:
  processor_type: qwen_vision_simple
  processor_name: Qwen/Qwen2.5-VL-7B
  extra_kwargs:
    image_max_pixels: 1000000
```

## Required Methods

Every processor must implement these methods:

### 1. `__init__(config: ProcessorConfig)`

Initialize processor with configuration:

```python
def __init__(self, config: ProcessorConfig) -> None:
    self.config = config
    self.processor = None  # Will be set in build()
```

### 2. `build()`

Load model components (tokenizer, image processor, etc.):

```python
def build(self):
    self.processor = self._build_processor()
    # Optional: set custom chat template
    self.processor.chat_template = self.chat_template_custom
```

### 3. `_build_processor()`

Load the actual processor from transformers:

```python
def _build_processor(self):
    from transformers import Qwen3VLProcessor
    return Qwen3VLProcessor.from_pretrained(self.config.processor_name)
```

### 4. `process(images, hf_messages, ...)`

Main method that transforms data:

```python
def process(
    self,
    images: List[Image.Image],
    hf_messages,
    audios: Optional[List[np.ndarray]] = None,
    sampling_rate: Optional[int] = None,
    videos=None,
    **kwargs,
):
    """
    Transform multimodal data into model-ready tensors.
    
    Args:
        images: List of PIL Images or None
        hf_messages: Messages in HuggingFace format with roles
        audios: List of audio arrays or None
        sampling_rate: Audio sampling rate
        videos: List of video frames or None
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with keys:
        - input_ids: Token IDs tensor
        - attention_mask: Attention mask tensor
        - labels: Training labels tensor
        - pixel_values: Image tensors (if images present)
        - audio_values: Audio tensors (if audio present)
        - video_pixel_values: Video tensors (if videos present)
    """
    # Implementation
    pass
```

### 5. `save_pretrained(path: str)`

Save processor state:

```python
def save_pretrained(self, save_directory: str):
    if not hasattr(self, "processor"):
        raise ValueError("Processor has not been built yet")
    self.processor.save_pretrained(save_directory)
```

## Message Format

Input messages should follow the HuggingFace format:

```python
hf_messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}},
            {"type": "text", "text": "What's in this image?"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "This image shows..."}
        ]
    }
]
```

Content types supported:
- `text` - Text content
- `image_url` - Image content
- `audio_url` - Audio content  
- `video_url` - Video content

## Key Implementation Patterns

### Pattern 1: Simple Inheritance from Base

When your model is a variant of Qwen2.5-VL:

```python
@register_processor("my_variant")
class MyVariantProcessor(BaseQwen2_5_DataProcessor):
    def _build_processor(self):
        processor = MyProcessor.from_pretrained(self.config.processor_name)
        # Custom configuration
        return processor
    
    # Usually no need to override process() if parent handles it
```

### Pattern 2: Custom Processing Logic

When you need special tokenization:

```python
def get_qwen_template_labels(
    self,
    hf_messages,
    num_image_tokens,
    num_video_tokens,
    system_message="You are a helpful assistant",
    add_system_prompt=True,
):
    """Custom label generation with your logic."""
    input_ids = []
    labels = []
    
    # Add system prompt if needed
    if add_system_prompt:
        system_tokens = self.tokenizer.encode(system_message)
        input_ids.extend(system_tokens)
        labels.extend([-100] * len(system_tokens))  # Mask system
    
    # Process each message
    for message in hf_messages:
        role = message["role"]
        
        # Encode message content
        message_tokens = self._encode_message(message)
        input_ids.extend(message_tokens)
        
        # Label masking: user messages masked, assistant unmasked
        if role in ["user", "system"]:
            labels.extend([-100] * len(message_tokens))
        else:
            labels.extend(message_tokens)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
```

### Pattern 3: Token Expansion

Handle special media tokens that need expansion:

```python
def _expand_encode_id_image_tokens(
    self,
    encode_id: List[int],
    image_token_num: List[int],
    start_from: int = 0,
):
    """
    Expand image placeholder tokens to actual token count.
    
    Example: If <|image|> represents 196 tokens, expand it to 196 copies.
    """
    image_pos = [i for i, x in enumerate(encode_id) if x == self.image_token_id]
    expanded_encode_id = []
    prev = 0
    
    for idx, pos in enumerate(image_pos):
        # Add tokens before image position
        expanded_encode_id.extend(encode_id[prev:pos])
        
        # Expand the image token
        token_count = image_token_num[idx + start_from]
        expanded_encode_id.extend([self.image_token_id] * token_count)
        
        prev = pos + 1
    
    # Add remaining tokens
    expanded_encode_id.extend(encode_id[prev:])
    
    return expanded_encode_id, len(image_pos)
```

## Properties and Utilities

Define useful properties for token access:

```python
@property
def image_token_id(self):
    """Get the special token ID for images."""
    image_token = getattr(self.processor, "image_token", None)
    if image_token is None:
        return None
    return self.processor.tokenizer.convert_tokens_to_ids(image_token)

@property
def audio_token_id(self):
    """Get the special token ID for audio."""
    audio_token = getattr(self.processor, "audio_token", None)
    if audio_token is None:
        return None
    return self.processor.tokenizer.convert_tokens_to_ids(audio_token)

@property
def video_token_id(self):
    """Get the special token ID for videos."""
    video_token = getattr(self.processor, "video_token", None)
    if video_token is None:
        return None
    return self.processor.tokenizer.convert_tokens_to_ids(video_token)

@property
def tokenizer(self):
    """Get the underlying tokenizer."""
    return self.processor.tokenizer

@property
def sampling_rate(self):
    """Get audio sampling rate."""
    if hasattr(self.processor, 'audio_processor'):
        return self.processor.audio_processor.sampling_rate
    return None
```

## Chat Templates

Customize how conversations are formatted:

```python
@property
def chat_template_custom(self):
    """Define how messages are formatted for the model."""
    return (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}<|im_end|>\n"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'text' %}"
        "{{ content['text'] }}"
        "{% elif content['type'] == 'image_url' %}"
        "<|image|>"
        "{% elif content['type'] == 'audio_url' %}"
        "<|audio|>"
        "{% endif %}"
        "{% endfor %}"
        "<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
    )
```

## ProcessorConfig

Configure your processor in dataset config:

```yaml
processor_config:
  processor_type: my_custom_processor      # Name used in @register_processor
  processor_name: org/model-name           # HF model identifier
  extra_kwargs:                            # Optional custom parameters
    image_max_pixels: 1000000
    video_max_frames: 100
    custom_param: value
```

Access extra kwargs in your processor:

```python
def _build_processor(self):
    max_pixels = self.config.extra_kwargs.get("image_max_pixels", 1000000)
    # Use max_pixels for configuration
```

## Testing Your Processor

Test your implementation:

```python
from lmms_engine.datasets.processor.config import ProcessorConfig
from PIL import Image
import numpy as np

# Setup
config = ProcessorConfig(
    processor_type="my_processor",
    processor_name="Qwen/Qwen2.5-VL-7B"
)
processor = MyProcessor(config)
processor.build()

# Test with sample data
images = [Image.new('RGB', (224, 224))]
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "placeholder"}},
            {"type": "text", "text": "What is this?"}
        ]
    }
]

# Process
output = processor.process(
    images=images,
    hf_messages=messages
)

# Verify output
assert "input_ids" in output
assert "labels" in output
assert output["input_ids"].shape[0] > 0
print("✓ Processor test passed")
```

## Best Practices

1. **Inherit from appropriate base class**
   - Text-only: Inherit from no base
   - Multimodal: Inherit from `AeroDataProcessor` or `BaseQwen2_5_DataProcessor`

2. **Always implement `_build_processor()`**
   - This is where model-specific loading happens
   - Keep it focused and readable

3. **Document modality support**
   - Clearly state what your processor supports in docstrings
   - Use assertions to validate inputs

4. **Handle edge cases**
   ```python
   if images is None:
       image_inputs = {}
   else:
       # Process images
   ```

5. **Use the @register_processor decorator**
   - This makes your processor discoverable
   - Use lowercase with underscores for processor names

6. **Mask appropriately for training**
   - User/system messages typically masked (-100)
   - Assistant responses unmasked for training
   - Special tokens handled carefully

7. **Return consistent output format**
   - Always include: `input_ids`, `attention_mask`, `labels`
   - Include modality-specific keys when applicable

## Common Issues and Solutions

### Issue: KeyError for 'input_ids'
**Cause**: Processor not returning required keys
**Solution**: Ensure `process()` returns dict with `input_ids` and `labels`

### Issue: Token count mismatch
**Cause**: Media token expansion incorrect
**Solution**: Verify token counting logic and that `_expand_encode_id_*` methods are correct

### Issue: Chat template not applied
**Cause**: Forgot to set template in `build()`
**Solution**: Add `self.processor.chat_template = self.chat_template_custom` in `build()`

### Issue: Audio/Video not processed
**Cause**: Missing processor components
**Solution**: Verify `_build_processor()` loads all needed components

## Advanced: Custom Collator Integration

Processors often work with custom collators:

```python
# In your dataset
def get_collator(self):
    return CustomCollator(self.processor)

# Define collator to work with processor outputs
class CustomCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        # Batch is list of processor outputs
        # Stack tensors appropriately
        pass
```

## See Also

- [Creating New Datasets](./new_dataset_guide.md)
- [Video Configuration Reference](../reference/video_configuration.md)
- [API Reference](../reference/api.md)
