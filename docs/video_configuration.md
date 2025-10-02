# Video Configuration Guide

This guide explains the video processing configuration options available in LMMs Engine and provides migration instructions for users upgrading from older versions.

## Video Configuration Parameters

The following parameters can be configured in the `dataset_config` section of your training configuration:

### Basic Video Parameters

- **`video_backend`** (Optional[str], default: "qwen_vl_utils")
  - Specifies the backend to use for video loading
  - Available options: `"decord"`, `"qwen_vl_utils"`, `"qwen_omni_utils"`
  - Note: The `"torchvision"` backend has been removed. See [Migration Guide](#migration-from-torchvision-backend) below.

- **`video_sampling_strategy`** (Optional[str], default: "fps")
  - Determines how frames are sampled from videos
  - Options:
    - `"fps"`: Sample frames based on frames per second
    - `"frame_num"`: Sample a fixed number of frames

### Frame Sampling Parameters

- **`fps`** (Optional[int], default: 1)
  - Frames per second to sample when using `video_sampling_strategy: "fps"`
  - Must be a positive integer

- **`frame_num`** (Optional[int], default: 64)
  - Number of frames to sample when using `video_sampling_strategy: "frame_num"`
  - Must be a positive integer

### Video Size Limits

- **`video_max_pixels`** (Optional[int], default: 768 * 28 * 28)
  - Maximum number of pixels per video frame
  - Helps control memory usage during training
  - Must be a positive integer

- **`video_max_frames`** (Optional[int], default: 768)
  - Maximum number of frames to load from a video
  - Prevents loading excessively long videos
  - Must be a positive integer

### Filtering Options

- **`filter_overlong`** (Optional[bool], default: True)
  - When `packing` is enabled, filter out samples that exceed `packing_length`
  - Set to `False` to keep all samples regardless of length

## Example Configuration

```yaml
dataset_config:
  dataset_type: "vision"
  dataset_format: "json"
  
  # Video configuration
  video_backend: "qwen_vl_utils"
  video_sampling_strategy: "fps"
  fps: 2
  video_max_pixels: 602112  # 768 * 28 * 28
  video_max_frames: 512
  
  # Packing configuration
  packing: true
  packing_length: 32000
  filter_overlong: true
```

## Processor Configuration for Video

When using the Qwen2.5-VL processor, you can also configure video-specific parameters through `extra_kwargs`:

```yaml
processor_config:
  processor_name: "Qwen/Qwen2.5-VL-Instruct"
  processor_type: "qwen2_5_vl"
  extra_kwargs:
    video_max_pixels: 602112
    video_min_pixels: 28800
```

## Migration from Torchvision Backend

The `torchvision` video backend has been removed since it was implemented as a fallback in qwen-vl-utils

### Migration Steps

1. **Update your configuration file:**
   ```yaml
   # Old configuration
   video_backend: "torchvision"
   
   # New configuration (recommended)
   video_backend: "qwen_vl_utils"
   ```

2. **Install the new backend:**
   ```bash
   # For decord backend
   uv pip install decord
   
   # For qwen_vl_utils backend
   uv pip install qwen-vl-utils
   ```

3. **Verify compatibility:**
   - `decord` naive decord video loading, used in load from cloud storage
   - `qwen_vl_utils` is optimized for Qwen models and provides additional features
   - `qwen_omni_utils` supports audio extraction from videos for Qwen Omni variants

## Training Performance Optimization

### Memory Management

The `torch_empty_cache_steps` parameter in the trainer configuration helps manage GPU memory:

```yaml
# Clear CUDA cache every 100 steps
torch_empty_cache_steps: 100
```

This periodically clears the CUDA memory cache to prevent fragmentation during long training runs.

## Troubleshooting

### Common Issues

1. **Video loading failures:**
   - Check that the video file exists and is readable
   - Verify the video backend is properly installed
   - Review error logs for specific failure reasons

2. **Out of memory errors:**
   - Reduce `video_max_pixels` or `video_max_frames`
   - Enable `filter_overlong` to skip oversized samples
   - Use `torch_empty_cache_steps` to clear memory periodically

3. **Validation errors:**
   - Ensure all numeric parameters are positive integers
   - Check that `video_backend` is one of the supported options

## Best Practices

1. **Start with conservative limits:** Begin with smaller `video_max_pixels` and `video_max_frames` values, then increase as needed.

2. **Monitor memory usage:** Use tools like `nvidia-smi` to track GPU memory during training.

3. **Choose the right backend:**
   - We recommend to use `qwen_vl_utils` as it has much more features to config video loading
   - If you want to load from the cloud storage, `decord` is the only option now and is currently not configurable for video options

4. **Optimize sampling strategy:**
   - Use `"fps"` for videos with consistent motion
   - Use `"frame_num"` when you need exactly N frames regardless of video length

5. **Audio extraction from videos:**
   - Use `video_backend: "qwen_omni_utils"` with `use_audio_in_video: true` in processor config to extract audio from video files

## Audio from Video Extraction

When training Qwen Omni models, you can extract audio tracks from video files automatically.

### Configuration

```yaml
dataset_config:
  dataset_type: vision_audio
  video_backend: "qwen_omni_utils"
  video_sampling_strategy: "fps"
  fps: 1
  video_max_frames: 60

  processor_config:
    processor_name: "Qwen/Qwen2.5-Omni-7B"
    processor_type: "Qwen2_5OmniProcessor"
    extra_kwargs:
      use_audio_in_video: true
      audio_max_length: 60
```