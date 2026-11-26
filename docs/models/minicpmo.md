# MiniCPM-o Training

## Overview

MiniCPM-o 4.5 is a 9B end-to-end omni-modal model (SigLip2 + Whisper-medium + Qwen3-8B + CosyVoice2) from OpenBMB supporting vision, speech and full-duplex multimodal live streaming.

The lmms-engine integration vendors the upstream model code (trimmed to the training-relevant forward path) under `src/lmms_engine/models/minicpmo/` and registers it as `minicpmo`, so the official checkpoint loads without `trust_remote_code`.

Two training modes are supported:

- **Simplex** (turn-based SFT): image / audio / video conversations. The processor reproduces the official `MiniCPMOProcessor` tokenization exactly (verified token-by-token), including high-resolution image slicing, 1-second audio chunking and the chat template's thinking markers.
- **Duplex / realtime SFT**: a continuous audio (optionally video) stream is split into 1-second units; the model learns per unit to keep listening (`<|listen|>`) or to speak (`<|speak|> text <|chunk_tts_eos|>`). This matches the model's duplex streaming inference interface.

In both modes only the text stream is supervised — the TTS head is loaded from the checkpoint but frozen (`trainer_args.freeze_modules=["tts"]`), so checkpoints round-trip unchanged.

## Supported Features

| Feature | Support |
|---------|---------|
| **FSDP2** | ✅ |
| **USP** | ❌ |
| **Muon Optimizer** | ❌ |
| **Liger Kernel** | ❌ |
| **Packing** | ❌ |
| **NSA** | ❌ |
| **Expert Parallelism** | ❌ |

**Highlights**: Omni-modal (image + audio + video) understanding, full-duplex realtime training

## Quick Start

See the example configurations and run script:
- **Simplex**: [examples/minicpmo/example_config.yaml](../../examples/minicpmo/example_config.yaml)
- **Realtime**: [examples/minicpmo/realtime_config.yaml](../../examples/minicpmo/realtime_config.yaml)
- **Run Script**: [examples/minicpmo/run.sh](../../examples/minicpmo/run.sh)

```bash
bash examples/minicpmo/run.sh
```

## Key Configuration

```yaml
dataset_config:
  dataset_type: minicpmo_iterable      # or minicpmo_realtime_iterable
  processor_config:
    processor_type: minicpmo
    extra_kwargs:
      max_slice_nums: 9
      use_audio_in_video: true
  video_backend: av

model_config:
  load_from_pretrained_path: openbmb/MiniCPM-o-4_5
  attn_implementation: flash_attention_2

trainer_args:
  freeze_modules: ["tts"]
  use_rmpad: false
  fsdp2: true
  fsdp_config:
    transformer_layer_cls_to_wrap: [Qwen3DecoderLayer]
```

## Data Formats

### Simplex

Standard OpenAI chat messages. Media content items are ordered by appearance:

```json
{"messages": [
  {"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "img.jpg"}},
    {"type": "audio_url", "audio_url": {"url": "speech.wav"}},
    {"type": "video_url", "video_url": {"url": "clip.mp4", "video_start": 0, "video_end": 30}},
    {"type": "text", "text": "Describe what you see and hear."}]},
  {"role": "assistant", "content": [{"type": "text", "text": "It shows ..."}]}]}
```

Videos are decoded into frames (sampled at `dataset_config.fps`, capped at 10 fps) plus their audio track.

### Realtime (duplex)

One continuous environment stream plus assistant turns anchored to the timeline. Assistant text can either be a span (characters are linearly interpolated across `[start_sec, end_sec]`) or carry character-level timestamps:

```json
{"messages": [
  {"role": "user", "content": [
    {"type": "audio_url", "audio_url": {"url": "stream.wav"}},
    {"type": "text", "text": "Respond in real time."}]},
  {"role": "assistant", "content": [
    {"type": "realtime_text", "start_sec": 3.2, "end_sec": 5.8, "text": "OK, I see it."},
    {"type": "realtime_text", "twt": [{"text": "好", "start": 6.1, "end": 6.3}]}]}]}
```

The rendered sequence interleaves one unit per second:

```
<|im_start|>system
Streaming Omni Conversation.<|im_end|>
<unit><|audio_start|>...10 audio tokens...<|audio_end|><|listen|></unit>
<unit><|audio_start|>...10 audio tokens...<|audio_end|><|speak|>OK, I<|chunk_tts_eos|></unit>
...
```

`<|listen|>` / `<|speak|>` / the text and `<|chunk_tts_eos|>` are supervised; the `<unit>` markers and audio slots are inputs only.

## Freezing Modules

Use `trainer_args.freeze_modules` to select what to train (resolved as dotted attribute paths from the model root):

- `["tts"]` — default: train LLM + audio + vision, keep the speech head intact
- `["tts", "vpm", "resampler"]` — additionally freeze the vision tower
- `["tts", "apm", "audio_projection_layer"]` — additionally freeze the audio tower
