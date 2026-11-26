# coding=utf-8
# Copyright 2025 LMMs-Lab team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MiniCPM-o duplex (realtime) iterable dataset.

A realtime sample is one continuous environment stream (audio, optionally
with video) plus assistant turns anchored to the timeline:

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "audio_url", "audio_url": {"url": "stream.wav"}},
      {"type": "video_url", "video_url": {"url": "stream.mp4"}},
      {"type": "text", "text": "optional instruction"}
    ]},
    {"role": "assistant", "content": [
      {"type": "realtime_text", "start_sec": 3.2, "end_sec": 5.8,
       "text": "OK, I see it."},
      {"type": "realtime_text",
       "twt": [{"text": "好", "start": 6.1, "end": 6.3}, ...]}
    ]}
  ]
}
```

The processor splits the stream into fixed 1-second units and builds the
duplex sequence with ``<|listen|>`` / ``<|speak|>`` supervision.
"""

import json
import os
from typing import Dict, List, Optional

import av
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Dataset as HFDataset
from PIL import Image

from lmms_engine.datasets.iterable.multimodal_iterable_dataset import MultiModalIterableDataset
from lmms_engine.mapping_func import register_dataset


@register_dataset("minicpmo_realtime_iterable")
class MiniCPMORealtimeIterableDataset(MultiModalIterableDataset):
    """Duplex (realtime) dataset for MiniCPM-o 4.5."""

    def _build_from_config(self):
        super()._build_from_config()
        if not isinstance(self.data_list, HFDataset):
            self.data_list = HFDataset.from_list(self.data_list)
            if not hasattr(self, "data_folder") or self.data_folder is None:
                self.data_folder = [None] * len(self.data_list)

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        messages = data["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        env_audio: Optional[np.ndarray] = None
        frames: Optional[List[Image.Image]] = None
        segments: List[Dict] = []

        for message in messages:
            if message.get("role") == "assistant":
                for content in message.get("content", []):
                    if content.get("type") == "realtime_text":
                        segments.append(content)
                continue
            for content in message.get("content", []):
                content_type = content.get("type")
                if content_type == "audio_url":
                    env_audio = self._load_audio_file(content["audio_url"]["url"], data_folder)
                elif content_type == "video_url":
                    video_url = content["video_url"]["url"]
                    video_path = os.path.join(data_folder, video_url) if data_folder else video_url
                    frames, env_audio = self._load_realtime_video(video_path)
                elif content_type == "realtime_text":
                    segments.append(content)

        if env_audio is None:
            raise ValueError("Realtime samples require an audio_url or a video_url with an audio track.")

        return self.processor.process_realtime(
            env_audio=env_audio,
            segments=segments,
            frames=frames,
        )

    def _load_audio_file(self, url: str, data_folder: Optional[str]) -> np.ndarray:
        path = os.path.join(data_folder, url) if data_folder else url
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != self.processor.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.processor.sampling_rate)
        return audio.astype(np.float32)

    def _load_realtime_video(self, video_path: str):
        """Decode realtime video into per-unit frames plus the audio track."""
        frames_per_unit = self.processor.frames_per_unit
        with av.open(video_path) as container:
            video_stream = container.streams.video[0]
            native_fps = float(video_stream.average_rate) or 24.0
            target_fps = native_fps  # keep native; processor groups by frames_per_unit
            frame_period = 1.0 / target_fps
            frames = []
            next_t = 0.0
            for frame in container.decode(video_stream):
                t = float(frame.pts * frame.time_base)
                if t + 1e-6 >= next_t:
                    frames.append(frame.to_image())
                    next_t += frame_period
            audio = None
            if container.streams.audio:
                chunks = []
                for frame in container.decode(container.streams.audio[0]):
                    arr = frame.to_ndarray()
                    if arr.ndim == 2:
                        arr = arr.mean(axis=0)
                    chunks.append(arr)
                if chunks:
                    audio = np.concatenate(chunks).astype(np.float32)
                    if np.abs(audio).max() > 1.0:
                        audio = audio / 32768.0
                    src_sr = int(container.streams.audio[0].rate)
                    if src_sr != self.processor.sampling_rate:
                        audio = librosa.resample(audio, orig_sr=src_sr, target_sr=self.processor.sampling_rate)
        return frames, audio

    def get_collator(self):
        from lmms_engine.datasets.collator.minicpmo_collator import MiniCPMOCollator

        return MiniCPMOCollator(self.processor)
