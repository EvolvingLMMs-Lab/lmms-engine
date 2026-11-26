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

"""MiniCPM-o simplex iterable dataset.

Handles turn-based conversations containing ``image_url``, ``audio_url`` and
``video_url`` content items.  Videos are decoded into frames (sampled at
``dataset_config.fps``) plus an optional audio track, both fed to the
processor as plain images / audio.
"""

import json
import os
from typing import Dict, List, Optional

import av
import numpy as np
import torch
from datasets import Dataset as HFDataset
from loguru import logger
from PIL import Image

from lmms_engine.datasets.iterable.multimodal_iterable_dataset import MultiModalIterableDataset
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("minicpmo_iterable")
class MiniCPMOIterableDataset(MultiModalIterableDataset):
    """Simplex (turn-based) dataset for MiniCPM-o 4.5."""

    def _build_from_config(self):
        super()._build_from_config()
        if not isinstance(self.data_list, HFDataset):
            self.data_list = HFDataset.from_list(self.data_list)
            if not hasattr(self, "data_folder") or self.data_folder is None:
                self.data_folder = [None] * len(self.data_list)

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        images: List[Image.Image] = []
        audios: List[np.ndarray] = []
        messages = data["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        for message in messages:
            for content in message.get("content", []):
                content_type = content.get("type")
                if content_type == "image_url":
                    images.append(self.load_image(content["image_url"]["url"], data_folder=data_folder))
                elif content_type == "audio_url":
                    audios.append(
                        self.load_audio(
                            content["audio_url"]["url"],
                            sr=self.processor.sampling_rate,
                            data_folder=data_folder,
                        )
                    )
                elif content_type == "video_url":
                    video_url = content["video_url"]["url"]
                    video_path = os.path.join(data_folder, video_url) if data_folder else video_url
                    extra = {k: v for k, v in content["video_url"].items() if k != "url" and v is not None}
                    frames, audio = self._load_video_with_audio(
                        video_path,
                        fps=self.config.fps,
                        use_audio=bool((self.config.extra_kwargs or {}).get("use_audio_in_video", True)),
                        **extra,
                    )
                    images.extend(frames)
                    if audio is not None:
                        audios.append(audio)

        hf_messages = TrainUtilities.convert_open_to_hf(messages)

        if len(images) == 0:
            images = None
        if len(audios) == 0:
            audios = None

        return self.processor.process(
            images=images,
            hf_messages=hf_messages,
            audios=audios,
            sampling_rate=self.processor.sampling_rate,
        )

    # ------------------------------------------------------------------
    # Video decoding
    # ------------------------------------------------------------------

    def _load_video_with_audio(
        self,
        video_path: str,
        fps: Optional[float] = None,
        use_audio: bool = True,
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
    ):
        """Decode a video into uniformly-sampled frames plus its audio track.

        Args:
            video_path: path to the video file.
            fps: target sampling rate; None keeps the native rate (capped
                at 10 fps following the model card).
            use_audio: whether to extract the audio track.
            video_start / video_end: optional sub-clip window in seconds.
        """
        with av.open(video_path) as container:
            video_stream = container.streams.video[0]
            native_fps = float(video_stream.average_rate)
            target_fps = min(fps or native_fps, 10.0)
            start = float(video_start or 0.0)
            end = float(video_end) if video_end is not None else float("inf")

            frames = []
            frame_period = 1.0 / target_fps
            next_t = start
            for frame in container.decode(video_stream):
                t = float(frame.pts * frame.time_base)
                if t < start:
                    continue
                if t > end:
                    break
                if t + 1e-6 >= next_t:
                    image = frame.to_image()
                    frames.append(image)
                    next_t += frame_period

            audio = self._extract_audio(container, start=start, end=None if end == float("inf") else end) if use_audio else None

        if not frames:
            logger.warning(f"No frames decoded from {video_path}; inserting a black frame.")
            frames = [Image.new("RGB", (448, 448))]
        return frames, audio

    @staticmethod
    def _extract_audio(container, start: float = 0.0, end: Optional[float] = None) -> Optional[np.ndarray]:
        """Extract a mono float32 audio window from an open container."""
        if not container.streams.audio:
            return None
        chunks = []
        for frame in container.decode(container.streams.audio[0]):
            t = float(frame.pts * frame.time_base)
            if t < start:
                continue
            if end is not None and t > end:
                break
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            chunks.append(arr)
        if not chunks:
            return None
        audio = np.concatenate(chunks).astype(np.float32)
        # Container audio is usually s16-le; normalize integer PCM.
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        return audio

    def get_collator(self):
        from lmms_engine.datasets.collator.minicpmo_collator import MiniCPMOCollator

        return MiniCPMOCollator(self.processor)
