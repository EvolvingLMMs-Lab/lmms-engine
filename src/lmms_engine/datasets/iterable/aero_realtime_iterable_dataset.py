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

"""AeroRealtime iterable dataset.

Handles loading of video+audio data and realtime text segments for
AeroRealtime training.  Supports both normal video QA and realtime
training data (where assistant messages contain ``realtime_text``
content items with ``start_sec`` timestamps).

Audio is auto-extracted from video files using librosa.
"""

import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import av
import librosa
import soundfile as sf
from loguru import logger

warnings.filterwarnings("ignore", message=".*PySoundFile.*")
warnings.filterwarnings("ignore", message=".*__audioread_load.*", category=FutureWarning)

import numpy as np
import torch
from PIL import Image

from lmms_engine.datasets.collator import AeroRealtimeCollator
from lmms_engine.datasets.iterable.multimodal_iterable_dataset import (
    MultiModalIterableDataset,
)
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("aero_realtime_iterable")
class AeroRealtimeIterableDataset(MultiModalIterableDataset):
    """Dataset for AeroRealtime training.

    Extends VisionSFTIterableDataset with:
    - Audio extraction from video files
    - ``realtime_text`` content type handling
    - Video metadata passthrough for timestamp computation
    """

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        images_list = []
        videos = []
        audios = []
        realtime_segments = []
        video_paths = []
        kwargs = {}

        messages = data["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        is_realtime = bool(data.get("realtime", False))

        # First pass: collect media references and realtime segments
        for message in messages:
            message_time = message.get("time")
            if is_realtime and message_time is not None and message["role"] in ["user", "assistant"]:
                text = self._extract_text_content(message.get("content", []))
                if text:
                    realtime_segments.append(
                        {
                            "time": float(message_time),
                            "role": message["role"],
                            "text": text,
                        }
                    )
                continue

            for content in message["content"]:
                content_type = content.get("type")
                if content_type == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content_type == "video_url":
                    video_url_dict = content["video_url"]
                    video_url = video_url_dict["url"]
                    if data_folder is not None:
                        video_path = os.path.join(data_folder, video_url)
                    else:
                        video_path = video_url
                    extra = {k: v for k, v in video_url_dict.items() if k != "url" and v is not None}
                    video_paths.append((video_path, extra))

                    # Load video frames with metadata
                    frames, video_metadata, sample_fps = self._load_video_with_metadata(
                        video_url, data_folder=data_folder, video_kwargs=extra or None
                    )
                    videos.append(frames)
                    kwargs["fps"] = sample_fps
                    kwargs["video_metadata"] = video_metadata
                    kwargs["do_sample_frames"] = False

                elif content_type == "realtime_text":
                    realtime_segments.append(
                        {
                            "time": content["start_sec"],
                            "role": "assistant",
                            "text": content["text"],
                        }
                    )

        # Extract audio from video files
        if video_paths:
            for video_path, extra in video_paths:
                offset = float(extra.get("video_start", 0.0) or 0.0)
                end = extra.get("video_end")
                duration = (float(end) - offset) if end is not None else None
                audio = self._extract_audio_from_video(video_path, offset=offset, duration=duration)
                audios.append(audio)

        # Mix any user-provided TTS audio onto the (single) video audio track.
        # ``user_audio`` is a list of ``{"path", "start_time"}`` where
        # ``start_time`` is relative to the chunk start (= video_start).  This
        # is how EgoIT realtime data injects spoken questions.
        user_audio_entries = data.get("user_audio")
        if user_audio_entries is not None and len(user_audio_entries) > 0:
            if not audios:
                raise ValueError("user_audio requires a video track to mix onto")
            audios[0] = self._mix_user_audio_onto_track(
                base=audios[0],
                user_audio_entries=user_audio_entries,
                target_sr=self.processor.sampling_rate,
                data_folder=data_folder,
            )

        # Convert messages to HF format (realtime_text items are passed through)
        hf_messages = TrainUtilities.convert_open_to_hf(messages)

        # Load images
        if data_folder is not None:
            images = [Image.open(os.path.join(data_folder, img_path)) for img_path in images_list]
        else:
            images = [Image.open(img_path) for img_path in images_list]

        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None
        if len(audios) == 0:
            audios = None
        if len(realtime_segments) == 0:
            realtime_segments = None

        inputs = self.processor.process(
            images=images,
            hf_messages=hf_messages,
            audios=audios,
            sampling_rate=self.processor.sampling_rate,
            videos=videos,
            realtime_segments=realtime_segments,
            **kwargs,
        )
        return inputs

    @staticmethod
    def _extract_text_content(content) -> str:
        if isinstance(content, str):
            return content
        texts = []
        for item in content:
            if item and item.get("type") == "text" and item.get("text"):
                texts.append(item["text"])
        return "\n".join(texts)

    def _load_video_with_metadata(
        self,
        video_path: str,
        data_folder: Optional[str] = None,
        video_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, object, float]:
        """Load video frames and return metadata for timestamp computation.

        Uses qwen_vl_utils fetch_video with return_video_metadata=True to get
        frame indices and fps needed for timestamp computation.

        Disables torchvision fallback to avoid 30-min hangs on corrupted
        videos that would cause NCCL timeouts.

        Args:
            video_path: Path to video file (relative to data_folder).
            data_folder: Optional folder to prepend.
            video_kwargs: Optional extra ele fields forwarded to ``fetch_video``
                (e.g. ``video_start`` / ``video_end`` for sub-clip seek).

        Returns:
            Tuple of (frames, video_metadata, sample_fps).
        """
        from qwen_vl_utils import fetch_video
        from qwen_vl_utils import vision_process as _vp

        if data_folder is not None:
            full_path = os.path.join(data_folder, video_path)
        else:
            full_path = video_path

        video_dict = {
            "type": "video",
            "video": f"file://{full_path}",
            "min_frames": 1,
            "max_pixels": getattr(self.config, "video_max_pixels", 360 * 420),
            "max_frames": getattr(self.config, "video_max_frames", 512),
            "min_pixels": getattr(self.config, "video_min_pixels", 28 * 28),
        }
        if video_kwargs:
            video_dict.update(video_kwargs)

        if self.config.video_sampling_strategy == "frame_num":
            n_frames = self.config.frame_num
            video_dict["nframes"] = n_frames
        elif self.config.video_sampling_strategy == "fps":
            video_dict["fps"] = self.config.fps
        else:
            raise ValueError(f"Invalid video sampling strategy: {self.config.video_sampling_strategy}")

        # Temporarily remove torchvision from backends to prevent slow fallback
        # that can hang for 30+ minutes on problematic videos, causing NCCL timeout.
        _tv_backup = _vp.VIDEO_READER_BACKENDS.pop("torchvision", None)
        try:
            video_inputs, sample_fps = fetch_video(
                video_dict,
                return_video_sample_fps=True,
                return_video_metadata=True,
            )
        finally:
            if _tv_backup is not None:
                _vp.VIDEO_READER_BACKENDS["torchvision"] = _tv_backup

        frames, video_metadata = video_inputs
        frames = frames.numpy()
        return frames, video_metadata, sample_fps

    def _extract_audio_from_video(
        self,
        video_path: str,
        target_sr: Optional[int] = None,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """Extract a mono audio window from a video using PyAV.

        PyAV does container-level seek so reading a small window deep inside a
        long mp4 (e.g. Inf-Stream's multi-hour clips) costs O(seek) instead of
        O(offset) — ~100x faster than ``librosa.load(..., offset=...)`` which
        falls back to audioread and decodes from t=0.

        On any failure (no audio track, decode error) returns silence of length
        ``duration`` (or the video's own duration if unknown, then 1s) so the
        audio tower still participates in the forward pass and downstream
        supervision still has audio_pad slots to land in. For video-only data
        like LiveCC, silence is itself a valid training signal.
        """
        target_sr = target_sr or self.processor.sampling_rate
        try:
            with av.open(video_path) as container:
                stream = container.streams.audio[0]
                src_sr = int(stream.rate)
                end_t = offset + duration if duration is not None else float("inf")
                if offset > 0:
                    container.seek(int(offset / float(stream.time_base)), stream=stream)
                chunks = []
                for frame in container.decode(stream):
                    t = float(frame.pts * frame.time_base)
                    if t > end_t:
                        break
                    if t + frame.samples / float(frame.sample_rate) < offset:
                        continue
                    arr = frame.to_ndarray()
                    if arr.ndim == 2:  # planar (channels, samples) -> mono
                        arr = arr.mean(axis=0)
                    chunks.append(arr)
            if not chunks:
                raise RuntimeError("decoded zero audio frames")

            audio = np.concatenate(chunks)
            if audio.dtype.kind == "i":
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            else:
                audio = audio.astype(np.float32)
            if src_sr != target_sr:
                audio = librosa.resample(audio, orig_sr=src_sr, target_sr=target_sr)
            if duration is not None:
                audio = audio[: int(round(duration * target_sr))]
            return audio
        except Exception as e:
            silence_secs = duration if duration and duration > 0 else self._probe_video_duration(video_path) or 1.0
            # logger.debug(
            #     f"Audio extraction failed for {video_path} "
            #     f"(offset={offset}, duration={duration}): {e}. "
            #     f"Falling back to {silence_secs:.2f}s silence."
            # )
            return np.zeros(int(round(silence_secs * target_sr)), dtype=np.float32)

    @staticmethod
    def _probe_video_duration(video_path: str) -> Optional[float]:
        try:
            with av.open(video_path) as c:
                return float(c.duration) / float(av.time_base) if c.duration else None
        except Exception:
            return None

    @staticmethod
    def _load_audio_file(path: str, target_sr: int) -> Optional[np.ndarray]:
        """Load a standalone wav file as mono float32 at ``target_sr``."""
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            return None
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32, copy=False)

    @classmethod
    def _mix_user_audio_onto_track(
        cls,
        base: np.ndarray,
        user_audio_entries: List[Dict[str, Any]],
        target_sr: int,
        data_folder: Optional[str],
        base_attenuation: float = 0.3,
    ) -> np.ndarray:
        """Overlay TTS waveforms on a base audio track at their ``start_time``.

        ``base`` is the audio extracted from the video clip; each entry in
        ``user_audio_entries`` is ``{"path", "start_time"}`` with ``start_time``
        in seconds relative to the clip start.  The base track is attenuated in
        the overlap window so the spoken TTS dominates without clipping; the
        whole track is clipped to ``[-1, 1]`` at the end.

        Returns the modified base array (mutated in place for efficiency).
        """
        base_len = base.shape[0]
        for entry in user_audio_entries:
            rel = entry["path"]
            full = os.path.join(data_folder, rel) if data_folder is not None else rel
            wav = cls._load_audio_file(full, target_sr=target_sr)
            if wav is None or wav.shape[0] == 0:
                continue
            st = max(0.0, float(entry.get("start_time", 0.0)))
            ofs = int(round(st * target_sr))
            if ofs >= base_len:
                continue
            end_ofs = min(ofs + wav.shape[0], base_len)
            base[ofs:end_ofs] = base[ofs:end_ofs] * base_attenuation + wav[: end_ofs - ofs]
        np.clip(base, -1.0, 1.0, out=base)
        return base

    def get_collator(self):
        return AeroRealtimeCollator(self.processor)
