"""Mixin for loading videos through the ``lmms_video_utils`` codec backend.

Separates the dataset-level codec orchestration (collecting canvases plus
their ``CodecVideoOutput`` metadata across a message list) from both the
backend implementation (``load_video_lmms_video_utils`` lives in
``MultiModalDataLoadingMixin``) and the concrete dataset class. A dataset
that mixes this in can pass the collected ``video_metadata`` straight into a
codec-aware processor (e.g. LLaVA-OneVision-2).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


class CodecVideoLoadingMixin:
    """Collects codec-stream video inputs from OpenAI-style messages.

    Requires the host class to also provide
    ``load_video_lmms_video_utils`` (from ``MultiModalDataLoadingMixin``)
    and a ``config`` with ``video_backend`` / ``fps``.
    """

    def load_codec_videos(
        self,
        video_path: str,
        data_folder: Optional[str] = None,
        fps: int = 1,
        video_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, float, Any]:
        assert (
            self.config.video_backend == "lmms_video_utils"
        ), "CodecVideoLoadingMixin requires video_backend='lmms_video_utils'"
        if data_folder is not None:
            video_path = os.path.join(data_folder, video_path)
        return self.load_video_lmms_video_utils(video_path, fps, video_kwargs=video_kwargs)

    def collect_codec_video_inputs(
        self,
        messages: List[dict],
        data_folder: Optional[str] = None,
    ) -> Tuple[List[Any], List[Any], Optional[float]]:
        """Walk ``messages`` and load every ``video_url`` via the codec
        backend.

        Returns ``(videos, video_metadata_list, sample_fps)`` where
        ``videos`` are the canvas arrays, ``video_metadata_list`` holds the
        matching ``CodecVideoOutput`` objects, and ``sample_fps`` is the fps
        of the last loaded video (or ``None`` if no video was present).
        """
        videos: List[Any] = []
        video_metadata_list: List[Any] = []
        sample_fps: Optional[float] = None

        for message in messages:
            for content in message["content"]:
                if content.get("type") != "video_url":
                    continue
                video_url = content["video_url"]
                extra = {k: v for k, v in video_url.items() if k != "url" and v is not None}
                frames, sample_fps, codec_output = self.load_codec_videos(
                    video_url["url"],
                    data_folder=data_folder,
                    fps=self.config.fps,
                    video_kwargs=extra or None,
                )
                videos.append(frames)
                video_metadata_list.append(codec_output)

        return videos, video_metadata_list, sample_fps
