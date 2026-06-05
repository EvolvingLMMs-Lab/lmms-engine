import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image

from lmms_engine.datasets.iterable.vision_iterable_dataset import (
    VisionSFTIterableDataset,
)
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("llava_ov2_iterable")
class LlavaOv2IterableDataset(VisionSFTIterableDataset):
    """Iterable dataset for LLaVA-OneVision-2 with codec-stream video input.

    Reuses ``VisionSFTIterableDataset`` plumbing but routes video loading
    through the ``lmms_video_utils`` backend so each video produces a
    ``CodecVideoOutput`` (canvases + patch_positions + source_pts) that the
    downstream processor can consume directly instead of re-deriving
    timestamps from frame index.
    """

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        images_list = []
        videos = []
        video_metadata_list = []
        kwargs: Dict[str, Any] = {}
        messages = data["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "video_url":
                    video_url = content["video_url"]
                    extra = {k: v for k, v in video_url.items() if k != "url" and v is not None}
                    frames, sample_fps, codec_output = self.load_videos(
                        video_url["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                        video_kwargs=extra or None,
                    )
                    videos.append(frames)
                    video_metadata_list.append(codec_output)
                    kwargs["fps"] = sample_fps

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if data_folder is not None:
            images = [Image.open(os.path.join(data_folder, image)) for image in images_list]
        else:
            images = [Image.open(image) for image in images_list]
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None
            video_metadata_list = None
        if video_metadata_list is not None:
            kwargs["video_metadata"] = video_metadata_list

        inputs = self.processor.process(images=images, hf_messages=hf_messages, videos=videos, **kwargs)
        return inputs

    def load_videos(
        self,
        video_path: str,
        data_folder=None,
        fps: int = 1,
        video_kwargs=None,
    ) -> Tuple[np.ndarray, float, Any]:
        assert (
            self.config.video_backend == "lmms_video_utils"
        ), "LlavaOv2IterableDataset only supports lmms_video_utils backend"
        if data_folder is not None:
            video_path = os.path.join(data_folder, video_path)
        return self.load_video_lmms_video_utils(video_path, fps, video_kwargs=video_kwargs)
