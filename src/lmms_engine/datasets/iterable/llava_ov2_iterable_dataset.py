import json
import os
from typing import Any, Dict

import torch
from PIL import Image

from lmms_engine.datasets.codec_video_mixin import CodecVideoLoadingMixin
from lmms_engine.datasets.iterable.vision_iterable_dataset import (
    VisionSFTIterableDataset,
)
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("llava_ov2_iterable")
class LlavaOv2IterableDataset(CodecVideoLoadingMixin, VisionSFTIterableDataset):
    """Iterable dataset for LLaVA-OneVision-2 with codec-stream video input.

    Reuses ``VisionSFTIterableDataset`` plumbing but routes video loading
    through the ``lmms_video_utils`` backend (via ``CodecVideoLoadingMixin``)
    so each video produces a ``CodecVideoOutput`` (canvases + patch_positions
    + source_pts) that the downstream processor can consume directly instead
    of re-deriving timestamps from frame index.
    """

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        images_list = []
        kwargs: Dict[str, Any] = {}
        messages = data["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])

        videos, video_metadata_list, sample_fps = self.collect_codec_video_inputs(messages, data_folder=data_folder)
        if sample_fps is not None:
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
        else:
            kwargs["video_metadata"] = video_metadata_list

        inputs = self.processor.process(images=images, hf_messages=hf_messages, videos=videos, **kwargs)
        return inputs
