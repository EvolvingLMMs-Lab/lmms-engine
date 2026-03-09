import os
from typing import Dict

import torch
from PIL import Image

from lmms_engine.datasets.iterable.vision_iterable_dataset import (
    VisionSFTIterableDataset,
)
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("qwen3_5_iterable")
class Qwen3_5IterableDataset(VisionSFTIterableDataset):
    """Iterable dataset for Qwen3.5 text model training.

    Supports text-only and optionally image/video data.
    """

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        images_list = []
        videos = []
        kwargs = {}
        messages = data["messages"]

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

        hf_messages = TrainUtilities.convert_open_to_hf(messages)

        if data_folder is not None:
            images = [Image.open(os.path.join(data_folder, image)) for image in images_list]
        else:
            images = [Image.open(image) for image in images_list]

        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None

        inputs = self.processor.process(
            images=images, hf_messages=hf_messages, videos=videos, **kwargs
        )
        return inputs
