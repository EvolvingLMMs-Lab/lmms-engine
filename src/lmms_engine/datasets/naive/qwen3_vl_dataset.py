import os
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import fetch_video

from lmms_engine.datasets.collator import VisionCollator
from lmms_engine.datasets.naive.vision_dataset import VisionSFTDataset
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("qwen3_vl")
class Qwen3VLDataset(VisionSFTDataset):
    """
    Naive (map-style) dataset for Qwen3 Vision-Language model.

    This dataset uses qwen_vl_utils for video loading and returns video metadata
    alongside frames for proper processing.
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
                    # Loading videos with fps and metadata
                    frames, video_metadata, sample_fps = self.load_videos(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    videos.append(frames)
                    # Update kwargs with Qwen3VL-specific parameters
                    kwargs["fps"] = sample_fps
                    kwargs["video_metadata"] = video_metadata
                    kwargs["do_sample_frames"] = False

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if data_folder is not None:
            images = [Image.open(os.path.join(data_folder, image)) for image in images_list]
        else:
            images = [Image.open(image) for image in images_list]
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None
        inputs = self.processor.process(images=images, hf_messages=hf_messages, videos=videos, **kwargs)
        return inputs

    def load_videos(self, video_path: str, data_folder=None, fps: int = 1):
        """
        Load videos using qwen_vl_utils backend with metadata support.

        Args:
            video_path: Path to video file
            data_folder: Optional folder path to prepend
            fps: Target frames per second

        Returns:
            Tuple of (video frames, video metadata, sample fps)
        """
        if data_folder is not None:
            video_path = os.path.join(data_folder, video_path)

        assert self.config.video_backend == "qwen_vl_utils", "Qwen3VLDataset only supports qwen_vl_utils backend"
        frames, video_metadata, sample_fps = self.load_video_qwen_vl_utils(video_path, fps)
        return frames, video_metadata, sample_fps

    def load_video_qwen_vl_utils(
        self,
        video_path: str,
        fps: int,
    ) -> Tuple[np.ndarray, dict, float]:
        """
        Load video using Qwen VL utils with metadata support.

        Args:
            video_path: Path to video file
            fps: Target frames per second

        Returns:
            Tuple of (video frames, video metadata, sample fps)
        """
        video_dict = {
            "type": "video",
            "video": f"file://{video_path}",
            "min_frames": 1,
            "max_pixels": self.config.video_max_pixels,
            "max_frames": self.config.video_max_frames,
            "min_pixels": self.config.video_min_pixels,
        }

        if self.config.video_sampling_strategy == "frame_num":
            n_frames = self.config.frame_num
            video_dict["nframes"] = n_frames
            video_inputs, sample_fps = fetch_video(video_dict, return_video_sample_fps=True, return_video_metadata=True)
            frames, video_metadata = video_inputs
            frames = frames.numpy()
            return frames, video_metadata, sample_fps
        elif self.config.video_sampling_strategy == "fps":
            video_dict["fps"] = fps
            video_inputs, sample_fps = fetch_video(video_dict, return_video_sample_fps=True, return_video_metadata=True)
            frames, video_metadata = video_inputs
            frames = frames.numpy()
            return frames, video_metadata, sample_fps
        else:
            raise ValueError(f"Invalid video sampling strategy: {self.config.video_sampling_strategy}")

    def get_collator(self):
        return VisionCollator(self.processor)
