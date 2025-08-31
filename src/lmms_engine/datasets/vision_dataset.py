import os
import re
import base64
from typing import Dict
from io import BytesIO

import torch
from PIL import Image
import pathlib
import os
from lmms_engine.mapping_func import register_dataset

from ..utils.train_utils import TrainUtilities
from .collator import VisionCollator
from .multimodal_dataset import MultiModalDataset


@register_dataset("vision")
class VisionSFTDataset(MultiModalDataset):
    def load_from_csv(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        """Load from CSV data directly without intermediate transformation."""
        images_list = []
        videos = []
        kwargs = {}

        # Build messages directly from CSV data
        # Seems that this will force the user to provide a video url?
        # Maybe we need to add this check
        # Btw I'm not very sure if we are using video understanding or video generation
        if "video" in data and isinstance(data["video"], str):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": data["video"]}},
                        {"type": "text", "text": data["prompt"]},
                    ],
                }
            ]

        # Process video content directly
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
            images = [
                Image.open(os.path.join(data_folder, image)) for image in images_list
            ]
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
    
    def get_pil_image(self, image: os.PathLike | pathlib.Path | str | pathlib.Path | Image.Image | bytes, data_folder: os.PathLike | pathlib.Path | None = None) -> Image.Image:
        """
        Load an image from various input types and return a PIL Image.
        
        Args:
            image: Can be a file path, URL, PIL Image object, bytes data, or base64 string
            
        Returns:
            PIL Image object
        """
        if data_folder is not None:
            data_folder = pathlib.Path(data_folder)
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, bytes):
            # Handle raw bytes data
            return Image.open(BytesIO(image))
        elif isinstance(image, (str, os.PathLike, pathlib.Path)):
            # Check if it's a base64 data URI (e.g., "data:image/jpeg;base64,...")
            if isinstance(image, str) and image.startswith('data:image/'):
                # Extract base64 data after the comma
                header, base64_data = image.split(',', 1)
                image_bytes = base64.b64decode(base64_data)
                return Image.open(BytesIO(image_bytes))
            
            # Check if it's a URL
            if isinstance(image, str) and image.startswith(('http://', 'https://')):
                import requests
                response = requests.get(image)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            else:
                # Local file path
                image = data_folder / image if data_folder is not None else image
                return Image.open(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        # TODO Write a protocol for vision openai input
        images_list = []
        videos = []
        kwargs = {}
        messages = data["messages"]
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "image_col":
                    image_col, image_idx = re.match(
                        r"(\w+)\[(\d+)\]", content["image_col"]
                    ).groups()
                    images_list.append(data[image_col][int(image_idx)])
                elif content["type"] == "video_url":
                    # Loading videos with fps
                    frames, sample_fps = self.load_videos(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    videos.append(frames)
                    # Update kwargs
                    kwargs["fps"] = sample_fps

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        images = [self.get_pil_image(image, data_folder=data_folder) for image in images_list]
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None
        inputs = self.processor.process(
            images=images, hf_messages=hf_messages, videos=videos, **kwargs
        )
        return inputs

    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        # TODO: Add video support
        messages = data["messages"]
        images = []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images.append(content["image_url"]["url"])
                elif content["type"] == "image_col":
                    image_col, image_idx = re.match(
                        r"(\w+)\[(\d+)\]", content["image_col"]
                    ).groups()
                    images.append(data[image_col][int(image_idx)])
        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        inputs = self.processor.process(images=images, hf_messages=hf_messages)
        return inputs

    def get_collator(self):
        return VisionCollator(self.processor)
