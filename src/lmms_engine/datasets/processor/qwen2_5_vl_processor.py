from typing import List, Optional

import numpy as np
from PIL.Image import Image
from transformers import Qwen2_5_VLProcessor

from lmms_engine.mapping_func import register_processor

from .base_qwen2_5_processor import BaseQwen2_5_DataProcessor


@register_processor("qwen2_5_vl")
class Qwen2_5_VLDataProcessor(BaseQwen2_5_DataProcessor):
    def _build_processor(self):
        processor = Qwen2_5_VLProcessor.from_pretrained(self.config.processor_name)

        # Set image processor parameters
        image_max_pixels = self.config.extra_kwargs.get("image_max_pixels", None)
        image_min_pixels = self.config.extra_kwargs.get("image_min_pixels", None)
        if image_max_pixels:
            processor.image_processor.max_pixels = image_max_pixels
        if image_min_pixels:
            processor.image_processor.min_pixels = image_min_pixels

        # Set video processor parameters
        video_max_pixels = self.config.extra_kwargs.get("video_max_pixels", None)
        video_min_pixels = self.config.extra_kwargs.get("video_min_pixels", None)
        if video_max_pixels:
            processor.video_processor.max_pixels = video_max_pixels
        if video_min_pixels:
            processor.video_processor.min_pixels = video_min_pixels
        return processor

    def process(
        self,
        images: List[Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt=True,
        add_generation_prompt=False,  # Whether add a generation prompt at the end
        **kwargs,
    ):
        assert audios is None, "Qwen2_5_VLDataProcessor does not support audio"
        return super().process(
            images,
            hf_messages,
            audios,
            sampling_rate,
            videos,
            system_message,
            add_system_prompt,
            add_generation_prompt,
            **kwargs,
        )

    @property
    def audio_token_id(self):
        return None
