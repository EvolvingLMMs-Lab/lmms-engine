from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from lmms_engine.mapping_func import register_processor
from lmms_engine.models.wanvideo import WanVideoProcessor as WanVideoModelProcessor


@register_processor("wanvideo")
class WanVideoDataProcessor:
    def __init__(self, config, model_id=None):
        self.config = config
        self.model_id = model_id

    def apply_prompt_template(self, hf_messages: str) -> str:
        """Apply prompt template for WanVideo."""
        # WanVideo uses direct prompts without special formatting
        prompt = hf_messages[0]["content"][1]["text"]
        return prompt

    def save_pretrained(self, save_directory: str):
        # Build a clean processor for saving
        # wanvideo_kwargs = self.config.kwargs
        # new_processor = WanVideoModelProcessor(**wanvideo_kwargs)
        # new_processor.save_pretrained(save_directory)
        pass

    def build(self):
        wanvideo_kwargs = self.config["kwargs"]
        self.processor = WanVideoModelProcessor(**wanvideo_kwargs)
        self.tokenizer = self.processor.tokenizer

    def process(
        self, images: List[Image.Image], hf_messages, videos=None, **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single sample for WanVideo training.

        Args:
            images: List of images (for I2V mode)
            hf_messages: Text prompt/caption for the video
            videos: List of video frames
            video_kwargs: Additional video parameters (fps, etc.)

        Returns:
            Dictionary with processed inputs for training
        """
        if hf_messages is None:
            hf_messages = ""

        # Apply prompt template
        formatted_prompt = self.apply_prompt_template(hf_messages)

        # Process text
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
        else:
            # Dummy text inputs if no tokenizer
            text_inputs = {
                "input_ids": torch.zeros((1, 256), dtype=torch.long),
                "attention_mask": torch.ones((1, 256), dtype=torch.long),
            }

        # Process video frames
        if videos is not None and len(videos) > 0:
            # Videos is a list of frame lists
            video_frames = videos[0] if isinstance(videos[0], list) else videos
            # Process frames using the image processor
            video_inputs = self.processor.image_processor.preprocess(
                video_frames,
                return_tensors="pt",
            )
            pixel_values = video_inputs["pixel_values"]
        else:
            raise ValueError("No video frames provided")

        # Prepare output dictionary
        output = {
            "input_ids": text_inputs["input_ids"].squeeze(0),  # B, L
            # "attention_mask": text_inputs["attention_mask"].squeeze(0), # B, L
            "pixel_values": pixel_values.squeeze(0),  # B, T, C, H, W
            "labels": text_inputs["input_ids"]
            .squeeze(0)
            .clone(),  # For training # B, L
        }

        # Add video-specific kwargs if provided
        # if video_kwargs:
        #     output.update(video_kwargs)

        return output
