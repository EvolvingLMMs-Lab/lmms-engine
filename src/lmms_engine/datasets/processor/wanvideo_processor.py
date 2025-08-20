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

    def apply_prompt_template(self, prompt: str) -> str:
        """Apply prompt template for WanVideo."""
        # WanVideo uses direct prompts without special formatting
        return prompt

    def build(self):
        self.processor = WanVideoModelProcessor()

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
        if self.processor.tokenizer is not None:
            text_inputs = self.processor.tokenizer(
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
        elif images is not None and len(images) > 0:
            # For I2V mode, use the first image
            image_inputs = self.processor.image_processor.preprocess(
                images[0],
                return_tensors="pt",
            )
            pixel_values = image_inputs["pixel_values"]
        else:
            # Dummy pixel values if no visual input
            pixel_values = torch.zeros((1, 3, 8, 480, 832))

        # Prepare output dictionary
        output = {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "pixel_values": pixel_values.squeeze(0),
            "labels": text_inputs["input_ids"].squeeze(0).clone(),  # For training
        }

        # Add video-specific kwargs if provided
        # if video_kwargs:
        #     output.update(video_kwargs)

        return output
