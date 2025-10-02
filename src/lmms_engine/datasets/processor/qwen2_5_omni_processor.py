from typing import List, Optional

import numpy as np
import torch
from PIL.Image import Image
from transformers import Qwen2_5OmniProcessor
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
    Qwen2_5OmniProcessorKwargs,
)

from lmms_engine.mapping_func import register_processor

from .base_qwen2_5_processor import BaseQwen2_5_DataProcessor


@register_processor("Qwen2_5OmniProcessor")
class Qwen2_5OmniDataProcessor(BaseQwen2_5_DataProcessor):
    def _build_processor(self):
        model_path = getattr(self.config, "processor_path", self.config.processor_name)
        processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=False
        )

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

        # Set audio processor parameters
        audio_max_length = self.config.extra_kwargs.get("audio_max_length", None)
        if audio_max_length and hasattr(processor, "audio_processor"):
            processor.audio_processor.max_length = audio_max_length

        return processor

    def build(self):
        # Override build to handle Qwen2.5-Omni specifics
        self.processor = self._build_processor()
        # Don't override chat_template for Qwen2.5-Omni as it has its own

    @property
    def audio_processor(self):
        # For Qwen2.5-Omni, audio processing is done via feature_extractor
        # Create a wrapper to make it compatible with parent's expectations
        return self.processor.feature_extractor

    @property
    def audio_token_id(self):
        # Return the audio token ID if processor has one
        if hasattr(self.processor, "audio_token_id"):
            return self.processor.audio_token_id
        # Fallback: try to get from tokenizer
        if hasattr(self.tokenizer, "audio_token_id"):
            return self.tokenizer.audio_token_id
        # Try to convert the audio token string to ID
        if hasattr(self.processor, "audio_token") and self.processor.audio_token:
            return self.tokenizer.convert_tokens_to_ids(self.processor.audio_token)
        return None

    @property
    def tokenizer(self):
        # Return the tokenizer from the processor
        return self.processor.tokenizer

    @property
    def sampling_rate(self):
        # Qwen2.5-Omni uses feature_extractor instead of audio_processor
        return self.processor.feature_extractor.sampling_rate
