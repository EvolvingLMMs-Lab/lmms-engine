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
        self.processor = self._build_processor()

    @property
    def audio_processor(self):
        # For Qwen2.5Omni, audio processing is done via feature_extractor
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
        return self.processor.tokenizer

    @property
    def sampling_rate(self):
        # Qwen2.5Omni uses feature_extractor instead of audio_processor
        return self.processor.feature_extractor.sampling_rate

    def process(
        self,
        images: List[Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt=True,
        add_generation_prompt=False,
        **kwargs,
    ):
        if hasattr(self.processor, "_merge_kwargs"):
            output_kwargs = self.processor._merge_kwargs(
                Qwen2_5OmniProcessorKwargs,
                tokenizer_init_kwargs=self.tokenizer.init_kwargs,
                **kwargs,
            )
        else:
            output_kwargs = kwargs

        image_inputs = {}
        videos_inputs = {}
        audio_inputs = {}

        if images is not None:
            new_images = []
            for image in images:
                height = image.size[0]
                width = image.size[1]
                if width < 28 and height < 28:
                    image = image.resize((28, 28))
                elif height < 28:
                    image = image.resize((28, width))
                elif width < 28:
                    image = image.resize((height, 28))
                new_images.append(image)
            images = new_images
            image_inputs = self.processor.image_processor(
                images, return_tensors="pt", **output_kwargs["images_kwargs"]
            )
            image_inputs["image_sizes"] = image_inputs.pop("image_grid_thw")
            merge_size = self.processor.image_processor.merge_size
            num_image_tokens = [
                (image_size[-2] * image_size[-1]).item() // (merge_size**2)
                for image_size in image_inputs["image_sizes"]
            ]
        else:
            num_image_tokens = None

        if videos is not None:
            videos_inputs = self.processor.video_processor(
                videos=videos,
                **output_kwargs["videos_kwargs"],
                return_tensors="pt",
            )
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [
                    self.processor.video_processor.temporal_patch_size / fps
                ] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [
                    self.processor.video_processor.temporal_patch_size / tmp
                    for tmp in fps
                ]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update(
                {"video_second_per_grid": torch.tensor(second_per_grid_ts)}
            )
            merge_length = self.processor.video_processor.merge_size**2
            num_video_tokens = [
                (video_grid_thw[index].prod() // merge_length)
                for index in range(len(video_grid_thw))
            ]
        else:
            num_video_tokens = None

        if audios is not None:
            audio_inputs = self.audio_processor(
                audios,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="max_length",
                return_tensors="pt",
                **kwargs,
            )
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
            audio_inputs["audio_feature_lengths"] = (
                audio_inputs["feature_attention_mask"].sum(-1) - 1
            ) // 2 + 1
            num_audio_tokens = (audio_inputs["audio_feature_lengths"] - 2) // 2 + 1
        else:
            num_audio_tokens = None

        inputs = self.get_qwen_template_labels(
            hf_messages,
            num_image_tokens,
            num_audio_tokens,
            num_video_tokens,
            system_message=system_message,
            add_system_prompt=add_system_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        if images is not None:
            inputs["pixel_values"] = image_inputs["pixel_values"]
            inputs["image_grid_thw"] = image_inputs["image_sizes"]
        if audios is not None:
            inputs["input_features"] = audio_inputs["input_features"]
            inputs["feature_attention_mask"] = audio_inputs["feature_attention_mask"]
            inputs["audio_feature_lengths"] = audio_inputs["audio_feature_lengths"]
        if videos is not None:
            for key, value in videos_inputs.items():
                inputs[key] = value
        if "use_audio_in_video" in kwargs:
            inputs["use_audio_in_video"] = kwargs["use_audio_in_video"]

        return inputs
