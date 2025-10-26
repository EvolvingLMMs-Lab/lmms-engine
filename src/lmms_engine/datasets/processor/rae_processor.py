import random
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import AutoProcessor

from lmms_engine.mapping_func import register_processor

from .base_qwen2_5_processor import BaseQwen2_5_DataProcessor


@register_processor("rae_siglip")
class RaeSiglipDataProcessor(BaseQwen2_5_DataProcessor):
    def _build_processor(self):
        # Handle local paths by using local_files_only and trust_remote_code
        processor = AutoProcessor.from_pretrained(
            self.config.processor_name, local_files_only=True, trust_remote_code=True
        )
        image_processor = processor.image_processor
        size_attr = getattr(image_processor, "size", 256)
        if isinstance(size_attr, dict):
            image_size = size_attr.get("shortest_edge") or size_attr.get("height") or size_attr.get("width")
        else:
            image_size = size_attr
        if image_size is None:
            image_size = 256
        self.image_size = int(image_size)
        patch_size = getattr(image_processor, "patch_size", 16)
        self.patch_size = int(patch_size)
        self.num_tokens = (self.image_size // self.patch_size) ** 2
        if self.image_size == 256:
            self.first_crop_size = 384
        else:
            self.first_crop_size = int(round(self.image_size * 1.5))
        return processor

    def process(
        self,
        images: List[Image.Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt=True,
        add_generation_prompt=False,  # Whether add a generation prompt at the end
        **kwargs,
    ):
        assert audios is None and videos is None, "RaeSiglipDataProcessor does not support audio and video"

        processed = []
        for image in images:
            image = image.convert("RGB")
            resized = image.resize(
                (self.first_crop_size, self.first_crop_size),
                resample=Image.BICUBIC,
            )
            if self.first_crop_size > self.image_size:
                max_offset = self.first_crop_size - self.image_size
                left = random.randint(0, max_offset)
                top = random.randint(0, max_offset)
            else:
                left = 0
                top = 0
            cropped = resized.crop((left, top, left + self.image_size, top + self.image_size))
            tensor = TF.to_tensor(cropped)
            processed.append(tensor)

        pixel_values = torch.stack(processed, dim=0)

        batch_size = pixel_values.shape[0]
        input_ids = torch.ones(batch_size, self.num_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    @property
    def audio_token_id(self):
        return None
