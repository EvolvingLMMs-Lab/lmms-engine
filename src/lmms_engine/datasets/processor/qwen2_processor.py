from typing import List, Optional

import numpy as np
import torch
from PIL.Image import Image
from transformers import AutoProcessor

from lmms_engine.mapping_func import register_processor
from lmms_engine.utils import DataUtilities

from .base_qwen2_5_processor import BaseQwen2_5_DataProcessor


@register_processor("qwen2")
class Qwen2DataProcessor(BaseQwen2_5_DataProcessor):
    def _build_processor(self):
        processor = AutoProcessor.from_pretrained(self.config.processor_name)
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
        assert audios is None, "Qwen2DataProcessor does not support audio"
        assert videos is None, "Qwen2DataProcessor does not support video"
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

    @property
    def tokenizer(self):
        return self.processor

    @property
    def special_tokens(self):
        if not hasattr(self, "_special_tokens"):
            self._special_tokens = DataUtilities.get_special_tokens(
                self.processor, extra_tokens=["<|im_start|>", "<|im_end|>"]
            )
        return self._special_tokens

    def get_qwen_template_labels(
        self,
        hf_messages,
        num_image_tokens: List[int],
        num_audio_tokens: List[int],
        num_video_tokens: List[int],
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
        add_generation_prompt: bool = False,
    ):
        unmask_tokens_idx = [self.processor.convert_tokens_to_ids(t) for t in self.special_tokens]
        input_id, target = [], []
        if add_system_prompt and hf_messages[0]["role"] != "system":
            input_id += DataUtilities.apply_chat_template(
                self.processor, [{"role": "system", "content": system_message}]
            )
            target += [-100] * len(input_id)
        for message in hf_messages:
            role = message["role"]
            encode_id = DataUtilities.apply_chat_template(self.processor, [message])

            input_id += encode_id
            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            else:
                # Adopted from llava-ov that mask out the assistant
                encode_id[:3] = [-100] * 3
                target += encode_id

        if add_generation_prompt:
            generation_tokens = self.processor.encode("<|im_start|>assistant\n")
            input_id += generation_tokens
            target += [-100] * len(generation_tokens)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return dict(
            input_ids=input_id,
            labels=target,
        )
