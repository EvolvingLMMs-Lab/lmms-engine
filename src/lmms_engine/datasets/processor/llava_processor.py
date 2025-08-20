from typing import List

import torch
from PIL import Image
from transformers import LlavaOnevisionProcessor
from transformers.models.llava_onevision.processing_llava_onevision import (
    LlavaOnevisionProcessorKwargs,
)

from lmms_engine.mapping_func import register_processor

from .config import ProcessorConfig


@register_processor("llava")
class LLaVADataProcessor:
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config

    def build(self):
        self.processor = self._build_processor()

    def _build_processor(self):
        processor = LlavaOnevisionProcessor.from_pretrained(self.config.processor_name)
        return processor

    def save_pretrained(self, output_dir: str):
        self.processor.save_pretrained(output_dir)

    def process(self, images: List[Image.Image], hf_messages, videos=None, **kwargs):
        """
        A wrapper method to process single data
        """

        output_kwargs = self.processor._merge_kwargs(
            LlavaOnevisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        video_inputs = {}

        if images is not None:
            image_inputs = self.processor.image_processor(
                images, return_tensors="pt", **output_kwargs["images_kwargs"]
            )

            image_sizes = iter(image_inputs["image_sizes"])
            height = image_inputs["pixel_values"].shape[-2]
            width = image_inputs["pixel_values"].shape[-1]
            image_sizes = image_inputs["image_sizes"]
            num_image_tokens = [
                self.processor._get_number_of_features(
                    image_size[0], image_size[1], height, width
                )
                for image_size in image_sizes
            ]
        else:
            num_image_tokens = None

        inputs = self.get_qwen_template_labels(hf_messages, num_image_tokens)
        if images is not None:
            inputs["pixel_values"] = image_inputs["pixel_values"]
            inputs["image_sizes"] = image_inputs["image_sizes"]

        return inputs

    def get_qwen_template_labels(self, hf_messages, num_image_tokens: List[int]):
        image_token_index = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
        special_tokens = self.processor.tokenizer.additional_special_tokens
        unmask_tokens_idx = [
            self.processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens
        ]
        input_id, target = [], []
        start_from = 0
        for message in hf_messages:
            role = message["role"]
            encode_id = self.processor.apply_chat_template([message], tokenize=True)[0]
            # If num image tokens is not None, it means we have images in the batch
            # otherwise something like <image> tag in html is used
            if image_token_index in encode_id and num_image_tokens is not None:
                encode_id, used_images = self._expand_encode_id_image_tokens(
                    encode_id, num_image_tokens, start_from
                )
                start_from += used_images
            input_id += encode_id
            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            else:
                # Adopted from llava-ov that mask out the assistant
                encode_id[:3] = [-100] * 3
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                # Revert image so that we recognize it in labels
                # Unmask later
                target[idx] = image_token_index

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return dict(
            input_ids=input_id,
            labels=target,
        )

    def _expand_encode_id_image_tokens(
        self,
        encode_id: List[int],
        image_token_num: List[int],
        start_from: int = 0,
    ):
        image_pos = [i for i, x in enumerate(encode_id) if x == self.image_token_id]
        expanded_encode_id = []
        prev = 0
        for idx, pos in enumerate(image_pos):
            # Before image pos, no expand
            expanded_encode_id.extend(encode_id[prev:pos])
            # Image pos, expand
            expanded_encode_id.extend(
                [self.image_token_id] * image_token_num[idx + start_from]
            )
            prev = pos + 1

            if idx == len(image_pos) - 1:
                # Last image pos, Add the rest to the end
                expanded_encode_id.extend(encode_id[prev:])

        return expanded_encode_id, len(image_pos)

    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )

    @property
    def tokenizer(self):
        return self.processor.tokenizer
