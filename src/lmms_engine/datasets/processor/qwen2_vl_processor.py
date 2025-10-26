from typing import List

import torch
from PIL import Image
from transformers import Qwen2VLProcessor
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessorKwargs

from lmms_engine.mapping_func import register_processor

from .config import ProcessorConfig


@register_processor("qwen2_vl")
class Qwen2VLDataProcessor:
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config

    def build(self):
        self.processor = self._build_processor()
        self.set_chat_template()

    def _build_processor(self):
        processor = Qwen2VLProcessor.from_pretrained(self.config.processor_name)
        return processor

    def process(self, images: List[Image.Image], hf_messages, videos=None, **kwargs):
        """
        A wrapper method to process single data
        """

        output_kwargs = self.processor._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.processor.image_processor(
                images=images,
                videos=None,
                return_tensors="pt",
                **output_kwargs["images_kwargs"],
            )
            image_grid_thw = image_inputs["image_grid_thw"]
            # pixel_values [pixel_patches, hidden_dim] (14308, 1176)
            # image_grid_thw [grid_t, grid_h, grid_w] (1, 98, 146)
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(
                images=None,
                videos=videos,
                return_tensors="pt",
                **output_kwargs["videos_kwargs"],
            )
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            image_token_num = [grid_thw.prod() // merge_length for grid_thw in image_grid_thw]
        else:
            image_token_num = []
        # Get input_ids, labels (seq_len,)
        inputs = self.get_qwen_inputs(hf_messages, image_token_num)
        inputs["pixel_values"] = image_inputs["pixel_values"]
        inputs["image_grid_thw"] = image_grid_thw

        return inputs

    def get_qwen_inputs(
        self,
        hf_messages,
        image_token_num: List[int],
        system_message: str = "You are a helpful assistant",
    ):
        special_tokens = self.processor.tokenizer.additional_special_tokens
        unmask_tokens_idx = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens]
        input_id, target = [], []

        # Image start from 0
        start_from = 0
        input_id += self.processor.tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [-100] * len(input_id)
        for message in hf_messages:
            role = message["role"]
            encode_id = self.processor.apply_chat_template([message], tokenize=True)
            if self.image_token_id in encode_id:
                encode_id, used_images = self._expand_encode_id_image_tokens(encode_id, image_token_num, start_from)
                start_from += used_images
            input_id += encode_id
            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx and encode_id != self.image_token_id:
                target[idx] = encode_id

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return dict(
            input_ids=input_id,
            labels=target,
        )

    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

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
            expanded_encode_id.extend([self.image_token_id] * image_token_num[idx + start_from])
            prev = pos + 1

            if idx == len(image_pos) - 1:
                # Last image pos, Add the rest to the end
                expanded_encode_id.extend(encode_id[prev:])

        return expanded_encode_id, len(image_pos)

    def set_chat_template(self):
        self.processor.chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    def reset_chat_template(self):
        self.processor.chat_template = (
            "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        )
