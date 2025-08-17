from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from lmms_engine.mapping_func import register_processor

from ...models.aero.processing_aero import AeroProcessor, AeroProcessorKwargs
from .config import ProcessorConfig


@register_processor("aero")
class AeroDataProcessor:
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config

    def build(self):
        self.processor = self._build_processor()
        self.processor.chat_template = self.chat_template_no_system

    def _build_processor(self):
        processor = AeroProcessor.from_pretrained(self.config.processor_name)
        return processor

    def save_pretrained(self, save_directory: str):
        if not hasattr(self, "processor"):
            raise ValueError(
                "Processor has not been built yet. Please call build() first."
            )
        # Build a clean processor for saving
        new_processor = self._build_processor()
        new_processor.save_pretrained(save_directory)

    def process(
        self,
        images: List[Image.Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        add_system_prompt=True,
        **kwargs,
    ):
        """
        A wrapper method to process single data
        """

        output_kwargs = self.processor._merge_kwargs(
            AeroProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        audio_inputs = {}

        if audios is not None:
            audio_inputs = self.processor.audio_processor(
                audios,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="max_length",
                return_tensors="pt",
                **kwargs,
            )
            audio_inputs["audio_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename attention_mask to prevent conflicts later on
            audio_inputs["audio_values"] = audio_inputs.pop("input_features")
            input_lengths = (audio_inputs["audio_attention_mask"].sum(-1) - 1) // 2 + 1
            num_audio_tokens = (input_lengths - 2) // 2 + 1
        else:
            num_audio_tokens = None

        inputs = self.get_qwen_template_labels(
            hf_messages,
            num_audio_tokens,
            add_system_prompt=add_system_prompt,
        )
        if audios is not None:
            inputs["audio_values"] = audio_inputs["audio_values"]
            inputs["audio_attention_mask"] = audio_inputs["audio_attention_mask"]

        return inputs

    def get_qwen_template_labels(
        self,
        hf_messages,
        num_audio_tokens: List[int],
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
    ):
        special_tokens = self.processor.tokenizer.additional_special_tokens
        special_tokens.extend(["<|im_start|>", "<|im_end|>"])
        unmask_tokens_idx = [
            self.processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens
        ]
        input_id, target = [], []
        # The purpose of start from is to record which mm token we are at. Supposing the format is interleaved
        # Then we need to record this so that the mm token can be expanded correctly per conversation
        # If the format is not interleaved, then nothing special (Say always at the from). Start from does not matter
        image_start_from = 0
        audio_start_from = 0
        video_start_from = 0

        if add_system_prompt and hf_messages[0]["role"] != "system":
            input_id += self.processor.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_message}]
            )
            target += [-100] * len(input_id)
        for message in hf_messages:
            role = message["role"]
            # Cautions, qwen2_5 vl tokenizer wrap into a list
            encode_id = self.processor.apply_chat_template([message], tokenize=True)[0]
            if self.audio_token_id in encode_id:
                encode_id, used_audio = self._expand_encode_id_audio_tokens(
                    encode_id, num_audio_tokens, audio_start_from
                )
                audio_start_from += used_audio
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
            if encode_id == self.audio_token_id:
                target[idx] = -100

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

    def _expand_encode_id_audio_tokens(
        self,
        encode_id: List[int],
        audio_token_num: List[int],
        start_from: int = 0,
    ):
        audio_pos = [i for i, x in enumerate(encode_id) if x == self.audio_token_id]
        expanded_encode_id = []
        prev = 0
        for idx, pos in enumerate(audio_pos):
            # Before image pos, no expand
            expanded_encode_id.extend(encode_id[prev:pos])
            # Image pos, expand
            expanded_encode_id.extend(
                [self.audio_token_id] * audio_token_num[idx + start_from]
            )
            prev = pos + 1

            if idx == len(audio_pos) - 1:
                # Last image pos, Add the rest to the end
                expanded_encode_id.extend(encode_id[prev:])

        return expanded_encode_id, len(audio_pos)

    def _expand_encode_id_video_tokens(
        self,
        encode_id: List[int],
        video_token_num: List[int],
        start_from: int = 0,
    ):
        video_pos = [i for i, x in enumerate(encode_id) if x == self.video_token_id]
        expanded_encode_id = []
        prev = 0
        for idx, pos in enumerate(video_pos):
            # Before image pos, no expand
            expanded_encode_id.extend(encode_id[prev:pos])
            # Image pos, expand
            expanded_encode_id.extend(
                [self.video_token_id] * video_token_num[idx + start_from]
            )
            prev = pos + 1

            if idx == len(video_pos) - 1:
                # Last image pos, Add the rest to the end
                expanded_encode_id.extend(encode_id[prev:])

        return expanded_encode_id, len(video_pos)

    @property
    def image_token_id(self):
        image_token = getattr(self.processor, "image_token", None)
        if image_token is None:
            return None
        else:
            return self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.image_token
            )

    @property
    def audio_token_id(self):
        audio_token = getattr(self.processor, "audio_token", None)
        if audio_token is None:
            return None
        else:
            return self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.audio_token
            )

    @property
    def video_token_id(self):
        video_token = getattr(self.processor, "video_token", None)
        if video_token is None:
            return None
        else:
            return self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.video_token
            )

    def get_input_mode(self, audios, images, videos):
        if audios is not None:
            if images is not None or videos is not None:
                # Audio Vision
                input_mode = torch.tensor([3], dtype=torch.int8)
            else:
                # Audio
                input_mode = torch.tensor([1], dtype=torch.int8)
        elif images is not None or videos is not None:
            # Vision
            input_mode = torch.tensor([2], dtype=torch.int8)
        else:
            # Text
            input_mode = torch.tensor([0], dtype=torch.int8)
        return input_mode

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def sampling_rate(self):
        return self.processor.audio_processor.sampling_rate

    @property
    def chat_template_no_system(self):
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{% if message['content'] is string %}"
            "{{ message['content'] }}<|im_end|>\n"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if 'audio' in content or 'audio_url' in content %}"
            "{% set audio_count.value = audio_count.value + 1 %}"
            "<|AUDIO|>\n"
            "{% elif 'text' in content %}"
            "{{ content['text'] }}"
            "{% endif %}"
            "{% endfor %}"
            "<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        )
