from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import (
    Qwen2_5_VLProcessorKwargs,
)

from .aero_processor import AeroDataProcessor


class BaseQwen2_5_DataProcessor(AeroDataProcessor):
    def _build_processor(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

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
        """
        A wrapper method to process single data
        """

        if hasattr(self.processor, "_merge_kwargs"):
            output_kwargs = self.processor._merge_kwargs(
                Qwen2_5_VLProcessorKwargs,
                tokenizer_init_kwargs=self.tokenizer.init_kwargs,
                **kwargs,
            )
        else:
            output_kwargs = kwargs

        image_inputs = {}
        videos_inputs = {}
        audio_inputs = {}

        if images is not None:
            # Handle smart resize edge case
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
                {"second_per_grid_ts": torch.tensor(second_per_grid_ts)}
            )
            merge_length = self.processor.video_processor.merge_size**2
            num_video_tokens = [
                (video_grid_thw[index].prod() // merge_length)
                for index in range(len(video_grid_thw))
            ]
        else:
            num_video_tokens = None

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
            inputs["audio_values"] = audio_inputs["audio_values"]
            inputs["audio_attention_mask"] = audio_inputs["audio_attention_mask"]
        if videos is not None:
            for key, value in videos_inputs.items():
                inputs[key] = value

        return inputs

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
        special_tokens = self.processor.tokenizer.additional_special_tokens
        special_tokens.extend(["<|im_start|>", "<|im_end|>"])
        unmask_tokens_idx = [
            self.processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens
        ]
        input_id, target = [], []
        image_start_from = 0
        audio_start_from = 0
        video_start_from = 0
        if add_system_prompt and hf_messages[0]["role"] != "system":
            input_id += self.processor.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_message}],
            )
            target += [-100] * len(input_id)
        for message in hf_messages:
            role = message["role"]
            # Cautions, qwen2_5 vl tokenizer wrap into a list
            encode_id = self.processor.apply_chat_template([message], tokenize=True)[0]
            # Should be 3 if instead of if else, so that can expand for each case
            if self.image_token_id in encode_id:
                encode_id, used_images = self._expand_encode_id_image_tokens(
                    encode_id, num_image_tokens, image_start_from
                )
                image_start_from += used_images
            if self.audio_token_id in encode_id:
                encode_id, used_audio = self._expand_encode_id_audio_tokens(
                    encode_id, num_audio_tokens, audio_start_from
                )
                audio_start_from += used_audio
            if self.video_token_id in encode_id:
                encode_id, used_video = self._expand_encode_id_video_tokens(
                    encode_id, num_video_tokens, video_start_from
                )
                video_start_from += used_video

            input_id += encode_id
            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            else:
                # Adopted from llava-ov that mask out the assistant
                encode_id[:3] = [-100] * 3
                target += encode_id

        if add_generation_prompt:
            generation_tokens = self.processor.tokenizer.encode(
                "<|im_start|>assistant\n"
            )
            input_id += generation_tokens
            target += [-100] * len(generation_tokens)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == self.image_token_id:
                target[idx] = -100
            if encode_id == self.audio_token_id:
                target[idx] = -100
            if encode_id == self.video_token_id:
                target[idx] = -100

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return dict(
            input_ids=input_id,
            labels=target,
        )

    @property
    def chat_template_no_system(self):
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% set image_count = namespace(value=0) %}"
            "{% set video_count = namespace(value=0) %}"
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{% if message['content'] is string %}"
            "{{ message['content'] }}<|im_end|>\n"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if 'audio' in content or 'audio_url' in content %}"
            "{% set audio_count.value = audio_count.value + 1 %}"
            "<|AUDIO|>\n"
            "{% elif content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
            "{% set image_count.value = image_count.value + 1 %}"
            "{% if add_vision_id %}"
            "Picture {{ image_count.value }}: "
            "{% endif %}"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            "{% elif content['type'] == 'video' or 'video' in content %}"
            "{% set video_count.value = video_count.value + 1 %}"
            "{% if add_vision_id %}"
            "Video {{ video_count.value }}: "
            "{% endif %}"
            "<|vision_start|><|video_pad|><|vision_end|>\n"
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
        # fmt: on
