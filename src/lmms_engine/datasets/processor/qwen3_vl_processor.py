from typing import List, Optional

import numpy as np
import torch
from PIL.Image import Image
from transformers import Qwen3VLProcessor
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs

from lmms_engine.mapping_func import register_processor

from .base_qwen2_5_processor import BaseQwen2_5_DataProcessor


@register_processor("qwen3_vl")
class Qwen3_VLDataProcessor(BaseQwen2_5_DataProcessor):
    def _build_processor(self):
        processor = Qwen3VLProcessor.from_pretrained(self.config.processor_name)

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
        assert audios is None, "Qwen3_VLDataProcessor does not support audio"
        output_kwargs = self.processor._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.processor.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.processor.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            # If user has not requested video metadata, pop it
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None
            video_metadata = None

        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            num_image_tokens = [grid_thw.prod() // merge_length for grid_thw in image_grid_thw]
        else:
            num_image_tokens = None

        if video_grid_thw is not None:
            merge_length = self.processor.video_processor.merge_size**2
            num_video_tokens = [grid_thw[1:].prod() // (merge_length**2) for grid_thw in video_grid_thw]
        else:
            num_video_tokens = None

        inputs = self.get_qwen_template_labels(
            hf_messages,
            num_image_tokens,
            num_video_tokens,
            video_metadata,
            video_grid_thw,
            system_message,
            add_system_prompt,
            add_generation_prompt,
        )

        if images is not None:
            for key, value in image_inputs.items():
                inputs[key] = value
        if videos is not None:
            for key, value in videos_inputs.items():
                inputs[key] = value

        return inputs

    def get_qwen_template_labels(
        self,
        hf_messages,
        num_image_tokens: List[int],
        num_video_tokens: List[int],
        video_metadata: List[dict],
        video_grid_thw=None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
        add_generation_prompt: bool = False,
    ):
        special_tokens = self.processor.tokenizer.additional_special_tokens
        special_tokens.extend(["<|im_start|>", "<|im_end|>"])
        unmask_tokens_idx = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens]
        input_id, target = [], []
        image_start_from = 0
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
            if self.video_token_id in encode_id:
                # Qwen3 VL new logic, build timestamp for different video frames
                metadata = video_metadata[video_start_from]
                if metadata.fps is None:
                    metadata.fps = 24 if metadata.fps is None else metadata.fps
                curr_timestamp = self.processor._calculate_timestamps(
                    metadata.frames_indices,
                    metadata.fps,
                    self.processor.video_processor.merge_size,
                )
                encode_id, used_video = self._expand_encode_id_video_tokens(
                    encode_id,
                    num_video_tokens,
                    video_start_from,
                    curr_timestamp,
                    video_grid_thw,
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
            generation_tokens = self.processor.tokenizer.encode("<|im_start|>assistant\n")
            input_id += generation_tokens
            target += [-100] * len(generation_tokens)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == self.image_token_id:
                target[idx] = -100
            if encode_id == self.video_token_id:
                target[idx] = -100

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return dict(
            input_ids=input_id,
            labels=target,
        )

    def _expand_encode_id_video_tokens(
        self,
        encode_id: List[int],
        video_token_num: List[int],
        start_from: int = 0,
        curr_timestamp: List[str] = None,
        video_grid_thw: List[List[int]] = None,
    ):
        video_pos = [i for i, x in enumerate(encode_id) if x == self.video_token_id]
        expanded_encode_id = []
        prev = 0
        merge_length = self.processor.video_processor.merge_size**2
        for idx, pos in enumerate(video_pos):
            # Before image pos, no expand
            expanded_encode_id.extend(encode_id[prev:pos])
            # Image pos, expand
            frame_seq_len = video_grid_thw[idx + start_from][1:].prod() // merge_length
            for frame_idx in range(video_grid_thw[idx + start_from][0]):
                curr_time = curr_timestamp[frame_idx]
                timestamp_token = f"<{curr_time:.1f} seconds>"
                timestamp_token_id = self.processor.tokenizer.encode(timestamp_token)
                visual_tokens = [self.video_token_id] * frame_seq_len
                # Three cases
                # If first frame, the start token in being added to the expanded encode id already, no need to include
                # If last frame, the end token will be added to the expanded encode id later, no need to include
                # If middle frame, both start and end tokens need to be included
                if frame_idx == 0:
                    curr_expand_video_ids = timestamp_token_id + visual_tokens + [self.processor.vision_end_token_id]
                elif frame_idx == video_grid_thw[idx + start_from][0] - 1:
                    curr_expand_video_ids = [self.processor.vision_start_token_id] + timestamp_token_id + visual_tokens
                else:
                    curr_expand_video_ids = (
                        [self.processor.vision_start_token_id]
                        + timestamp_token_id
                        + visual_tokens
                        + [self.processor.vision_end_token_id]
                    )
                expanded_encode_id.extend(curr_expand_video_ids)
            prev = pos + 1

            if idx == len(video_pos) - 1:
                # Last image pos, Add the rest to the end
                expanded_encode_id.extend(encode_id[prev:])

        return expanded_encode_id, len(video_pos)
