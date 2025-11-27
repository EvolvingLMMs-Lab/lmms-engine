from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import LlavaOnevisionProcessor
from transformers.models.llava_onevision.processing_llava_onevision import (
    LlavaOnevisionProcessorKwargs,
)

from lmms_engine.mapping_func import register_processor

from .config import ProcessorConfig
from .llava_processor import LLaVADataProcessor


@register_processor("llava_video")
class LLaVAVideoDataProcessor(LLaVADataProcessor):
    def __init__(self, config: ProcessorConfig) -> None:
        super().__init__(config)
        self.video_token = "<video>"
        self.image_token = "<image>"

    def process(
        self,
        images: Optional[List[Image.Image]] = None,
        hf_messages: List[dict] = None,
        videos: Optional[List[torch.Tensor]] = None,
        video_metadata: Optional[dict] = None,
        **kwargs,
    ):
        # Extract slow-fast parameters before _merge_kwargs filters them
        slow_fast_params = {}
        for key in ["faster_token_stride", "mm_spatial_pool_stride", "mm_spatial_pool_mode"]:
            if key in kwargs:
                slow_fast_params[key] = kwargs.pop(key)

        output_kwargs = self.processor._merge_kwargs(
            LlavaOnevisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        num_image_tokens = []
        num_video_tokens = []
        image_inputs = {}
        video_inputs = {}

        # Process images
        if images is not None and len(images) > 0:
            image_inputs = self.processor.image_processor(images, return_tensors="pt", **output_kwargs["images_kwargs"])
            height = image_inputs["pixel_values"].shape[-2]
            width = image_inputs["pixel_values"].shape[-1]

            for image_size in image_inputs["image_sizes"]:
                num_tokens = self.processor._get_number_of_features(
                    image_size[0].item(), image_size[1].item(), height, width
                )
                num_image_tokens.append(num_tokens)

        # Process videos - align with LlavaOnevisionProcessor (processing_llava_onevision.py line 177-190)
        if videos is not None and len(videos) > 0:
            # Use video_processor from transformers with return_tensors="pt" like image_processor
            video_inputs = self.processor.video_processor(
                videos, return_tensors="pt", **output_kwargs.get("videos_kwargs", {})
            )

            # Calculate num_video_tokens with slow-fast support
            pixel_values_videos = video_inputs.get("pixel_values_videos", [])

            for one_video in pixel_values_videos:
                # one_video shape: [num_frames, C, H, W]
                if isinstance(one_video, (list, tuple)):
                    import numpy as np

                    one_video = np.array(one_video)

                num_frames = one_video.shape[0]

                # Calculate tokens with slow-fast frame support
                patches_height_width = int(self.processor.num_image_tokens**0.5)  # sqrt
                num_tokens = self._calculate_video_tokens(num_frames, patches_height_width, **slow_fast_params)

                num_video_tokens.append(num_tokens)

        # Get tokenized inputs with labels
        inputs = self.get_video_template_labels(hf_messages, num_image_tokens, num_video_tokens)

        # Add visual inputs - separate storage like transformers
        if images is not None and len(image_inputs) > 0:
            inputs["pixel_values"] = image_inputs["pixel_values"]
            inputs["image_sizes"] = image_inputs["image_sizes"]

        if videos is not None and len(video_inputs) > 0:
            inputs["pixel_values_videos"] = video_inputs.get("pixel_values_videos")

        return inputs

    def get_video_template_labels(
        self,
        hf_messages: List[dict],
        num_image_tokens: List[int],
        num_video_tokens: List[int],
    ):
        """
        Tokenize messages and create labels with proper image/video token expansion.

        Args:
            hf_messages: Messages in HuggingFace format
            num_image_tokens: Number of tokens for each image
            num_video_tokens: Number of tokens for each video

        Returns:
            Dict with input_ids and labels
        """
        image_token_index = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        video_token_index = self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)
        special_tokens = self.processor.tokenizer.additional_special_tokens
        unmask_tokens_idx = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens]

        input_id, target = [], []
        image_idx = 0
        video_idx = 0

        for message in hf_messages:
            role = message["role"]
            encode_id = self.processor.apply_chat_template([message], tokenize=True)[0]

            # Expand image tokens
            if image_token_index in encode_id and num_image_tokens and image_idx < len(num_image_tokens):
                encode_id, used_images = self._expand_visual_tokens(
                    encode_id, num_image_tokens, image_idx, image_token_index
                )
                image_idx += used_images

            # Expand video tokens
            if video_token_index in encode_id and num_video_tokens and video_idx < len(num_video_tokens):
                encode_id, used_videos = self._expand_visual_tokens(
                    encode_id, num_video_tokens, video_idx, video_token_index
                )
                video_idx += used_videos

            input_id += encode_id

            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            else:
                # Mask out the assistant prefix tokens
                encode_id_copy = list(encode_id)
                encode_id_copy[:3] = [-100] * 3
                target += encode_id_copy

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        # Handle special tokens and image/video tokens in labels
        for idx, token_id in enumerate(input_id):
            if token_id in unmask_tokens_idx:
                target[idx] = token_id
            if token_id == image_token_index:
                target[idx] = image_token_index
            if token_id == video_token_index:
                target[idx] = video_token_index

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return dict(
            input_ids=input_id,
            labels=target,
        )

    def _expand_visual_tokens(
        self,
        encode_id: List[int],
        num_tokens: List[int],
        start_from: int = 0,
        token_id: int = None,
    ) -> Tuple[List[int], int]:
        """
        Expand image/video placeholder tokens to actual number of visual tokens.

        Args:
            encode_id: Tokenized message
            num_tokens: List of token counts for each visual input
            start_from: Starting index in num_tokens
            token_id: The specific token ID to expand (image or video token)

        Returns:
            Tuple of (expanded_encode_id, number_of_visuals_used)
        """
        if token_id is None:
            token_id = self.image_token_id

        visual_pos = [i for i, x in enumerate(encode_id) if x == token_id]

        expanded_encode_id = []
        prev = 0

        for idx, pos in enumerate(visual_pos):
            # Add tokens before the visual placeholder
            expanded_encode_id.extend(encode_id[prev:pos])

            # Expand the placeholder to actual number of tokens
            if (idx + start_from) < len(num_tokens):
                expanded_encode_id.extend([token_id] * num_tokens[idx + start_from])
            else:
                # Fallback if num_tokens doesn't have enough entries
                expanded_encode_id.append(token_id)

            prev = pos + 1

            # Add remaining tokens after last visual placeholder
            if idx == len(visual_pos) - 1:
                expanded_encode_id.extend(encode_id[prev:])

        return expanded_encode_id, len(visual_pos)

    def _calculate_video_tokens(
        self,
        num_frames: int,
        patches_height_width: int,
        **kwargs,
    ) -> int:
        """
        Calculate the number of video tokens based on slow-fast configuration.

        This aligns with the actual token generation in llava_video_forward.py.
        When slow-fast mode parameters are present, different frames are pooled with
        different strides, and each frame gets a faster_token appended.

        Args:
            num_frames: Number of frames in the video
            patches_height_width: Height/width of the patch grid (sqrt of num_image_tokens)
            **kwargs: Additional arguments that may contain slow-fast config

        Returns:
            Total number of video tokens including newline token
        """
        # Get slow-fast configuration from multiple sources (in order of priority):
        # 1. From kwargs (passed at runtime)
        # 2. From self.config.extra_kwargs (from dataset config)
        # 3. If faster_token_stride is not present, use standard mode

        extra_kwargs = getattr(self.config, "extra_kwargs", {}) or {}

        faster_token_stride = kwargs.get("faster_token_stride", extra_kwargs.get("faster_token_stride", None))
        mm_spatial_pool_stride = kwargs.get("mm_spatial_pool_stride", extra_kwargs.get("mm_spatial_pool_stride", 2))

        # If faster_token_stride is configured, enable slow-fast mode
        if faster_token_stride is not None and mm_spatial_pool_stride > 1:
            # Slow-fast mode: calculate tokens for mixed stride frames
            # Slow frames: pooled with stride=mm_spatial_pool_stride
            # Fast frames: pooled with stride=mm_spatial_pool_stride*2
            # Each frame gets +1 for faster_token

            pooled_slow = (patches_height_width + 1) // mm_spatial_pool_stride
            pooled_fast = (patches_height_width + 1) // (mm_spatial_pool_stride * 2)

            # Count slow and fast frames
            num_slow_frames = (num_frames + faster_token_stride - 1) // faster_token_stride
            num_fast_frames = num_frames - num_slow_frames

            # Calculate tokens: each pooled frame + faster_token
            slow_tokens = num_slow_frames * (pooled_slow * pooled_slow + 1)  # +1 for faster_token per frame
            fast_tokens = num_fast_frames * (pooled_fast * pooled_fast + 1)  # +1 for faster_token per frame

            num_tokens = slow_tokens + fast_tokens + 1  # +1 for final newline token
        else:
            # Standard mode: all frames pooled uniformly
            pooled_height_width = (patches_height_width + 1) // mm_spatial_pool_stride
            num_tokens = (num_frames * pooled_height_width * pooled_height_width) + 1  # +1 for newline

        return num_tokens

    @staticmethod
    def inject_time_instruction(
        messages: List[dict],
        video_time: float,
        num_frames: int,
        frame_time: str,
    ) -> List[dict]:
        """
        Inject time instruction into the first user message.

        This adds temporal context to help the model understand:
        - Total video duration
        - Number of sampled frames
        - Timestamps of each sampled frame

        Args:
            messages: List of conversation messages in OpenAI format
            video_time: Total video duration in seconds
            num_frames: Number of sampled frames
            frame_time: String of frame timestamps like "0.00s,0.50s,1.00s"

        Returns:
            Modified messages with time instruction injected
        """
        if not messages:
            return messages

        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames "
            f"are uniformly sampled from it. These frames are located at {frame_time}. "
            f"Please answer the following questions related to this video."
        )

        # Find the first user message
        for i, message in enumerate(messages):
            if message.get("role") == "user":
                content = message.get("content", "")

                if isinstance(content, list):
                    # Multi-modal content format
                    new_content = []
                    text_inserted = False

                    for item in content:
                        new_content.append(item)
                        # Insert time instruction after video/image placeholder
                        if item.get("type") in ["video_url", "image_url"] and not text_inserted:
                            new_content.append({"type": "text", "text": time_instruction})
                            text_inserted = True

                    if not text_inserted:
                        # Prepend if no video/image found
                        new_content.insert(0, {"type": "text", "text": time_instruction})

                    messages[i]["content"] = new_content
                else:
                    # Simple string content
                    messages[i]["content"] = f"{time_instruction}\n{content}"

                break

        return messages
