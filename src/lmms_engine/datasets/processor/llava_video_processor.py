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
    """
    LLaVA-Video data processor with video support and time instruction injection.

    This processor extends the base LLaVA processor to handle:
    1. Video frame processing with proper token expansion
    2. Time instruction injection for temporal understanding
    3. Mixed image and video batches
    """

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
        """
        Process images and/or videos for LLaVA-Video model.

        Args:
            images: List of PIL images
            hf_messages: Messages in HuggingFace format
            videos: List of video tensors, each [T, C, H, W]
            video_metadata: Dict containing video time information:
                - video_time: Total video duration in seconds
                - frame_time: String of frame timestamps like "0.00s,0.50s,1.00s"
                - num_frames: Number of sampled frames
            **kwargs: Additional arguments like fps

        Returns:
            Dict containing input_ids, labels, pixel_values, image_sizes, and modalities
        """
        output_kwargs = self.processor._merge_kwargs(
            LlavaOnevisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        all_pixel_values = []
        all_image_sizes = []
        all_modalities = []
        all_num_tokens = []

        # Process images
        if images is not None and len(images) > 0:
            image_inputs = self.processor.image_processor(
                images, return_tensors="pt", **output_kwargs["images_kwargs"]
            )
            height = image_inputs["pixel_values"].shape[-2]
            width = image_inputs["pixel_values"].shape[-1]

            for i, image_size in enumerate(image_inputs["image_sizes"]):
                num_tokens = self.processor._get_number_of_features(
                    image_size[0].item(), image_size[1].item(), height, width
                )
                all_num_tokens.append(num_tokens)
                all_pixel_values.append(image_inputs["pixel_values"][i])
                all_image_sizes.append(image_size)
                all_modalities.append("image")

        # Process videos
        if videos is not None and len(videos) > 0:
            for video in videos:
                # video shape: [T, C, H, W]
                video_frames = [Image.fromarray(frame.permute(1, 2, 0).numpy().astype("uint8")) for frame in video]

                video_inputs = self.processor.image_processor(
                    video_frames, return_tensors="pt", **output_kwargs["images_kwargs"]
                )

                height = video_inputs["pixel_values"].shape[-2]
                width = video_inputs["pixel_values"].shape[-1]

                # Calculate total tokens for all frames in this video
                total_video_tokens = 0
                for image_size in video_inputs["image_sizes"]:
                    num_tokens = self.processor._get_number_of_features(
                        image_size[0].item(), image_size[1].item(), height, width
                    )
                    total_video_tokens += num_tokens

                all_num_tokens.append(total_video_tokens)
                all_pixel_values.append(video_inputs["pixel_values"])
                all_image_sizes.append(video_inputs["image_sizes"])
                all_modalities.append("video")

        # Get tokenized inputs with labels
        inputs = self.get_video_template_labels(hf_messages, all_num_tokens, all_modalities)

        # Add visual inputs
        if len(all_pixel_values) > 0:
            inputs["pixel_values"] = all_pixel_values
            inputs["image_sizes"] = all_image_sizes
            inputs["modalities"] = all_modalities

        return inputs

    def get_video_template_labels(
        self,
        hf_messages: List[dict],
        num_tokens: List[int],
        modalities: List[str],
    ):
        """
        Tokenize messages and create labels with proper image/video token expansion.

        Args:
            hf_messages: Messages in HuggingFace format
            num_tokens: Number of visual tokens for each image/video
            modalities: List indicating "image" or "video" for each visual input

        Returns:
            Dict with input_ids and labels
        """
        image_token_index = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        special_tokens = self.processor.tokenizer.additional_special_tokens
        unmask_tokens_idx = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens]

        input_id, target = [], []
        visual_idx = 0

        for message in hf_messages:
            role = message["role"]
            encode_id = self.processor.apply_chat_template([message], tokenize=True)[0]

            # Expand image/video tokens
            if image_token_index in encode_id and num_tokens is not None and visual_idx < len(num_tokens):
                encode_id, used_visuals = self._expand_visual_tokens(
                    encode_id, num_tokens, visual_idx
                )
                visual_idx += used_visuals

            input_id += encode_id

            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            else:
                # Mask out the assistant prefix tokens
                encode_id_copy = list(encode_id)
                encode_id_copy[:3] = [-100] * 3
                target += encode_id_copy

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        # Handle special tokens and image tokens in labels
        for idx, token_id in enumerate(input_id):
            if token_id in unmask_tokens_idx:
                target[idx] = token_id
            if token_id == image_token_index:
                target[idx] = image_token_index

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
    ) -> Tuple[List[int], int]:
        """
        Expand image/video placeholder tokens to actual number of visual tokens.

        Args:
            encode_id: Tokenized message
            num_tokens: List of token counts for each visual input
            start_from: Starting index in num_tokens

        Returns:
            Tuple of (expanded_encode_id, number_of_visuals_used)
        """
        image_token_id = self.image_token_id
        image_pos = [i for i, x in enumerate(encode_id) if x == image_token_id]

        expanded_encode_id = []
        prev = 0

        for idx, pos in enumerate(image_pos):
            # Add tokens before the image placeholder
            expanded_encode_id.extend(encode_id[prev:pos])

            # Expand the placeholder to actual number of tokens
            if (idx + start_from) < len(num_tokens):
                expanded_encode_id.extend([image_token_id] * num_tokens[idx + start_from])
            else:
                # Fallback if num_tokens doesn't have enough entries
                expanded_encode_id.append(image_token_id)

            prev = pos + 1

            # Add remaining tokens after last image placeholder
            if idx == len(image_pos) - 1:
                expanded_encode_id.extend(encode_id[prev:])

        return expanded_encode_id, len(image_pos)

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
                            new_content.append({
                                "type": "text",
                                "text": time_instruction
                            })
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
