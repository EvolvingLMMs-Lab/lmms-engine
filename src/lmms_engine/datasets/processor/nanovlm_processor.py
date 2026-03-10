from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from lmms_engine.mapping_func import register_processor
from lmms_engine.utils import DataUtilities

from .config import ProcessorConfig


@register_processor("nanovlm")
class NanovlmDataProcessor:
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config

    def build(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.processor_name)

        # Load image processor from the same local/remote checkpoint as tokenizer.
        # `NanoVLM_init` now carries both tokenizer and preprocessor configs.
        loaded_processor = AutoProcessor.from_pretrained(self.config.processor_name)
        self.image_processor = getattr(loaded_processor, "image_processor", loaded_processor)

        self.image_token = self.config.extra_kwargs.get("image_token", "<|image_pad|>")
        self.video_token = self.config.extra_kwargs.get("video_token", "<|video_pad|>")

        for name, token in (("image_token", self.image_token), ("video_token", self.video_token)):
            if token not in self._tokenizer.get_vocab():
                raise ValueError(f"{name} {token} not found in tokenizer vocab. Please use a Qwen3 token.")

        self.processor = SimpleNamespace(
            tokenizer=self._tokenizer,
            image_token=self.image_token,
            video_token=self.video_token,
            batch_decode=self._tokenizer.batch_decode,
        )

    @property
    def special_tokens(self):
        if not hasattr(self, "_special_tokens"):
            self._special_tokens = DataUtilities.get_special_tokens(
                self._tokenizer, extra_tokens=["<|im_start|>", "<|im_end|>"]
            )
        return self._special_tokens

    def save_pretrained(self, output_dir: str):
        self._tokenizer.save_pretrained(output_dir)
        self.image_processor.save_pretrained(output_dir)

    def process(
        self,
        images: Optional[List[Image.Image]],
        hf_messages,
        videos=None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt=True,
        add_generation_prompt=False,
        **kwargs,
    ):
        flat_images, num_image_tokens, num_video_tokens = self._prepare_visual_inputs(
            images=images,
            videos=videos,
            hf_messages=hf_messages,
        )

        if flat_images is not None and len(flat_images) > 0:
            image_inputs = self.image_processor(images=flat_images, return_tensors="pt")
        else:
            image_inputs = {}
            num_image_tokens = None
            num_video_tokens = None

        inputs = self.get_qwen_template_labels(
            hf_messages,
            num_image_tokens,
            num_video_tokens,
            system_message=system_message,
            add_system_prompt=add_system_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes"):
            if key in image_inputs:
                inputs[key] = image_inputs[key]
        return inputs

    def get_qwen_template_labels(
        self,
        hf_messages,
        num_image_tokens: Optional[List[int]],
        num_video_tokens: Optional[List[int]],
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
        add_generation_prompt: bool = False,
    ):
        unmask_tokens_idx = [self._tokenizer.convert_tokens_to_ids(t) for t in self.special_tokens]
        input_id, target = [], []
        image_start_from = 0
        video_start_from = 0
        if add_system_prompt and hf_messages[0]["role"] != "system":
            input_id += self._apply_chat_template([{"role": "system", "content": system_message}], tokenize=True)
            target += [-100] * len(input_id)

        for message in hf_messages:
            role = message["role"]
            encode_id = self._apply_chat_template([message], tokenize=True)
            if self.image_token_id in encode_id and num_image_tokens is not None:
                encode_id, used_images = self._expand_encode_id_image_tokens(
                    encode_id, num_image_tokens, image_start_from
                )
                image_start_from += used_images
            if self.video_token_id in encode_id and num_video_tokens is not None:
                encode_id, used_videos = self._expand_encode_id_video_tokens(
                    encode_id, num_video_tokens, video_start_from
                )
                video_start_from += used_videos
                # Nanovlm only understands image_token_id, map video tokens to image tokens
                encode_id = [self.image_token_id if t == self.video_token_id else t for t in encode_id]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            else:
                encode_id_copy = list(encode_id)
                encode_id_copy[:3] = [-100] * 3
                target += encode_id_copy

        if add_generation_prompt:
            generation_tokens = self._tokenizer.encode("<|im_start|>assistant\n")
            input_id += generation_tokens
            target += [-100] * len(generation_tokens)

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == self.image_token_id:
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
            expanded_encode_id.extend(encode_id[prev:pos])
            expanded_encode_id.extend([self.image_token_id] * image_token_num[idx + start_from])
            prev = pos + 1

            if idx == len(image_pos) - 1:
                expanded_encode_id.extend(encode_id[prev:])

        return expanded_encode_id, len(image_pos)

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
            expanded_encode_id.extend(encode_id[prev:pos])
            expanded_encode_id.extend([self.video_token_id] * video_token_num[idx + start_from])
            prev = pos + 1

            if idx == len(video_pos) - 1:
                expanded_encode_id.extend(encode_id[prev:])

        return expanded_encode_id, len(video_pos)

    def _apply_chat_template(self, messages, tokenize: bool = False):
        result = self._tokenizer.apply_chat_template(messages, tokenize=tokenize)
        if isinstance(result, list) and result and isinstance(result[0], list):
            return result[0]
        return result

    def _prepare_visual_inputs(
        self,
        images: Optional[List[Image.Image]],
        videos: Optional[Sequence],
        hf_messages,
    ) -> Tuple[Optional[List], Optional[List[int]], Optional[List[int]]]:
        if images is None and videos is None:
            return None, None, None

        image_token_count = self.config.extra_kwargs.get(
            "image_token_count",
            getattr(self.image_processor, "max_num_patches", 256),
        )
        flat_images: List = []
        num_image_tokens: List[int] = []
        num_video_tokens: List[int] = []

        image_idx = 0
        video_idx = 0

        for message in hf_messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                item_type = item.get("type")
                if item_type == "image":
                    if images is None or image_idx >= len(images):
                        raise ValueError("Missing image input for <image> placeholder.")
                    flat_images.append(self._to_pil_image(images[image_idx]))
                    num_image_tokens.append(image_token_count)
                    image_idx += 1
                elif item_type == "video":
                    if videos is None or video_idx >= len(videos):
                        raise ValueError("Missing video input for <video> placeholder.")
                    frames = self._normalize_video_frames(videos[video_idx])
                    flat_images.extend([self._to_pil_image(frame) for frame in frames])
                    num_video_tokens.append(image_token_count * len(frames))
                    video_idx += 1

        if len(flat_images) == 0:
            return None, None, None

        return flat_images, (num_image_tokens or None), (num_video_tokens or None)

    def _normalize_video_frames(self, video) -> List:
        if isinstance(video, list):
            return video
        if isinstance(video, np.ndarray):
            if video.ndim == 3:
                return [video]
            if video.ndim == 4:
                return [frame for frame in video]
        if torch.is_tensor(video):
            video_np = video.detach().cpu().numpy()
            if video_np.ndim == 3:
                return [video_np]
            if video_np.ndim == 4:
                return [frame for frame in video_np]
        raise ValueError(f"Unsupported video format: {type(video)}")

    def _to_pil_image(self, image: object) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = arr[:, :, None]
            if arr.ndim != 3:
                raise ValueError(f"Unsupported image shape: {arr.shape}")
            # If channel-first, transpose to HWC
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                max_val = float(arr.max()) if arr.size > 0 else 1.0
                if max_val <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            return pil.convert("RGB")
        raise ValueError(f"Unsupported image type: {type(image)}")

    @property
    def image_token_id(self):
        return self._tokenizer.convert_tokens_to_ids(self.image_token)

    @property
    def video_token_id(self):
        return self._tokenizer.convert_tokens_to_ids(self.video_token)

    @property
    def tokenizer(self):
        return self._tokenizer
