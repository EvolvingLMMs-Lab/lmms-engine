"""Data processor for LLaVA-OneVision2 (8B-Instruct, trust_remote_code).

OV2 reuses the Qwen2VL image processor (patches in 2x2 block order) and a
custom video processor that emits per-frame patches + ``patch_positions``.
Token-side, every video is rewritten as a sequence of per-frame blocks of
the form ``<X.X seconds><|vision_start|><|image_pad|>*n<|vision_end|>``,
so the model only ever sees the *image* path (videos are aliased into
``pixel_values`` / ``image_grid_thw`` / ``patch_positions``).

This processor inherits Qwen3-VL's image-side logic and overrides the video
expansion to produce the OV2 block format.
"""

from typing import List, Optional

import numpy as np
import torch
from PIL.Image import Image
from transformers import AutoProcessor

from lmms_engine.mapping_func import register_processor
from lmms_engine.utils import DataUtilities

from .qwen3_vl_processor import Qwen3_VLDataProcessor


@register_processor("llava_onevision2")
class LlavaOnevision2DataProcessor(Qwen3_VLDataProcessor):
    def _build_processor(self):
        # OV2 ships its processor via auto_map / trust_remote_code.
        processor = AutoProcessor.from_pretrained(self.config.processor_name, trust_remote_code=True)

        # Optional pixel-budget overrides (consistent with Qwen3VL).
        image_max_pixels = self.config.extra_kwargs.get("image_max_pixels", None)
        image_min_pixels = self.config.extra_kwargs.get("image_min_pixels", None)
        if image_max_pixels is not None or image_min_pixels is not None:
            self._set_vision_processor_size(processor.image_processor, image_min_pixels, image_max_pixels)

        video_max_pixels = self.config.extra_kwargs.get("video_max_pixels", None)
        video_min_pixels = self.config.extra_kwargs.get("video_min_pixels", None)
        if video_processor := getattr(processor, "video_processor", None):
            if video_max_pixels is not None:
                video_processor.max_pixels = int(video_max_pixels)
            if video_min_pixels is not None:
                video_processor.min_pixels = int(video_min_pixels)
        return processor

    # ------------------------------------------------------------------ process

    def process(
        self,
        images: List[Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        assert audios is None, "LlavaOnevision2DataProcessor does not support audio"

        # ---------------- Image branch ----------------
        if images is not None:
            image_inputs = self.processor.image_processor(images=images, return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        # ---------------- Video branch ----------------
        # OV2 video_processor expects a list of videos as pre-decoded frame
        # arrays in **HWC uint8** format (so it can call PIL.Image.fromarray).
        # The iterable dataset (qwen3_vl_iterable / qwen_vl_utils) hands us
        # **CHW float32 / uint8** numpy arrays of shape ``[T, 3, H, W]``, so
        # convert here.
        if videos is not None:
            videos = [self._normalize_video_for_ov2(v) for v in videos]
            videos_inputs = self.processor.video_processor(videos=videos, return_tensors="pt")
            video_grid_thw = videos_inputs["video_grid_thw"]  # [num_videos, 3]
            frame_timestamps = videos_inputs["frame_timestamps"]  # list[list[float]]
            video_pixel_values = videos_inputs["pixel_values_videos"]
            video_patch_positions = videos_inputs["patch_positions"]
        else:
            video_grid_thw = None
            frame_timestamps = None
            video_pixel_values = None
            video_patch_positions = None

        # Token-count math (Qwen2VL-style, per merge-block).
        merge_length = self.processor.image_processor.merge_size**2

        if image_grid_thw is not None:
            num_image_tokens = [int(g.prod()) // merge_length for g in image_grid_thw]
        else:
            num_image_tokens = None

        # For videos: each frame becomes one "image". Token count per frame is
        # (H_p * W_p) // merge_length (T is always 1 after we split rows).
        if video_grid_thw is not None:
            num_video_tokens_per_frame = [int(g[1] * g[2]) // merge_length for g in video_grid_thw]
        else:
            num_video_tokens_per_frame = None

        # Build text/labels using OV2-specific video expansion.
        inputs = self._get_ov2_template_labels(
            hf_messages=hf_messages,
            num_image_tokens=num_image_tokens,
            num_video_tokens_per_frame=num_video_tokens_per_frame,
            frame_timestamps=frame_timestamps,
            video_grid_thw=video_grid_thw,
            system_message=system_message,
            add_system_prompt=add_system_prompt,
            add_generation_prompt=add_generation_prompt,
        )

        # ---------------- Build patch_positions for IMAGE inputs ----------------
        # Pull build_patch_positions from the OV2 remote module via the
        # processor (already imported in trust_remote_code load).
        build_patch_positions = self._get_build_patch_positions()
        sms = int(self.processor.image_processor.merge_size)

        # ---------------- Alias videos -> image path ---------------------------
        # Each video row [T, H, W] -> T rows of [1, H, W]; concat with images.
        pixel_values_parts = []
        image_grid_thw_parts = []
        patch_positions_parts = []

        if image_grid_thw is not None:
            pixel_values_parts.append(image_inputs["pixel_values"])
            image_grid_thw_parts.append(image_grid_thw)
            patch_positions_parts.append(build_patch_positions(image_grid_thw, spatial_merge_size=sms))

        if video_grid_thw is not None:
            pixel_values_parts.append(video_pixel_values)
            expanded_rows = []
            for row in video_grid_thw:
                T_v, H_v, W_v = int(row[0]), int(row[1]), int(row[2])
                expanded_rows.extend([[1, H_v, W_v]] * T_v)
            image_grid_thw_parts.append(torch.tensor(expanded_rows, dtype=video_grid_thw.dtype))
            # Video processor already produced block-layout patch_positions
            # using REAL frame indices for t — preserve that and just concat.
            patch_positions_parts.append(video_patch_positions)

        if pixel_values_parts:
            inputs["pixel_values"] = torch.cat(pixel_values_parts, dim=0)
            inputs["image_grid_thw"] = torch.cat(image_grid_thw_parts, dim=0)
            inputs["patch_positions"] = torch.cat(patch_positions_parts, dim=0)

        return inputs

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _normalize_video_for_ov2(video):
        """Coerce decoder output into a list[np.ndarray HWC uint8].

        OV2's ``LlavaOnevision2VideoProcessor._coerce_video_input`` only
        accepts list[PIL.Image], list[np.ndarray HWC uint8], or a path. The
        Qwen VL utils backend returns a torch tensor / numpy array shaped
        ``[T, 3, H, W]`` in CHW order with float or uint8 dtype, so we need
        to permute + cast it.
        """
        import torch

        if isinstance(video, str):
            return video

        if isinstance(video, torch.Tensor):
            arr = video.detach().cpu().numpy()
        elif isinstance(video, np.ndarray):
            arr = video
        elif isinstance(video, list):
            # Already a list[PIL.Image] / list[np.ndarray frame]: pass through.
            return video
        else:
            return video

        # CHW -> HWC if the leading inner dim looks like channels.
        if arr.ndim == 4 and arr.shape[1] in (1, 3, 4):
            arr = np.transpose(arr, (0, 2, 3, 1))

        # Cast to uint8 for PIL.
        if arr.dtype != np.uint8:
            arr_max = float(arr.max()) if arr.size else 0.0
            arr_min = float(arr.min()) if arr.size else 0.0
            if arr_max <= 1.5 and arr_min >= -0.01:
                # Looks like a [0,1] float tensor.
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

        # Hand back a list of per-frame HWC arrays so OV2's coercion takes the
        # ``list[np.ndarray]`` branch.
        return [arr[i] for i in range(arr.shape[0])]

    def _get_build_patch_positions(self):
        """Resolve ``build_patch_positions`` from the dynamically-loaded
        video_processing module shipped with the OV2 checkpoint."""
        if not hasattr(self, "_cached_build_patch_positions"):
            video_proc = self.processor.video_processor
            import sys

            mod = sys.modules[type(video_proc).__module__]
            self._cached_build_patch_positions = mod.build_patch_positions
        return self._cached_build_patch_positions

    @property
    def vision_start_token_id(self) -> int:
        return self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")

    @property
    def vision_end_token_id(self) -> int:
        return self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    # OV2's processor does not expose ``image_token`` / ``video_token`` string
    # attributes (unlike Qwen3VLProcessor), so the parent class properties
    # return None. Resolve straight from the special tokens used by the OV2
    # chat template.
    @property
    def image_token_id(self) -> int:
        if not hasattr(self, "_image_token_id"):
            self._image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        return self._image_token_id

    @property
    def video_token_id(self) -> int:
        if not hasattr(self, "_video_token_id"):
            self._video_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        return self._video_token_id

    # ----------------------------------------------------- label construction

    def _get_ov2_template_labels(
        self,
        hf_messages,
        num_image_tokens: Optional[List[int]],
        num_video_tokens_per_frame: Optional[List[int]],
        frame_timestamps: Optional[List[List[float]]],
        video_grid_thw=None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
        add_generation_prompt: bool = False,
    ):
        unmask_tokens_idx = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in self.special_tokens]
        input_id, target = [], []
        image_start_from = 0
        video_start_from = 0

        if add_system_prompt and hf_messages[0]["role"] != "system":
            input_id += DataUtilities.apply_chat_template(
                self.processor,
                [{"role": "system", "content": [{"type": "text", "text": system_message}]}],
            )
            target += [-100] * len(input_id)

        for message in hf_messages:
            role = message["role"]
            encode_id = DataUtilities.apply_chat_template(self.processor, [message])

            if self.image_token_id in encode_id and num_image_tokens is not None:
                encode_id, used_images = self._expand_encode_id_image_tokens(
                    encode_id, num_image_tokens, image_start_from
                )
                image_start_from += used_images

            if (
                self.video_token_id in encode_id
                and num_video_tokens_per_frame is not None
                and frame_timestamps is not None
            ):
                encode_id, used_video = self._expand_encode_id_video_tokens_ov2(
                    encode_id,
                    num_video_tokens_per_frame,
                    video_start_from,
                    frame_timestamps,
                    video_grid_thw,
                )
                video_start_from += used_video

            input_id += encode_id
            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            else:
                encode_id[:3] = [-100] * 3  # mask out the assistant header
                target += encode_id

        if add_generation_prompt:
            generation_tokens = self.processor.tokenizer.encode("<|im_start|>assistant\n")
            input_id += generation_tokens
            target += [-100] * len(generation_tokens)

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, tok in enumerate(input_id):
            if tok in unmask_tokens_idx:
                target[idx] = tok
            if tok == self.image_token_id:
                target[idx] = -100
            if tok == self.video_token_id:
                target[idx] = -100

        return dict(
            input_ids=torch.tensor(input_id, dtype=torch.long),
            labels=torch.tensor(target, dtype=torch.long),
        )

    def _expand_encode_id_video_tokens_ov2(
        self,
        encode_id: List[int],
        num_video_tokens_per_frame: List[int],
        start_from: int,
        frame_timestamps: List[List[float]],
        video_grid_thw,
    ):
        """Rewrite each ``<|vision_start|><|video_pad|><|vision_end|>`` triplet
        into per-frame ``<X.X seconds><|vision_start|><|image_pad|>*n<|vision_end|>``
        blocks (OV2 native format).
        """
        video_pos = [i for i, x in enumerate(encode_id) if x == self.video_token_id]
        expanded = []
        prev = 0
        tokenizer = self.processor.tokenizer
        vs_id = self.vision_start_token_id
        ve_id = self.vision_end_token_id

        for idx, pos in enumerate(video_pos):
            vidx = idx + start_from
            T_v = int(video_grid_thw[vidx, 0])
            n_per_frame = num_video_tokens_per_frame[vidx]
            seconds_seq = frame_timestamps[vidx]
            # Defensive pad/truncate to T_v.
            if len(seconds_seq) < T_v:
                pad_val = seconds_seq[-1] if seconds_seq else 0.0
                seconds_seq = list(seconds_seq) + [pad_val] * (T_v - len(seconds_seq))
            elif len(seconds_seq) > T_v:
                seconds_seq = list(seconds_seq[:T_v])

            # Strip the original surrounding <|vision_start|>/<|vision_end|>:
            # chat_template produces <|vision_start|><|video_pad|><|vision_end|>
            # at positions (pos-1, pos, pos+1). The per-frame blocks below
            # re-emit their own.
            expanded.extend(encode_id[prev : pos - 1])

            for t in range(T_v):
                ts_token = f"<{float(seconds_seq[t]):.1f} seconds>"
                ts_ids = tokenizer.encode(ts_token, add_special_tokens=False)
                expanded.extend(ts_ids)
                expanded.append(vs_id)
                expanded.extend([self.image_token_id] * n_per_frame)
                expanded.append(ve_id)

            prev = pos + 2  # skip <|vision_end|>

            if idx == len(video_pos) - 1:
                expanded.extend(encode_id[prev:])

        return expanded, len(video_pos)
