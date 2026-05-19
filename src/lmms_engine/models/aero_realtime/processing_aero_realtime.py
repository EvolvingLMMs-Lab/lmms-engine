# coding=utf-8
# Copyright 2025 LMMs-Lab team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AeroRealtime processor.

Handles preprocessing of audio, image/video, and text inputs before they
are passed to the AeroRealtime model.  Builds per-chunk audio-vision
envelopes and the realtime ``text_stream_ids`` used by the dual-stream
decoder.
"""

from typing import List, Optional, Union

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.video_utils import VideoInput

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Kwargs defaults
# ---------------------------------------------------------------------------


class AeroRealtimeProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
        },
        "audio_kwargs": {},
        "videos_kwargs": {
            "return_metadata": True,
        },
    }


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class AeroRealtimeProcessor(ProcessorMixin):
    r"""Processor for the AeroRealtime model.

    Combines an image processor (for images), a video processor (for video),
    a feature extractor (for audio), and a tokenizer into a single
    processor.

    The processor handles:
    - Image preprocessing via ``image_processor`` (e.g. ``Qwen2VLImageProcessor``).
    - Video preprocessing via ``video_processor`` (e.g. ``Qwen3VLVideoProcessor``).
      Supports both video file paths (loaded automatically) and pre-extracted
      frames (requires ``video_metadata`` with fps and frame indices).
    - Audio preprocessing via ``feature_extractor`` (e.g. ``WhisperFeatureExtractor``).
    - Text tokenization with placeholder expansion for images, videos, and
      audio tokens.
    - Construction of ``text_stream_ids`` carrying realtime context markers
      on audio positions when audio is present (streaming mode).

    Args:
        image_processor: Image processor instance (e.g. ``Qwen2VLImageProcessor``).
        video_processor: Video processor instance (e.g. ``Qwen3VLVideoProcessor``).
        feature_extractor: Audio feature extractor instance
            (e.g. ``WhisperFeatureExtractor`` with ``feature_size=128``).
        tokenizer: Text tokenizer instance.
        chat_template (`str`, *optional*): Jinja chat template.
        downsample_factor (`int`, *optional*, defaults to ``4``):
            Audio projector downsampling factor.  After the audio encoder,
            ``downsample_factor`` consecutive encoder tokens are concatenated
            before projection.
        image_token (`str`, *optional*, defaults to ``"<|image_pad|>"``):
            Placeholder token for image features.
        video_token (`str`, *optional*, defaults to ``"<|video_pad|>"``):
            Placeholder token for video features.
        audio_token (`str`, *optional*, defaults to ``"<|audio_pad|>"``):
            Placeholder token for audio features.
        vision_start_token (`str`, *optional*, defaults to ``"<|vision_start|>"``):
            Token marking the start of a vision segment.
        vision_end_token (`str`, *optional*, defaults to ``"<|vision_end|>"``):
            Token marking the end of a vision segment.
    """

    attributes = ["image_processor", "video_processor", "feature_extractor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "downsample_factor",
        "audio_length_per_tok",
        "image_token",
        "video_token",
        "audio_token",
        "vision_start_token",
        "vision_end_token",
        "audio_start_token",
        "audio_end_token",
        "rt_start_token",
        "rt_pad_token",
        "rt_speak_token",
        "rt_end_token",
    ]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        downsample_factor: int = 4,
        audio_length_per_tok: int = 8,
        image_token: str = "<|image_pad|>",
        video_token: str = "<|video_pad|>",
        audio_token: str = "<|audio_pad|>",
        vision_start_token: str = "<|vision_start|>",
        vision_end_token: str = "<|vision_end|>",
        audio_start_token: str = "<|audio_start|>",
        audio_end_token: str = "<|audio_end|>",
        rt_start_token: str = "<|rt_start|>",
        rt_pad_token: str = "<|rt_pad|>",
        rt_speak_token: str = "<|rt_speak|>",
        rt_end_token: str = "<|rt_end|>",
        **kwargs,
    ):
        # Resolve tokens from tokenizer if available, otherwise use defaults
        self.image_token = getattr(tokenizer, "image_token", image_token) if tokenizer else image_token
        self.video_token = getattr(tokenizer, "video_token", video_token) if tokenizer else video_token
        self.audio_token = getattr(tokenizer, "audio_token", audio_token) if tokenizer else audio_token
        self.vision_start_token = (
            getattr(tokenizer, "vision_start_token", vision_start_token) if tokenizer else vision_start_token
        )
        self.vision_end_token = (
            getattr(tokenizer, "vision_end_token", vision_end_token) if tokenizer else vision_end_token
        )
        self.audio_start_token = (
            getattr(tokenizer, "audio_start_token", audio_start_token) if tokenizer else audio_start_token
        )
        self.audio_end_token = getattr(tokenizer, "audio_end_token", audio_end_token) if tokenizer else audio_end_token
        self.rt_start_token = rt_start_token
        self.rt_pad_token = rt_pad_token
        self.rt_speak_token = rt_speak_token
        self.rt_end_token = rt_end_token

        # Token IDs needed by _expand_encode_id_video_tokens (inherited from Qwen3VL)
        if tokenizer is not None:
            self.vision_start_token_id = tokenizer.convert_tokens_to_ids(self.vision_start_token)
            self.vision_end_token_id = tokenizer.convert_tokens_to_ids(self.vision_end_token)
            self.audio_start_token_id = tokenizer.convert_tokens_to_ids(self.audio_start_token)
            self.audio_end_token_id = tokenizer.convert_tokens_to_ids(self.audio_end_token)
            self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
            self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)

        # Model config parameters needed for timestep computation
        self.downsample_factor = downsample_factor
        self.audio_length_per_tok = audio_length_per_tok

        if chat_template is None:
            chat_template = self.default_chat_template

        super().__init__(
            image_processor,
            video_processor,
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Main __call__
    # ------------------------------------------------------------------

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio: Union[np.ndarray, List[np.ndarray]] = None,
        video_metadata: Optional[list] = None,
        sampling_rate: Optional[int] = None,
        **kwargs: Unpack[AeroRealtimeProcessorKwargs],
    ) -> BatchFeature:
        """Preprocess multimodal inputs for the AeroRealtime model.

        Accepts any combination of text, images, videos, and audio.  When
        both video and audio are provided the processor emits per-chunk
        ``[VS][video_pad×S][VE][AS][audio_pad×N][AE]`` envelopes so that
        time alignment is expressed through token order and RoPE.

        Args:
            text: One or a batch of text strings (may contain placeholder
                tokens for image / video / audio).
            images: One or more images (PIL, ndarray, or tensor).
            videos: One or more videos.  Each element can be:
                - A file path (``str``) -- the video processor will load
                  and sample frames automatically.
                - A tensor / ndarray of pre-extracted frames -- in this
                  case ``video_metadata`` **must** be provided with at
                  least ``fps`` and ``frames_indices`` for each video.
            audio: One or more raw audio waveforms (mono, float32, 16 kHz).
            video_metadata: List of ``VideoMetadata`` (or dicts with
                ``fps`` and ``frames_indices``).  Required when ``videos``
                are pre-extracted frames; auto-populated when ``videos``
                are file paths.
            sampling_rate: Audio sampling rate.  If ``None``, uses the
                feature extractor's default (typically 16 000 Hz).

        Returns:
            :class:`~transformers.BatchFeature` with the following fields
            (present only when the corresponding input is given):

            - ``input_ids``, ``attention_mask`` -- tokenised text.
            - ``pixel_values``, ``image_grid_thw`` -- image features.
            - ``pixel_values_videos``, ``video_grid_thw`` -- video features.
            - ``input_features`` -- audio mel spectrogram.
            - ``audio_attention_mask`` -- audio attention mask (mel-level).
            - ``text_stream_ids`` -- realtime text-stream tokens (only when
              audio is present).
        """
        output_kwargs = self._merge_kwargs(
            AeroRealtimeProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # ==============================================================
        # 1. Normalise text to list
        # ==============================================================
        if text is not None:
            if isinstance(text, str):
                text = [text]
            text = list(text)  # copy so we can mutate

        # ==============================================================
        # 2. Process images
        # ==============================================================
        image_inputs = {}
        image_grid_thw = None
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs.get("images_kwargs", {}))
            image_grid_thw = image_inputs["image_grid_thw"]

        # ==============================================================
        # 3. Process videos
        # ==============================================================
        video_inputs = {}
        video_grid_thw = None
        _video_metadata = None
        if videos is not None:
            videos_kwargs = output_kwargs.get("videos_kwargs", {})
            # Always request metadata internally (needed for timesteps)
            videos_kwargs["return_metadata"] = True
            if video_metadata is not None:
                videos_kwargs["video_metadata"] = video_metadata

            video_inputs = self.video_processor(videos=videos, **videos_kwargs)
            video_grid_thw = video_inputs["video_grid_thw"]
            _video_metadata = video_inputs.pop("video_metadata")

        # ==============================================================
        # 4. Process audio
        # ==============================================================
        audio_inputs = {}
        if audio is not None:
            fe_kwargs = output_kwargs.get("audio_kwargs", {})
            audio_inputs = self.feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="longest",
                **fe_kwargs,
            )
            # Rename keys to avoid conflicts with text attention_mask
            audio_inputs["audio_attention_mask"] = audio_inputs.pop("attention_mask")

        # ==============================================================
        # 5. Expand placeholder tokens in text
        # ==============================================================
        if text is not None:
            # 5a. Image placeholders
            if image_grid_thw is not None:
                merge_length = self.image_processor.merge_size**2
                idx = 0
                for i in range(len(text)):
                    while self.image_token in text[i]:
                        num_tokens = int(image_grid_thw[idx].prod() // merge_length)
                        text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_tokens, 1)
                        idx += 1
                    text[i] = text[i].replace("<|placeholder|>", self.image_token)

            has_video = video_grid_thw is not None
            has_audio = bool(audio_inputs)

            # Pre-compute audio token counts per sample (mel-frame-derived).
            # ``feature_extractor.attention_mask`` lags ``input_features`` by
            # a small constant (reflection-pad frames at the mel boundary),
            # so we add ``pad_offset`` to each sample's mel length before the
            # ceil-div, matching what the post-conv2 mask emission does at
            # the end of ``__call__``.
            num_audio_tokens_list = None
            if has_audio:
                mel_mask = audio_inputs["audio_attention_mask"]
                T_mel = audio_inputs["input_features"].shape[-1]
                pad_offset = T_mel - mel_mask.shape[-1]
                mel_lengths = mel_mask.sum(-1).to(torch.long) + pad_offset
                num_audio_tokens_list = [self._get_num_audio_tokens(int(m.item())) for m in mel_lengths]

            # 5b. Video + Audio -> separated per-chunk vision/audio envelopes
            #
            #   <t.t seconds><|vision_start|><|video_pad|>×spatial<|vision_end|>
            #   <|audio_start|><|audio_pad|>×N_t<|audio_end|>
            #
            # One envelope per video temporal grid (each grid covers
            # ``second_per_grid = temporal_patch_size / fps`` seconds of
            # video).  Audio tokens are split across grids by a two-pointer
            # merge on the actual audio time axis: audio token ``a`` lives
            # at time ``a / audio_rate`` (seconds), and is placed in the
            # first grid whose end time is greater than ``a``'s time.
            if video_grid_thw is not None and has_audio:
                merge_length = self.video_processor.merge_size**2
                temporal_patch_size = getattr(self.video_processor, "temporal_patch_size", 2)
                v_idx = 0
                a_sample_idx = 0
                for i in range(len(text)):
                    while self.video_token in text[i]:
                        metadata = _video_metadata[v_idx]
                        if metadata.fps is None:
                            metadata.fps = 24.0
                        grid_t = int(video_grid_thw[v_idx][0])
                        spatial = int(video_grid_thw[v_idx][1:].prod() // merge_length)
                        curr_timestamp = self._calculate_timestamps(
                            metadata.frames_indices,
                            metadata.fps,
                            temporal_patch_size,
                        )

                        # Audio for this video sample
                        n_audio = num_audio_tokens_list[a_sample_idx]
                        audio_duration = self._get_audio_duration_seconds(
                            audio_inputs["audio_attention_mask"][a_sample_idx]
                        )
                        # Tokens-per-second for this audio (avoid div-by-zero)
                        audio_rate = (n_audio / audio_duration) if audio_duration > 0 else 0.0

                        # Distribute audio by the actual sampled video chunk timestamps.
                        audio_per_chunk = self._split_audio_across_chunk_times(
                            n_audio=n_audio,
                            chunk_start_times=curr_timestamp[:grid_t],
                            audio_rate=audio_rate,
                        )

                        chunk_strs = []
                        for t in range(grid_t):
                            t_sec = curr_timestamp[t] if t < len(curr_timestamp) else curr_timestamp[-1]
                            chunk_strs.append(
                                f"<{t_sec:.1f} seconds>"
                                f"{self.vision_start_token}"
                                f"{'<|videopad|>' * spatial}"
                                f"{self.vision_end_token}"
                                f"{self.audio_start_token}"
                                f"{'<|audiopad|>' * audio_per_chunk[t]}"
                                f"{self.audio_end_token}"
                            )
                        replacement = "".join(chunk_strs)

                        wrapped = f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
                        if wrapped in text[i]:
                            text[i] = text[i].replace(wrapped, replacement, 1)
                        else:
                            text[i] = text[i].replace(self.video_token, replacement, 1)
                        v_idx += 1
                        a_sample_idx += 1
                    text[i] = text[i].replace("<|videopad|>", self.video_token)
                    text[i] = text[i].replace("<|audiopad|>", self.audio_token)

            # 5c. Video-only -> per-frame VS/VE + timestamp text (legacy Qwen3VL style)
            elif video_grid_thw is not None:
                merge_length = self.video_processor.merge_size**2
                idx = 0
                for i in range(len(text)):
                    while self.video_token in text[i]:
                        metadata = _video_metadata[idx]
                        if metadata.fps is None:
                            logger.warning_once(
                                "Frame timestamps are required to construct prompts, but the `fps` of the input video "
                                "could not be inferred. Probably `video_metadata` was missing from inputs and you "
                                "passed pre-sampled frames. Defaulting to `fps=24`. Please provide `video_metadata` "
                                "for more accurate results."
                            )
                            metadata.fps = 24.0

                        curr_timestamp = self._calculate_timestamps(
                            metadata.frames_indices,
                            metadata.fps,
                            getattr(self.video_processor, "temporal_patch_size", 2),
                        )

                        video_placeholder = ""
                        frame_seqlen = int(video_grid_thw[idx][1:].prod() // merge_length)
                        for frame_idx in range(int(video_grid_thw[idx][0])):
                            curr_time = curr_timestamp[frame_idx]
                            video_placeholder += f"<{curr_time:.1f} seconds>"
                            video_placeholder += (
                                self.vision_start_token + "<|videopad|>" * frame_seqlen + self.vision_end_token
                            )

                        wrapped = f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
                        if wrapped in text[i]:
                            text[i] = text[i].replace(wrapped, video_placeholder, 1)
                        else:
                            text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                        idx += 1
                    text[i] = text[i].replace("<|videopad|>", self.video_token)

            # 5d. Audio-only -> single envelope <|audio_start|><|audio_pad|>×N<|audio_end|>
            if has_audio and not has_video:
                idx = 0
                for i in range(len(text)):
                    while self.audio_token in text[i]:
                        n_tok = num_audio_tokens_list[idx]
                        replacement = f"{self.audio_start_token}" f"{'<|audiopad|>' * n_tok}" f"{self.audio_end_token}"
                        text[i] = text[i].replace(self.audio_token, replacement, 1)
                        idx += 1
                    text[i] = text[i].replace("<|audiopad|>", self.audio_token)

        # ==============================================================
        # 6. Tokenize text
        # ==============================================================
        text_inputs = {}
        if text is not None:
            return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        else:
            return_tensors = None

        # ==============================================================
        # 7. Build text_stream_ids — only in streaming mode (audio present)
        # ==============================================================
        has_video = video_grid_thw is not None
        text_stream_outputs = {}
        if text_inputs and audio_inputs:
            input_ids = text_inputs["input_ids"]
            text_stream_ids = self._build_text_stream_ids(
                input_ids=input_ids,
                video_grid_thw=video_grid_thw,
                video_metadata=_video_metadata,
                audio_attention_mask=audio_inputs.get("audio_attention_mask", None),
                has_video=has_video,
            )
            text_stream_outputs["text_stream_ids"] = text_stream_ids

        # ==============================================================
        # 8. Assemble output
        # ==============================================================
        # Convert audio_attention_mask from mel-level (T_mel) to
        # post-conv2 encoder length (T_enc = T_mel // 2) —
        # this is what VoxtralRealtimeEncoder.forward expects as
        # ``attention_mask``. Internal helpers above operated on the
        # mel-level mask; downstream model code consumes the encoder-level mask.
        if audio_inputs and "audio_attention_mask" in audio_inputs:
            mel_mask = audio_inputs["audio_attention_mask"]
            if not isinstance(mel_mask, torch.Tensor):
                mel_mask = torch.as_tensor(mel_mask)
            # Canonical T_mel comes from the mel feature tensor itself, not
            # the FE attention_mask (which may be sample-grid-aligned and a
            # few frames shorter than input_features due to reflection
            # padding inside the FE).
            input_features = audio_inputs.get("input_features", None)
            if input_features is None:
                T_mel = mel_mask.shape[-1]
            else:
                if not isinstance(input_features, torch.Tensor):
                    input_features = torch.as_tensor(input_features)
                T_mel = input_features.shape[-1]
            mel_mask_len = mel_mask.shape[-1]
            # Per-sample valid frame count in the mel feature. Add the
            # constant pad offset (T_mel - mel_mask_len) to map valid
            # FE-mask frames onto the mel grid.
            pad_offset = max(0, T_mel - mel_mask_len)
            mel_lengths_t = mel_mask.sum(-1).to(torch.long) + pad_offset
            B = mel_mask.shape[0]
            # Voxtral conv2: kernel=3, stride=2, left_pad=1 →
            #   T_enc = (T_mel + left_pad - kernel) / stride + 1 = T_mel // 2
            T_enc = T_mel // 2
            enc_mask = torch.zeros(B, T_enc, dtype=torch.long)
            for i in range(B):
                m = int(mel_lengths_t[i].item())
                # Clamp to T_mel (a sample's valid mel can't exceed full grid)
                m = min(m, T_mel)
                enc_m = m // 2 if m > 0 else 0
                if enc_m > 0:
                    enc_mask[i, :enc_m] = 1
            audio_inputs["audio_attention_mask"] = enc_mask

        return BatchFeature(
            data={
                **text_inputs,
                **image_inputs,
                **video_inputs,
                **audio_inputs,
                **text_stream_outputs,
            },
            tensor_type=return_tensors,
        )

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _get_audio_duration_seconds(self, audio_attention_mask_row: torch.Tensor) -> float:
        """Return the wall-clock duration (seconds) of one audio sample."""
        hop_length = getattr(self.feature_extractor, "hop_length", 160)
        sampling_rate = getattr(self.feature_extractor, "sampling_rate", 16000)
        mel_len = int(audio_attention_mask_row.sum().item())
        return mel_len * hop_length / sampling_rate

    @staticmethod
    def _split_audio_across_chunks(
        n_audio: int,
        grid_t: int,
        second_per_grid: float,
        audio_rate: float,
    ) -> List[int]:
        """Distribute ``n_audio`` audio tokens across ``grid_t`` video chunks
        by their wall-clock time.

        Audio token ``a`` lives at time ``a / audio_rate`` (seconds).  It is
        assigned to the first chunk ``t`` whose end time
        ``(t + 1) * second_per_grid`` is greater than ``a``'s time.  Audio
        tokens past the last chunk's end time are appended to the last chunk.

        Returns a list of length ``grid_t`` whose sum equals ``n_audio``.
        """
        if grid_t <= 0:
            return []
        if n_audio <= 0 or audio_rate <= 0:
            counts = [0] * grid_t
            return counts

        counts = [0] * grid_t
        for a in range(n_audio):
            t_sec = a / audio_rate
            chunk_idx = int(t_sec // second_per_grid)
            if chunk_idx >= grid_t:
                chunk_idx = grid_t - 1
            counts[chunk_idx] += 1
        return counts

    @staticmethod
    def _split_audio_across_chunk_times(
        n_audio: int,
        chunk_start_times: List[float],
        audio_rate: float,
    ) -> List[int]:
        """Distribute audio tokens using actual video chunk start times."""
        grid_t = len(chunk_start_times)
        if grid_t <= 0:
            return []
        if n_audio <= 0 or audio_rate <= 0:
            return [0] * grid_t

        counts = [0] * grid_t
        boundaries = chunk_start_times[1:]
        for a in range(n_audio):
            t_sec = a / audio_rate
            chunk_idx = 0
            while chunk_idx < len(boundaries) and t_sec >= boundaries[chunk_idx]:
                chunk_idx += 1
            counts[chunk_idx] += 1
        return counts

    def _get_num_audio_tokens(self, mel_frames: int) -> int:
        """LM audio token count for a given mel-frame count.

        Voxtral encoder pipeline:
          - conv2 (kernel=3, stride=2, left_pad=1): mel → ``T_enc = mel // 2``
          - downsample_factor concat: ``T_enc // df`` LM tokens
        Combined: ``mel // (2 * df) = mel // audio_length_per_tok``.
        With ``audio_length_per_tok = 8`` this is 80ms per LM token.
        Uses floor (not ceil) to match the modeling-side truncation
        ``usable_len = (T_enc // df) * df``.
        """
        return mel_frames // self.audio_length_per_tok

    @staticmethod
    def _get_audio_encoder_output_length(mel_frames: int) -> int:
        """Compute audio encoder output length from mel spectrogram length.

        Default implementation matches the Qwen2Audio encoder architecture:
        ``conv2`` with stride=2 followed by ``avg_pool`` with stride=2.

        Override this method if using a different audio encoder.

        Args:
            mel_frames: Number of mel spectrogram frames.

        Returns:
            Number of encoder output tokens.
        """
        after_conv2 = (mel_frames - 1) // 2 + 1
        after_avg_pool = (after_conv2 - 2) // 2 + 1
        return after_avg_pool

    @staticmethod
    def _calculate_timestamps(
        frames_indices: Union[list, np.ndarray],
        fps: float,
        temporal_patch_size: int = 2,
    ) -> list:
        """Compute per-temporal-patch timestamps from sampled frame indices.

        Groups ``temporal_patch_size`` consecutive frames and returns the
        average timestamp (in seconds) for each group.  Follows the same
        logic as ``Qwen3VLProcessor._calculate_timestamps``.

        Args:
            frames_indices: Indices of sampled frames in the original video.
            fps: Native FPS of the video.
            temporal_patch_size: Number of frames merged into one temporal
                patch (default ``2``).

        Returns:
            List of timestamps in seconds, one per temporal patch.
        """
        if not isinstance(frames_indices, list):
            frames_indices = list(frames_indices)

        # Pad to be divisible by temporal_patch_size
        if len(frames_indices) % temporal_patch_size != 0:
            pad_count = temporal_patch_size - len(frames_indices) % temporal_patch_size
            frames_indices.extend([frames_indices[-1]] * pad_count)

        # Convert frame indices to seconds
        timestamps_sec = [idx / fps for idx in frames_indices]

        # Average consecutive groups
        grouped = []
        for i in range(0, len(timestamps_sec), temporal_patch_size):
            group = timestamps_sec[i : i + temporal_patch_size]
            grouped.append(sum(group) / len(group))

        return grouped

    # ------------------------------------------------------------------
    # Text stream construction
    # ------------------------------------------------------------------

    def _build_text_stream_ids(
        self,
        input_ids: Union[list, torch.Tensor],
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_metadata: Optional[list] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        has_video: bool = False,
    ) -> Union[list, torch.Tensor]:
        """Build ``text_stream_ids`` for the realtime dual-stream design.

        ``text_stream_ids`` mirrors ``input_ids`` everywhere except audio
        placeholder positions, where it carries realtime context tokens.

        Streaming mode is gated on the presence of audio.  Two layouts:

        - **video + audio (interleave)**: input contains per-chunk envelopes
          ``[VS][video_pad×S][VE][AS][audio_pad×N][AE]``.  Video placeholders
          stay as ``<|video_pad|>``; audio placeholders default to
          ``<|rt_pad|>`` and may carry teacher-forced speech tokens.
        - **audio-only**: ``[AS][audio_pad×N][AE]``. First ``audio_pad``
          becomes ``<|rt_speak|>`` (the model decides when to start
          speaking); the rest are ``<|rt_pad|>``.
        """
        is_tensor = isinstance(input_ids, torch.Tensor)
        if is_tensor:
            input_ids_list = input_ids.tolist()
        else:
            input_ids_list = [list(ids) for ids in input_ids]

        rt_pad_id = self.tokenizer.convert_tokens_to_ids(self.rt_pad_token)
        audio_start_id = self.tokenizer.convert_tokens_to_ids(self.audio_start_token)
        audio_end_id = self.tokenizer.convert_tokens_to_ids(self.audio_end_token)

        has_audio = audio_attention_mask is not None and audio_attention_mask.numel() > 0
        temporal_patch_size = getattr(self.video_processor, "temporal_patch_size", 2)

        result = []
        for batch_idx, ids in enumerate(input_ids_list):
            stream = list(ids)

            if has_video and has_audio:
                # --- video + audio interleave ---
                self._fill_text_stream_video_audio(
                    stream=stream,
                    video_grid_thw=video_grid_thw,
                    video_metadata=video_metadata,
                    temporal_patch_size=temporal_patch_size,
                    audio_start_id=audio_start_id,
                    audio_end_id=audio_end_id,
                    rt_pad_id=rt_pad_id,
                )
            elif has_audio:
                # --- audio-only envelope ---
                self._fill_text_stream_audio_only(
                    stream=stream,
                    sample_idx=batch_idx,
                    audio_start_id=audio_start_id,
                    audio_end_id=audio_end_id,
                    rt_pad_id=rt_pad_id,
                )

            result.append(stream)

        if is_tensor:
            return torch.tensor(result, dtype=input_ids.dtype, device=input_ids.device)
        return result

    # ------------------------------------------------------------------
    # text_stream fillers (one per mode)
    # ------------------------------------------------------------------

    def _fill_text_stream_video_audio(
        self,
        stream: list,
        video_grid_thw,
        video_metadata,
        temporal_patch_size: int,
        audio_start_id: int,
        audio_end_id: int,
        rt_pad_id: int,
    ) -> None:
        """In-place fill of text_stream for separated video+audio envelopes.

        Only ``<|audio_pad|>`` positions are overwritten with ``<|rt_pad|>``.
        Boundary and speech teacher forcing is applied by the data processor.

        ``<|video_pad|>`` positions keep their original ids because video
        features replace those embeddings in the model.

        Envelope boundary tokens (``<t.t seconds>``, ``<|vision_start|>``,
        ``<|vision_end|>``, ``<|audio_start|>``, ``<|audio_end|>``) keep
        their original ids so the LM sees the same special tokens it would
        in input_ids.
        """
        ids_t = torch.tensor(stream)
        as_idx = (ids_t == audio_start_id).nonzero(as_tuple=True)[0].tolist()
        ae_idx = (ids_t == audio_end_id).nonzero(as_tuple=True)[0].tolist()

        audio_envelopes = list(zip(as_idx, ae_idx))
        if not audio_envelopes:
            return

        # Flatten per-chunk metadata: (v_idx, t_in_video, t_sec, spatial)
        merge_length = self.video_processor.merge_size**2
        chunks = []
        for v_idx in range(len(video_grid_thw)):
            metadata = video_metadata[v_idx]
            fps = metadata.fps if metadata.fps is not None else 24.0
            grid_t = int(video_grid_thw[v_idx][0])
            spatial = int(video_grid_thw[v_idx][1:].prod() // merge_length)
            curr_timestamp = self._calculate_timestamps(
                metadata.frames_indices,
                fps,
                temporal_patch_size,
            )
            for t in range(grid_t):
                t_sec = curr_timestamp[t] if t < len(curr_timestamp) else curr_timestamp[-1]
                chunks.append((v_idx, t, t_sec, spatial))

        if len(chunks) != len(audio_envelopes):
            raise ValueError(
                f"Chunk count mismatch: {len(chunks)} expected audio envelopes from "
                f"video_grid_thw, found {len(audio_envelopes)} in tokenized stream."
            )

        for as_, ae in audio_envelopes:
            # Envelope layout (token positions):
            #   as_:          <|audio_start|>
            #   as_+1 .. ae-1:<|audio_pad|> × N_t
            #   ae:           <|audio_end|>
            audio_pad_start = as_ + 1
            audio_pad_end = ae - 1  # inclusive

            for k in range(audio_pad_start, audio_pad_end + 1):
                stream[k] = rt_pad_id

    def _fill_text_stream_audio_only(
        self,
        stream: list,
        sample_idx: int,
        audio_start_id: int,
        audio_end_id: int,
        rt_pad_id: int,
    ) -> None:
        """Audio-only envelope text_stream filler.

        Layout produced by the processor: ``[AS][audio_pad×N][AE]``.

        Every audio_pad position defaults to ``rt_pad`` (silence). The dataset
        processor overrides positions occupied by speech tokens at training
        time; at inference the model is free to emit text content from any
        audio_pad position.
        """
        ids_t = torch.tensor(stream)
        as_positions = (ids_t == audio_start_id).nonzero(as_tuple=True)[0].tolist()
        ae_positions = (ids_t == audio_end_id).nonzero(as_tuple=True)[0].tolist()
        if sample_idx >= len(as_positions) or sample_idx >= len(ae_positions):
            return
        as_pos = as_positions[sample_idx]
        ae_pos = ae_positions[sample_idx]
        if ae_pos - as_pos - 1 <= 0:
            return
        for k in range(as_pos + 1, ae_pos):
            stream[k] = rt_pad_id

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def batch_decode(self, *args, **kwargs):
        """Decode a batch of token ids to strings."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token ids to a string."""
        return self.tokenizer.decode(*args, **kwargs)

    # ------------------------------------------------------------------
    # Default chat template
    # ------------------------------------------------------------------

    @property
    def default_chat_template(self):
        # fmt: off
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% set image_count = namespace(value=0) %}"
            "{% set video_count = namespace(value=0) %}"
            "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'audio' in content or 'audio_url' in content %}"
                            "{% set audio_count.value = audio_count.value + 1 %}"
                            "<|audio_pad|>"
                        "{% elif 'image' in content or 'image_url' in content %}"
                            "{% set image_count.value = image_count.value + 1 %}"
                            "<|vision_start|><|image_pad|><|vision_end|>"
                        "{% elif 'video' in content or 'video_url' in content %}"
                            "{% set video_count.value = video_count.value + 1 %}"
                            "<|vision_start|><|video_pad|><|vision_end|>"
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
