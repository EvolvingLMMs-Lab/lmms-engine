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

"""AeroRealtime data processor.

Handles training data processing for the AeroRealtime model.  Supports both
normal video QA (assistant replies after video) and realtime training
(assistant text segments are placed at specific temporal positions during
video playback).

The processor builds ``text_stream_ids`` on the audio timeline. ``<|rt_pad|>``
is silence context only; labels supervise ``<|rt_speak|>``, speech span
boundaries (``<|rt_start|>`` / ``<|rt_end|>``), and speech text tokens.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL.Image import Image

from lmms_engine.datasets.processor.qwen3_vl_processor import Qwen3_VLDataProcessor
from lmms_engine.mapping_func import register_processor
from lmms_engine.models.aero_realtime.processing_aero_realtime import (
    AeroRealtimeProcessor,
    AeroRealtimeProcessorKwargs,
)
from lmms_engine.utils import DataUtilities

from .config import ProcessorConfig


@register_processor("aero_realtime")
class AeroRealtimeDataProcessor(Qwen3_VLDataProcessor):
    """Data processor for AeroRealtime training.

    Builds ``input_ids``, ``text_stream_ids``, and ``labels`` for the
    realtime audio-stream training design.  Handles:
    - Normal video QA: audio timeline filled with ``<|rt_pad|>`` context
    - Realtime training: boundary and text labels on audio tokens
    - Image-only: standard scatter (no text_stream_ids)
    - Audio extraction from video for audio-vision fusion
    """

    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config
        # Fraction of <|rt_pad|> labels to mask out (set to -100) so the
        # model is not asked to predict silence at every audio_pad slot.
        # 0.0 = supervise all, 1.0 = never supervise rt_pad.
        self.rt_pad_label_mask_ratio = float((config.extra_kwargs or {}).get("rt_pad_label_mask_ratio", 0.0))
        # Same idea on the codec side: at each silence-marked frame,
        # drop ALL 16 codec groups' labels to -100 with this probability.
        # 0.0 = supervise every silence frame, 1.0 = never supervise silence frames.
        self.codec_silence_label_mask_ratio = float(
            (config.extra_kwargs or {}).get("codec_silence_label_mask_ratio", 0.0)
        )
        # Shift the codec target forward by this many audio_pad frames so the
        # talker learns to "speak N frames ahead of the transcript". The first
        # N audio_pad slots get filled with a silence codec (both input and
        # label). 0 = no shift (original 1:1 alignment).
        self.codec_shift_frames = int((config.extra_kwargs or {}).get("codec_shift_frames", 0))

    def build(self):
        self.processor = self._build_processor()
        # Override chat template to handle realtime_text content type
        self.processor.chat_template = self.chat_template

    def _build_processor(self):
        processor = AeroRealtimeProcessor.from_pretrained(self.config.processor_name)
        # Set video processor parameters from extra_kwargs
        video_max_pixels = self.config.extra_kwargs.get("video_max_pixels", None)
        video_min_pixels = self.config.extra_kwargs.get("video_min_pixels", None)
        if video_max_pixels and hasattr(processor, "video_processor"):
            processor.video_processor.max_pixels = video_max_pixels
        if video_min_pixels and hasattr(processor, "video_processor"):
            processor.video_processor.min_pixels = video_min_pixels

        image_max_pixels = self.config.extra_kwargs.get("image_max_pixels", None)
        image_min_pixels = self.config.extra_kwargs.get("image_min_pixels", None)
        if image_max_pixels and hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = image_max_pixels
        if image_min_pixels and hasattr(processor, "image_processor"):
            processor.image_processor.min_pixels = image_min_pixels

        return processor

    def save_pretrained(self, save_directory: str):
        if not hasattr(self, "processor"):
            raise ValueError("Processor has not been built yet. Call build() first.")
        new_processor = self._build_processor()
        new_processor.save_pretrained(save_directory)

    # ------------------------------------------------------------------
    # Token ID helpers
    # ------------------------------------------------------------------

    @property
    def special_tokens(self):
        if not hasattr(self, "_special_tokens"):
            self._special_tokens = DataUtilities.get_special_tokens(
                self.processor.tokenizer,
                extra_tokens=["<|im_start|>", "<|im_end|>"],
            )
        return self._special_tokens

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def sampling_rate(self):
        return self.processor.feature_extractor.sampling_rate

    @property
    def image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.processor.image_token)

    @property
    def video_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.processor.video_token)

    @property
    def audio_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.processor.audio_token)

    @property
    def rt_start_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.processor.rt_start_token)

    @property
    def rt_pad_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.processor.rt_pad_token)

    @property
    def rt_speak_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.processor.rt_speak_token)

    @property
    def rt_end_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.processor.rt_end_token)

    # ------------------------------------------------------------------
    # Main process entry point
    # ------------------------------------------------------------------

    def process(
        self,
        images: Optional[List[Image]] = None,
        hf_messages=None,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        realtime_segments: Optional[List[Dict]] = None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt=True,
        assistant_codec=None,
        **kwargs,
    ):
        """Process a single training example.

        Args:
            images: List of PIL images.
            hf_messages: Messages in HF format (from convert_open_to_hf).
                The ``realtime_text`` content items should already have been
                stripped and passed via ``realtime_segments``.
            audios: List of audio waveforms (mono, float32, at sampling_rate).
            sampling_rate: Audio sampling rate.
            videos: List of video frames (numpy arrays, TCHW format).
            realtime_segments: List of ``{"start_sec": float, "text": str}``
                dicts extracted from assistant ``realtime_text`` content items.
                If None, this is treated as normal video QA.
            system_message: System prompt text.
            add_system_prompt: Whether to add a system prompt.
            **kwargs: Forwarded to the model processor (e.g. ``fps``,
                ``do_sample_frames``, ``video_metadata``).

        Returns:
            Dict with ``input_ids``, ``text_stream_ids``, ``labels``, and
            vision/audio tensors.
        """
        output_kwargs = self.processor._merge_kwargs(
            AeroRealtimeProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # ==============================================================
        # 0. Normal-QA tail planning: if hf_messages has assistant turns
        #    AND we have video, compute per-pair token counts so we can
        #    append matching silence audio features after the main FE pass
        #    (we don't push silence through the feature extractor because
        #    ``padding="longest"`` would pad short tails up to the video
        #    audio length, producing zero-length encoder chunks).
        # ==============================================================
        qa_silence_token_counts: List[int] = []
        if realtime_segments is None and videos is not None and hf_messages is not None:
            qa_pairs, _ = self._extract_qa_pairs(hf_messages)
            for _, a_text in qa_pairs:
                a_tok = self.tokenizer.encode(a_text, add_special_tokens=False)
                qa_silence_token_counts.append(len(a_tok))

        # ==============================================================
        # 1. Process images
        # ==============================================================
        image_inputs = {}
        image_grid_thw = None
        if images is not None:
            image_inputs = self.processor.image_processor(
                images=images, return_tensors="pt", **output_kwargs.get("images_kwargs", {})
            )
            image_grid_thw = image_inputs["image_grid_thw"]

        # ==============================================================
        # 2. Process videos
        # ==============================================================
        video_inputs = {}
        video_grid_thw = None
        _video_metadata = None
        if videos is not None:
            videos_kwargs = output_kwargs.get("videos_kwargs", {})
            video_inputs = self.processor.video_processor(videos=videos, return_tensors="pt", **videos_kwargs)
            video_grid_thw = video_inputs["video_grid_thw"]
            _video_metadata = video_inputs.pop("video_metadata")

        # ==============================================================
        # 3. Process audio
        # ==============================================================
        audio_inputs = {}
        if audios is not None:
            fe_kwargs = output_kwargs.get("audio_kwargs", {})
            audio_inputs = self.processor.feature_extractor(
                audios,
                sampling_rate=sampling_rate or self.sampling_rate,
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
                **fe_kwargs,
            )
            audio_inputs["audio_attention_mask"] = audio_inputs.pop("attention_mask")

        # ==============================================================
        # 4. Compute token counts for placeholder expansion
        # ==============================================================
        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            num_image_tokens = [int(grid_thw.prod() // merge_length) for grid_thw in image_grid_thw]
        else:
            num_image_tokens = None

        if video_grid_thw is not None:
            merge_length = self.processor.video_processor.merge_size**2
            num_video_tokens = [int(grid_thw.prod() // merge_length) for grid_thw in video_grid_thw]
        else:
            num_video_tokens = None

        has_video = video_grid_thw is not None
        has_audio = bool(audio_inputs)

        # Per-video audio token splits across video temporal chunks.
        # Required for envelope construction when both video and audio are
        # present (the inner ``<|audio_pad|>`` count of each per-chunk envelope).
        audio_per_chunk_per_video = None
        if has_video and has_audio:
            # FE attention_mask lags ``input_features`` by a small constant
            # (reflection-pad frames at the mel boundary); add ``pad_offset``
            # to recover the per-sample mel length before computing tokens.
            mel_mask = audio_inputs["audio_attention_mask"]
            T_mel = audio_inputs["input_features"].shape[-1]
            pad_offset = max(0, T_mel - mel_mask.shape[-1])
            mel_lengths = mel_mask.sum(-1).to(torch.long) + pad_offset
            num_audio_tokens_list = [self.processor._get_num_audio_tokens(int(m.item())) for m in mel_lengths]
            temporal_patch_size = getattr(self.processor.video_processor, "temporal_patch_size", 2)
            audio_per_chunk_per_video = []
            for v_idx in range(len(video_grid_thw)):
                metadata = _video_metadata[v_idx]
                fps = metadata.fps if metadata.fps is not None else 24.0
                grid_t = int(video_grid_thw[v_idx][0])
                curr_timestamp = self.processor._calculate_timestamps(
                    metadata.frames_indices,
                    fps,
                    temporal_patch_size,
                )
                # Audio sample paired with this video by positional index
                a_idx = v_idx if v_idx < len(num_audio_tokens_list) else 0
                n_audio = num_audio_tokens_list[a_idx]
                audio_duration = self.processor._get_audio_duration_seconds(audio_inputs["audio_attention_mask"][a_idx])
                audio_rate = (n_audio / audio_duration) if audio_duration > 0 else 0.0
                audio_per_chunk_per_video.append(
                    self.processor._split_audio_across_chunk_times(
                        n_audio=n_audio,
                        chunk_start_times=curr_timestamp[:grid_t],
                        audio_rate=audio_rate,
                    )
                )

        # ==============================================================
        # 5. Build input_ids, text_stream_ids, labels
        # ==============================================================
        if realtime_segments is None:
            inputs = self._build_normal_qa_ids_and_labels(
                hf_messages=hf_messages,
                num_image_tokens=num_image_tokens,
                num_video_tokens=num_video_tokens,
                video_grid_thw=video_grid_thw,
                video_metadata=_video_metadata,
                audio_per_chunk_per_video=audio_per_chunk_per_video,
                audio_attention_mask=audio_inputs.get("audio_attention_mask") if has_audio else None,
                system_message=system_message,
                add_system_prompt=add_system_prompt,
            )
        else:
            inputs = self._build_realtime_ids_and_labels(
                hf_messages=hf_messages,
                num_image_tokens=num_image_tokens,
                num_video_tokens=num_video_tokens,
                video_grid_thw=video_grid_thw,
                video_metadata=_video_metadata,
                audio_per_chunk_per_video=audio_per_chunk_per_video,
                audio_attention_mask=audio_inputs.get("audio_attention_mask") if has_audio else None,
                realtime_segments=realtime_segments,
                system_message=system_message,
                add_system_prompt=add_system_prompt,
            )

        # ==============================================================
        # 5b. Randomly mask a fraction of rt_pad supervision so the model
        #     isn't forced to predict silence at every audio_pad slot.
        # ==============================================================
        if self.rt_pad_label_mask_ratio > 0.0 and "labels" in inputs:
            labels = inputs["labels"]
            rt_pad_mask = labels == self.rt_pad_id
            if rt_pad_mask.any():
                drop = torch.rand_like(labels, dtype=torch.float) < self.rt_pad_label_mask_ratio
                labels = labels.masked_fill(rt_pad_mask & drop, -100)
                inputs["labels"] = labels

        # ==============================================================
        # 6. Pack vision/audio tensors into output
        # ==============================================================
        if images is not None:
            inputs["pixel_values"] = image_inputs["pixel_values"]
            inputs["image_grid_thw"] = image_inputs["image_grid_thw"]

        if videos is not None:
            for key, value in video_inputs.items():
                inputs[key] = value

        if audios is not None:
            # Emit audio features + post-conv2 encoder mask. Mirrors
            # ``AeroRealtimeProcessor.__call__`` section 8:
            #   - chunk_audio (default): reshape into [B*N, F, chunk_mel] rows,
            #     one per LM audio token; mask [B*N, chunk_enc].
            #   - flat: keep [B, F, T_mel]; downsample mel_mask by 2 → [B, T_enc].
            feats = torch.as_tensor(audio_inputs["input_features"])
            mel_mask = torch.as_tensor(audio_inputs["audio_attention_mask"])
            mel_mask = torch.nn.functional.pad(mel_mask, (0, feats.shape[-1] - mel_mask.shape[-1]), value=1)

            if self.processor.chunk_audio:
                chunk_mel = self.processor.audio_length_per_tok
                chunk_enc = chunk_mel // 2
                pad = (-feats.shape[-1]) % chunk_mel
                feats = torch.nn.functional.pad(feats, (0, pad))
                mel_mask = torch.nn.functional.pad(mel_mask, (0, pad))
                B, F, T_mel = feats.shape
                N = T_mel // chunk_mel
                feats = feats.view(B, F, N, chunk_mel).permute(0, 2, 1, 3).reshape(B * N, F, chunk_mel)
                chunk_valid = mel_mask.view(B, N, chunk_mel).all(-1)  # [B, N]
                enc_mask = chunk_valid.unsqueeze(-1).expand(B, N, chunk_enc).reshape(B * N, chunk_enc).long()
            else:
                T_enc = mel_mask.shape[-1] // 2
                enc_mask = mel_mask[:, : T_enc * 2].view(mel_mask.shape[0], T_enc, 2).all(-1).long()

            inputs["input_features"] = feats
            inputs["audio_attention_mask"] = enc_mask

            # ----------------------------------------------------------
            # Append silence audio chunks for normal-QA tail envelopes.
            # Each tail token = one valid silence chunk (features all
            # zero, enc_mask all one). Done *after* FE so we don't
            # trigger ``padding="longest"`` against the video audio.
            # ----------------------------------------------------------
            total_tail = sum(qa_silence_token_counts)
            if total_tail > 0 and self.processor.chunk_audio:
                F_dim = feats.shape[-2]
                tail_feats = feats.new_zeros((total_tail, F_dim, chunk_mel))
                tail_mask = enc_mask.new_ones((total_tail, chunk_enc))
                inputs["input_features"] = torch.cat([feats, tail_feats], dim=0)
                inputs["audio_attention_mask"] = torch.cat([enc_mask, tail_mask], dim=0)

        if assistant_codec is not None:
            if len(assistant_codec) == 3:
                flat, n_frames, is_silence = assistant_codec
            else:
                flat, n_frames = assistant_codec
                is_silence = None
            G = 16
            codec_arr = np.asarray(flat, dtype=np.int64).reshape(n_frames, G)
            input_ids = inputs["input_ids"]
            seq_len = int(input_ids.shape[0])
            codec_labels = np.full((seq_len, G), -100, dtype=np.int64)
            codec_input_ids = np.full((seq_len, G), -100, dtype=np.int64)
            audio_positions = [i for i, t in enumerate(input_ids.tolist()) if t == self.audio_token_id]

            shift = self.codec_shift_frames
            # Silence codec: first is_silence==1 frame from this sample's codec,
            # fallback to codec_arr[0] if no silence column or no silence frames.
            if shift > 0:
                silence_frame = codec_arr[0]
                if is_silence is not None and n_frames > 0:
                    silence_idx_arr = np.where(np.asarray(is_silence, dtype=np.uint8) == 1)[0]
                    if silence_idx_arr.size > 0:
                        silence_frame = codec_arr[int(silence_idx_arr[0])]

            n_supervised = min(len(audio_positions), n_frames + shift)
            for k in range(n_supervised):
                src = k - shift
                if src < 0:
                    codec_labels[audio_positions[k]] = silence_frame
                    codec_input_ids[audio_positions[k]] = silence_frame
                else:
                    codec_labels[audio_positions[k]] = codec_arr[src]
                    codec_input_ids[audio_positions[k]] = codec_arr[src]

            # Tail audio_pad positions with no codec frame: fill inputs with the
            # last real codec frame (so embeddings are in-range); labels stay -100
            # (no supervision).
            if n_supervised < len(audio_positions) and n_frames > 0:
                for k in range(n_supervised, len(audio_positions)):
                    codec_input_ids[audio_positions[k]] = codec_arr[n_frames - 1]

            if self.codec_silence_label_mask_ratio > 0.0 and is_silence is not None:
                silence_arr = np.asarray(is_silence, dtype=np.uint8)
                drop = np.random.rand(n_supervised) < self.codec_silence_label_mask_ratio
                for k in range(n_supervised):
                    src = k - shift
                    if src < 0 or src >= n_frames:
                        continue
                    if silence_arr[src] == 1 and drop[k]:
                        codec_labels[audio_positions[k]] = -100

            inputs["codec_labels"] = torch.from_numpy(codec_labels)
            inputs["codec_input_ids"] = torch.from_numpy(codec_input_ids)

        return inputs

    # ------------------------------------------------------------------
    # Core: build input_ids, text_stream_ids, labels
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_qa_pairs(hf_messages):
        """Split hf_messages into video-only messages + [(q_text, a_text), ...].

        Question text that lives in the same user turn as the video is
        attached to the *next* assistant turn (so all q+a tails go after
        the video envelope, inside the same user turn).
        """
        qa_pairs = []
        video_only = []
        pending_q_parts = []
        for msg in hf_messages:
            role = msg["role"]
            if role == "system":
                video_only.append(msg)
                continue
            if role == "user":
                video_content, text_parts = [], []
                for c in msg["content"]:
                    ctype = c.get("type")
                    if ctype in ("video", "image", "audio"):
                        video_content.append(c)
                    elif ctype == "text" and c.get("text"):
                        text_parts.append(c["text"])
                if video_content:
                    video_only.append({"role": "user", "content": video_content})
                pending_q_parts.extend(text_parts)
            elif role == "assistant":
                a_parts = [c["text"] for c in msg["content"] if c.get("type") == "text" and c.get("text")]
                if not a_parts:
                    continue
                qa_pairs.append(("\n".join(pending_q_parts).strip(), "\n".join(a_parts)))
                pending_q_parts = []
        return qa_pairs, video_only

    def _build_normal_qa_ids_and_labels(
        self,
        hf_messages,
        num_image_tokens: Optional[List[int]],
        num_video_tokens: Optional[List[int]],
        video_grid_thw=None,
        video_metadata=None,
        audio_per_chunk_per_video: Optional[List[List[int]]] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        realtime_segments: Optional[List[Dict]] = None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
    ) -> dict:
        """Build input_ids/text_stream_ids/labels for normal video QA.

        When the conversation contains assistant turns *and* a video, the
        assistant answers are NOT supervised as plain text after the video
        envelope.  Instead each ``(question, answer)`` pair is appended
        inside the same user turn as::

            q_k <|audio_start|><|audio_pad|>×n_k <|audio_end|>

        where ``n_k = len(tokenize(a_k))``.  The ``input_ids`` carry silent
        ``<|audio_pad|>`` slots, ``text_stream_ids`` carry the answer tokens
        in those same slots, and ``labels`` supervise only the answer tokens.
        The caller (dataset) is responsible for appending matching silence
        audio (``n_k * 1280`` samples per pair) so the audio feature
        extractor produces exactly ``n_k`` audio tokens per tail envelope.

        When there are no assistant turns OR no video, falls back to the
        original behaviour: standard Qwen template labels + ``rt_pad``
        rewrite on audio_pad positions.
        """

        # --------------------------------------------------------------
        # 1. Split hf_messages -> video_only_messages + qa_pairs
        #    (process() may have already done this; idempotent)
        # --------------------------------------------------------------
        qa_pairs, video_only_messages = self._extract_qa_pairs(hf_messages)
        has_video = video_grid_thw is not None

        use_qa_tail = bool(qa_pairs) and has_video
        base_messages = video_only_messages if use_qa_tail else hf_messages

        # --------------------------------------------------------------
        # 2+3. Delegate to realtime path with empty segments to get
        #      input_ids / text_stream_ids / labels for the video portion.
        #      Video-period audio_pad labels become rt_pad (silence).
        # --------------------------------------------------------------
        rt_result = self._build_realtime_ids_and_labels(
            hf_messages=base_messages,
            num_image_tokens=num_image_tokens,
            num_video_tokens=num_video_tokens,
            video_grid_thw=video_grid_thw,
            video_metadata=video_metadata,
            audio_per_chunk_per_video=audio_per_chunk_per_video,
            audio_attention_mask=audio_attention_mask,
            realtime_segments=[],
            system_message=system_message,
            add_system_prompt=add_system_prompt,
        )
        input_id = rt_result["input_ids"].tolist()
        text_stream_id = rt_result["text_stream_ids"].tolist()
        target = rt_result["labels"].tolist()

        # --------------------------------------------------------------
        # 4. Append tail (q + silent audio envelope) for each QA pair
        # --------------------------------------------------------------
        if use_qa_tail:
            audio_start_id = self.tokenizer.convert_tokens_to_ids(self.processor.audio_start_token)
            audio_end_id = self.tokenizer.convert_tokens_to_ids(self.processor.audio_end_token)
            audio_pad_id = self.audio_token_id

            # Strip trailing "<|im_end|>\n" so the tail sits inside the same user turn.
            im_end_tokens = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
            tail_len = len(im_end_tokens)
            assert (
                input_id[-tail_len:] == im_end_tokens
            ), f"expected trailing im_end tokens {im_end_tokens}, got {input_id[-tail_len:]}"
            input_id = input_id[:-tail_len]
            text_stream_id = text_stream_id[:-tail_len]
            target = target[:-tail_len]

            for q_text, a_text in qa_pairs:
                q_tok = self.tokenizer.encode(q_text, add_special_tokens=False) if q_text else []
                a_tok = self.tokenizer.encode(a_text, add_special_tokens=False)
                n_k = len(a_tok)
                if n_k == 0:
                    continue
                input_id += q_tok + [audio_start_id] + [audio_pad_id] * n_k + [audio_end_id]
                text_stream_id += q_tok + [audio_start_id] + a_tok + [audio_end_id]
                target += [-100] * len(q_tok) + [-100] + a_tok + [-100]

            # Re-append "<|im_end|>\n"
            input_id += im_end_tokens
            text_stream_id += im_end_tokens
            target += [-100] * tail_len

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        result = dict(
            input_ids=input_id,
            labels=target,
        )
        result["text_stream_ids"] = torch.tensor(text_stream_id, dtype=torch.long)

        return result

    def _build_realtime_ids_and_labels(
        self,
        hf_messages,
        num_image_tokens: Optional[List[int]],
        num_video_tokens: Optional[List[int]],
        video_grid_thw=None,
        video_metadata=None,
        audio_per_chunk_per_video: Optional[List[List[int]]] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        realtime_segments: Optional[List[Dict]] = None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
    ) -> dict:
        if video_grid_thw is None or audio_per_chunk_per_video is None or audio_attention_mask is None:
            raise ValueError("Realtime training requires both video and audio inputs.")

        base_messages, timed_user_segments = self._build_realtime_base_messages(
            hf_messages=hf_messages,
            realtime_segments=realtime_segments or [],
            video_grid_thw=video_grid_thw,
            video_metadata=video_metadata,
            audio_per_chunk_per_video=audio_per_chunk_per_video,
            system_message=system_message,
            add_system_prompt=add_system_prompt,
        )

        results = self.get_qwen_template_labels(
            base_messages,
            num_image_tokens,
            num_video_tokens,
            video_metadata,
            video_grid_thw,
            audio_per_chunk_per_video=audio_per_chunk_per_video,
            timed_user_segments=timed_user_segments,
            system_message=system_message,
            add_system_prompt=False,
        )
        input_id = results["input_ids"].tolist()
        text_stream_id = list(input_id)
        target = [-100] * len(input_id)

        self.processor._fill_text_stream_video_audio(
            stream=text_stream_id,
            video_grid_thw=video_grid_thw,
            video_metadata=video_metadata,
            temporal_patch_size=getattr(self.processor.video_processor, "temporal_patch_size", 2),
            audio_start_id=self.tokenizer.convert_tokens_to_ids(self.processor.audio_start_token),
            audio_end_id=self.tokenizer.convert_tokens_to_ids(self.processor.audio_end_token),
            rt_pad_id=self.rt_pad_id,
        )

        audio_positions = [idx for idx, token_id in enumerate(input_id) if token_id == self.audio_token_id]
        audio_times = self._get_audio_position_times(
            video_grid_thw=video_grid_thw,
            video_metadata=video_metadata,
            audio_per_chunk_per_video=audio_per_chunk_per_video,
        )
        if len(audio_positions) != len(audio_times):
            raise ValueError(f"Audio position/time mismatch: {len(audio_positions)} != {len(audio_times)}")

        # Default supervision on every audio_pad slot: predict rt_pad (silence).
        # Segments below overwrite this with text tokens where speech is happening.
        for pos in audio_positions:
            target[pos] = self.rt_pad_id

        assistant_segments = sorted(
            [seg for seg in (realtime_segments or []) if seg.get("role") == "assistant" and seg.get("text")],
            key=lambda item: float(item["time"]),
        )
        occupied_audio_indices: set = set()
        for segment in assistant_segments:
            start_time = float(segment["time"])
            start_audio_idx = self._first_index_at_or_after(audio_times, start_time)
            available_indices = self._next_available_indices(
                start=start_audio_idx,
                count=len(audio_positions),
                limit=len(audio_positions),
                occupied=occupied_audio_indices,
            )
            if len(available_indices) < 2:
                continue
            # First slot holds rt_pad (unsupervised) so the next token has a
            # context to attend to; remaining slots carry the speech tokens.
            text_token_budget = len(available_indices) - 1
            text_tokens = self._encode_realtime_text(segment["text"])[:text_token_budget]
            # Write rt_pad in the first slot of this segment, but DO NOT
            # supervise it (label stays at the default rt_pad set above? we
            # want -100 so the model is never asked to "decide to start"
            # against a boundary-aligned target).
            first_audio_idx = available_indices[0]
            first_pos = audio_positions[first_audio_idx]
            text_stream_id[first_pos] = self.rt_pad_id
            target[first_pos] = -100
            occupied_audio_indices.add(first_audio_idx)
            # Place speech tokens in the following slots; both stream and label
            # carry the actual token id.
            for audio_idx, token_id in zip(available_indices[1:], text_tokens):
                pos = audio_positions[audio_idx]
                text_stream_id[pos] = token_id
                target[pos] = token_id
                occupied_audio_indices.add(audio_idx)

        input_tensor = torch.tensor(input_id, dtype=torch.long)
        text_stream_tensor = torch.tensor(text_stream_id, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)

        return dict(
            input_ids=input_tensor,
            labels=target_tensor,
            text_stream_ids=text_stream_tensor,
        )

    def _build_realtime_base_messages(
        self,
        hf_messages,
        realtime_segments: List[Dict],
        video_grid_thw,
        video_metadata,
        audio_per_chunk_per_video: List[List[int]],
        system_message: str,
        add_system_prompt: bool,
    ):
        messages = []
        first_content = []
        timed_user_segments = sorted(
            [seg for seg in realtime_segments if seg.get("role") == "user" and seg.get("text")],
            key=lambda item: float(item["time"]),
        )

        if add_system_prompt and (not hf_messages or hf_messages[0]["role"] != "system"):
            messages.append({"role": "system", "content": [{"type": "text", "text": system_message}]})

        for message in hf_messages:
            if message["role"] == "system":
                messages.append(message)
                continue
            if message.get("time") is not None:
                continue
            for content in message["content"]:
                if content.get("type") in ["image", "video", "audio"]:
                    first_content.append(content)

        content = []
        content.extend(first_content)

        messages.append({"role": "user", "content": content})
        return messages, timed_user_segments

    def _get_chunk_start_times(self, video_grid_thw, video_metadata, audio_per_chunk_per_video: List[List[int]]):
        times = []
        for v_idx in range(len(video_grid_thw)):
            metadata = video_metadata[v_idx]
            fps = metadata.fps if metadata.fps is not None else 24.0
            curr_timestamp = self.processor._calculate_timestamps(
                metadata.frames_indices,
                fps,
                self.processor.video_processor.temporal_patch_size,
            )
            for t in range(len(audio_per_chunk_per_video[v_idx])):
                times.append(curr_timestamp[t] if t < len(curr_timestamp) else curr_timestamp[-1])
        return times

    def _get_audio_position_times(self, video_grid_thw, video_metadata, audio_per_chunk_per_video: List[List[int]]):
        times = []
        chunk_times = self._get_chunk_start_times(video_grid_thw, video_metadata, audio_per_chunk_per_video)
        chunk_idx = 0
        for audio_per_chunk in audio_per_chunk_per_video:
            for n_audio in audio_per_chunk:
                times.extend([chunk_times[chunk_idx]] * n_audio)
                chunk_idx += 1
        return times

    def _encode_realtime_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _first_index_at_or_after(values: List[float], target: float) -> int:
        for idx, value in enumerate(values):
            if value >= target:
                return idx
        return len(values)

    @staticmethod
    def _next_available_indices(start: int, count: int, limit: int, occupied: set) -> List[int]:
        indices = []
        idx = start
        while idx < limit and len(indices) < count:
            if idx not in occupied:
                indices.append(idx)
            idx += 1
        return indices

    def get_qwen_template_labels(
        self,
        hf_messages,
        num_image_tokens: List[int],
        num_video_tokens: List[int],
        video_metadata: List[dict],
        video_grid_thw=None,
        audio_per_chunk_per_video: Optional[List[List[int]]] = None,
        timed_user_segments: Optional[List[Dict]] = None,
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
                self.processor, [{"role": "system", "content": [{"type": "text", "text": system_message}]}]
            )
            target += [-100] * len(input_id)
        for message in hf_messages:
            role = message["role"]
            encode_id = DataUtilities.apply_chat_template(self.processor, [message])
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
                    self.processor.video_processor.temporal_patch_size,
                )
                encode_id, used_video = self._expand_encode_id_video_tokens(
                    encode_id,
                    num_video_tokens,
                    video_start_from,
                    curr_timestamp,
                    video_grid_thw,
                    audio_per_chunk_per_video=audio_per_chunk_per_video,
                    timed_user_segments=timed_user_segments,
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
            if encode_id == self.audio_token_id:
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
        curr_timestamp: List[float] = None,
        video_grid_thw=None,
        audio_per_chunk_per_video: Optional[List[List[int]]] = None,
        timed_user_segments: Optional[List[Dict]] = None,
    ):
        """Expand ``<|video_pad|>`` placeholders.

        - Without audio: per-frame Qwen3VL legacy expansion (delegated to
          parent).
        - With audio: per-chunk separated vision/audio envelopes::

            <t.t seconds><|vision_start|><|video_pad|>×spatial<|vision_end|>
            <|audio_start|><|audio_pad|>×N_t<|audio_end|>
        """
        if audio_per_chunk_per_video is None:
            return super()._expand_encode_id_video_tokens(
                encode_id, video_token_num, start_from, curr_timestamp, video_grid_thw
            )

        merge_length = self.processor.video_processor.merge_size**2
        vision_start_id = self.processor.vision_start_token_id
        vision_end_id = self.processor.vision_end_token_id
        audio_start_id = self.tokenizer.convert_tokens_to_ids(self.processor.audio_start_token)
        audio_end_id = self.tokenizer.convert_tokens_to_ids(self.processor.audio_end_token)
        temporal_patch_size = getattr(self.processor.video_processor, "temporal_patch_size", 2)
        timed_user_segments = timed_user_segments or []

        video_pos = [i for i, x in enumerate(encode_id) if x == self.video_token_id]
        expanded_encode_id = []
        prev = 0
        for idx, pos in enumerate(video_pos):
            v_global = idx + start_from
            grid = video_grid_thw[v_global]
            grid_t = int(grid[0])
            spatial = int(grid[1:].prod() // merge_length)

            # Figure out per-chunk audio counts; fps from grid (we only have
            # curr_timestamp which is per-frame timestamps in seconds).  Use
            # them directly for the chunk start times.
            audio_per_chunk = audio_per_chunk_per_video[v_global]
            assert len(audio_per_chunk) == grid_t, f"audio_per_chunk len {len(audio_per_chunk)} != grid_t {grid_t}"
            chunk_times = [
                curr_timestamp[t] if t < len(curr_timestamp) else (t * temporal_patch_size) for t in range(grid_t)
            ]
            user_by_chunk = [[] for _ in range(grid_t)]
            for segment in timed_user_segments:
                chunk_idx = self._first_index_at_or_after(chunk_times, float(segment["time"]))
                if chunk_idx >= grid_t:
                    chunk_idx = grid_t - 1
                user_by_chunk[chunk_idx].append(segment["text"])

            # Strip surrounding <|vision_start|> / <|vision_end|> from the
            # template (positions pos-1 and pos+1) -- we will emit our own.
            expanded_encode_id.extend(encode_id[prev : pos - 1])

            for t in range(grid_t):
                for user_text in user_by_chunk[t]:
                    expanded_encode_id.extend(self._encode_realtime_text(user_text))

                # Per-frame timestamp (seconds) from the video metadata
                t_sec = chunk_times[t]
                timestamp_token_ids = self.processor.tokenizer.encode(f"<{t_sec:.1f} seconds>")
                n_audio_t = audio_per_chunk[t]
                expanded_encode_id.extend(timestamp_token_ids)
                expanded_encode_id.append(vision_start_id)
                expanded_encode_id.extend([self.video_token_id] * spatial)
                expanded_encode_id.append(vision_end_id)
                expanded_encode_id.append(audio_start_id)
                expanded_encode_id.extend([self.audio_token_id] * n_audio_t)
                expanded_encode_id.append(audio_end_id)

            prev = pos + 2  # skip past original <|vision_end|>

            if idx == len(video_pos) - 1:
                expanded_encode_id.extend(encode_id[prev:])

        return expanded_encode_id, len(video_pos)

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    @property
    def chat_template(self):
        """Chat template that handles realtime_text content by ignoring it.

        Realtime text segments are extracted separately in the dataset
        and passed as ``realtime_segments`` to the processor.  The chat
        template only renders ``text``, ``image``, ``video``, and ``audio``
        content types.
        """
        # fmt: off
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
                            "<|audio_pad|>"
                        "{% elif content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
                            "{% set image_count.value = image_count.value + 1 %}"
                            "<|vision_start|><|image_pad|><|vision_end|>"
                        "{% elif content['type'] == 'video' or 'video' in content %}"
                            "{% set video_count.value = video_count.value + 1 %}"
                            "<|vision_start|><|video_pad|><|vision_end|>"
                        "{% elif 'text' in content %}"
                            "{{ content['text'] }}"
                        "{% elif content.get('type') == 'realtime_text' %}"
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
