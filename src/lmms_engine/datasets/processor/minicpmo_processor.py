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

"""MiniCPM-o data processor.

Builds ``input_ids`` / ``labels`` / modality bounds for MiniCPM-o 4.5
training, on top of the vendored ``MiniCPMOProcessor``.

Two modes:

- **Simplex** (``process``): turn-based conversations.  Media content items
  (``image`` / ``audio``) are rendered as ``<image>./</image>`` and
  ``<audio>./</audio>`` markers inside the message text and expanded to the
  real placeholders (image slicing / audio chunking) during tokenization.
  Only assistant spans (plus their ``<|im_end|>``) are supervised.

- **Duplex / realtime** (``process_realtime``): a continuous audio (optionally
  video) stream is split into fixed 1-second units.  Each unit is rendered as
  ``<unit><image>../</image><|audio_start|><unk>*N<|audio_end|>CONTENT</unit>``
  where CONTENT is ``<|listen|>`` for silent units and
  ``<|speak|> text… <|chunk_tts_eos|>`` for units covered by an assistant
  turn.  Assistant characters are assigned to units by their ``start_sec``
  timestamps (turn-level ``start_sec``/``end_sec`` are linearly
  interpolated).  Supervision covers the listen/speak decision, the text and
  the chunk terminator — the model learns *when* and *what* to speak.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL.Image import Image

from lmms_engine.datasets.processor.config import ProcessorConfig
from lmms_engine.mapping_func import register_processor
from lmms_engine.models.minicpmo.processing_minicpmo import (
    MiniCPMAAudioProcessor,
    MiniCPMOProcessor,
    MiniCPMVImageProcessor,
)
from lmms_engine.models.minicpmo.tokenization_minicpmo_fast import MiniCPMOTokenizerFast
from lmms_engine.utils import DataUtilities

SIMPLEX_SYSTEM_MESSAGE = "You are a helpful assistant."
DUPLEX_SYSTEM_MESSAGE = "Streaming Omni Conversation."

IMAGE_MARKER = "<image>./</image>"
AUDIO_MARKER = "<audio>./</audio>"


@register_processor("minicpmo")
class MiniCPMODataProcessor:
    """Data processor for MiniCPM-o 4.5 simplex and duplex SFT."""

    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config
        # Duplex unit length in seconds (audio_token_rate is 10 tokens/s).
        self.unit_duration = float((config.extra_kwargs or {}).get("unit_duration", 1.0))
        # Maximum slice nums for high-resolution images (None keeps the
        # image processor default of 9 for plain images).
        self.max_slice_nums = (config.extra_kwargs or {}).get("max_slice_nums", None)
        # For realtime video: frames sampled per unit.
        self.frames_per_unit = int((config.extra_kwargs or {}).get("frames_per_unit", 1))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        self.processor = self._build_processor()

    def _build_processor(self):
        """Construct the vendored MiniCPMOProcessor without trust_remote_code."""
        name = self.config.processor_name
        image_processor = MiniCPMVImageProcessor.from_pretrained(name)
        audio_processor = MiniCPMAAudioProcessor.from_pretrained(name)
        tokenizer = MiniCPMOTokenizerFast.from_pretrained(name)
        return MiniCPMOProcessor(
            image_processor=image_processor,
            audio_processor=audio_processor,
            tokenizer=tokenizer,
        )

    def save_pretrained(self, save_directory: str):
        if not hasattr(self, "processor"):
            raise ValueError("Processor has not been built yet. Call build() first.")
        self.processor.save_pretrained(save_directory)

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def sampling_rate(self):
        return self.processor.audio_processor.sampling_rate

    @property
    def special_tokens(self):
        if not hasattr(self, "_special_tokens"):
            self._special_tokens = DataUtilities.get_special_tokens(
                self.tokenizer,
                extra_tokens=["<|im_start|>", "<|im_end|>"],
            )
        return self._special_tokens

    @property
    def pad_token_id(self):
        # MiniCPM-o ships an empty-string pad token; fall back to eos.
        pad_id = self.tokenizer.pad_token_id
        return pad_id if pad_id is not None else self.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Simplex processing
    # ------------------------------------------------------------------

    def process(
        self,
        images: Optional[List[Image]] = None,
        hf_messages: Optional[List[Dict]] = None,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        system_message: str = SIMPLEX_SYSTEM_MESSAGE,
        add_system_prompt: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Build the training inputs for one turn-based conversation.

        Args:
            images: list of PIL images, ordered by appearance in the text.
            hf_messages: HF-format messages (``convert_open_to_hf`` output).
            audios: list of 16 kHz mono waveforms, ordered by appearance.
        """
        # ---- audio features + placeholders (chunked to 30 s / 1 s units) ----
        # audio_feature_extract returns a flat [n_chunks, 80, max_T] tensor
        # for the single sample plus per-chunk lengths and placeholder strings.
        audio_features: Optional[torch.Tensor] = None
        audio_feature_lens = torch.zeros(0, dtype=torch.long)
        audio_phs: List[str] = []
        if audios:
            feats, lens_list, phs_list = self.processor.audio_feature_extract(
                [audios], None, False, sampling_rate or self.sampling_rate
            )
            if isinstance(feats, torch.Tensor) and feats.numel() > 0:
                audio_features = feats
            if lens_list:
                audio_feature_lens = torch.as_tensor(lens_list[0], dtype=torch.long)
            if phs_list:
                audio_phs = list(phs_list[0])

        # ---- image features ----
        pixel_values, image_sizes, tgt_sizes = [], [], []
        if images:
            image_inputs = self.processor.process_image(
                images, do_pad=True, max_slice_nums=self.max_slice_nums or 9
            )
            pixel_values = image_inputs["pixel_values"]
            image_sizes = image_inputs["image_sizes"]
            tgt_sizes = image_inputs["tgt_sizes"]

        # ---- text segments ----
        segments = self._render_simplex_segments(
            hf_messages,
            system_message=system_message,
            add_system_prompt=add_system_prompt,
        )

        # ---- tokenize segments, expanding markers ----
        image_id = 0
        audio_id = 0
        input_ids: List[int] = []
        labels: List[int] = []
        image_bounds: List[torch.Tensor] = []
        audio_bounds: List[torch.Tensor] = []
        spk_bounds: List[torch.Tensor] = []

        for role, segment in segments:
            segment = segment.replace(IMAGE_MARKER, "\x00IMG\x00").replace(AUDIO_MARKER, "\x00AUD\x00")
            pieces = segment.split("\x00")
            seg_ids: List[int] = []
            for piece in pieces:
                if piece == "IMG":
                    placeholder = self.processor.image_processor.get_slice_image_placeholder(
                        image_sizes[0][image_id], image_id, self.max_slice_nums, True
                    )
                    image_id += 1
                    seg_ids.extend(self.tokenizer.encode(placeholder, add_special_tokens=False))
                elif piece == "AUD":
                    placeholder = audio_phs[audio_id]
                    audio_id += 1
                    seg_ids.extend(self.tokenizer.encode(placeholder, add_special_tokens=False))
                else:
                    seg_ids.extend(self.tokenizer.encode(piece, add_special_tokens=False))

            seg_tensor = torch.tensor(seg_ids, dtype=torch.long)
            _, seg_image_bounds, seg_audio_bounds, seg_spk_bounds = self._convert(seg_tensor)
            image_bounds.append(seg_image_bounds + len(input_ids))
            audio_bounds.append(seg_audio_bounds + len(input_ids))
            spk_bounds.append(seg_spk_bounds + len(input_ids))

            input_ids.extend(seg_ids)
            if role in ("user", "system"):
                labels.extend([-100] * len(seg_ids))
            else:
                # Mask the "<|im_start|>assistant\n" header (same convention
                # as the qwen_omni processor) but keep the content and
                # <|im_end|> supervised.
                seg_labels = list(seg_ids)
                for k in range(min(3, len(seg_labels))):
                    seg_labels[k] = -100
                labels.extend(seg_labels)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # Audio unk slots are never supervised (their embeddings are replaced).
        labels[input_ids == self.tokenizer.unk_token_id] = -100

        # process_image returns batched (bs == 1) structures:
        # pixel_values [sample][image] tensors, image_sizes [sample][image],
        # tgt_sizes [sample] [n_slices, 2] tensors — used as-is.
        return dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            tgt_sizes=tgt_sizes,
            image_bound=[torch.cat(image_bounds, dim=0) if image_bounds else torch.zeros(0, 2, dtype=torch.long)],
            audio_features=[audio_features] if audio_features is not None else [[]],
            audio_feature_lens=[audio_feature_lens],
            audio_bounds=[torch.cat(audio_bounds, dim=0) if audio_bounds else torch.zeros(0, 2, dtype=torch.long)],
            spk_bounds=[torch.cat(spk_bounds, dim=0) if spk_bounds else torch.zeros(0, 2, dtype=torch.long)],
        )

    def _render_simplex_segments(
        self,
        hf_messages: List[Dict],
        system_message: str,
        add_system_prompt: bool,
    ) -> List[tuple]:
        """Render each message as a Qwen-style block with media markers.

        Returns a list of ``(role, block_text)`` tuples whose concatenation is
        the full conversation string.
        """
        """Render the conversation with the tokenizer's official chat template
        and split it back into per-message segments.

        Media content items are inlined as ``<image>./</image>`` /
        ``<audio>./</audio>`` markers before rendering, so the official
        template semantics (including the thinking markers it emits for the
        final assistant turn) are reproduced exactly.

        Returns a list of ``(role, block_text)`` tuples whose concatenation is
        the full conversation string.
        """
        tmpl_messages = []
        if add_system_prompt and hf_messages and hf_messages[0]["role"] != "system":
            tmpl_messages.append({"role": "system", "content": system_message})

        for message in hf_messages:
            role = message["role"]
            content = message.get("content", [])
            if isinstance(content, str):
                text = content
            else:
                parts = []
                for item in content:
                    item_type = item.get("type")
                    if item_type == "image":
                        parts.append(IMAGE_MARKER)
                    elif item_type == "audio":
                        parts.append(AUDIO_MARKER)
                    elif item_type == "text":
                        parts.append(item.get("text", ""))
                text = "".join(parts)
            tmpl_messages.append({"role": role, "content": text})

        full = self.tokenizer.apply_chat_template(tmpl_messages, tokenize=False, add_generation_prompt=False)

        segments = []
        cursor = 0
        for message in tmpl_messages:
            header = f"<|im_start|>{message['role']}\n"
            start = full.index(header, cursor)
            nxt = full.find("<|im_start|>", start + len(header))
            end = nxt if nxt != -1 else len(full)
            segments.append((message["role"], full[start:end]))
            cursor = end
        return segments

    # ------------------------------------------------------------------
    # Duplex / realtime processing
    # ------------------------------------------------------------------

    def process_realtime(
        self,
        env_audio: np.ndarray,
        segments: List[Dict],
        frames: Optional[List[Image]] = None,
        unit_duration: Optional[float] = None,
        sampling_rate: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build the training inputs for one duplex (realtime) session.

        Args:
            env_audio: the continuous 16 kHz mono stream (world + user audio).
            segments: assistant turns, each
                ``{"start_sec": float, "end_sec": float, "text": str}`` or
                with char-level ``twt: [{"text": c, "start": s, "end": e}]``.
            frames: optional video frames ordered by time (frames_per_unit
                frames per unit are consumed).
        """
        unit_duration = unit_duration or self.unit_duration
        sr = sampling_rate or self.sampling_rate
        tokenizer = self.tokenizer

        total_units = max(1, math.ceil(len(env_audio) / sr / unit_duration))

        # ---- assign characters to units ----
        unit_chars: List[List[str]] = [[] for _ in range(total_units)]
        covered_units = set()
        for seg in sorted(segments, key=lambda s: float(s.get("start_sec", 0.0))):
            start_sec = float(seg.get("start_sec", 0.0))
            end_sec = float(seg.get("end_sec", start_sec))
            text = seg.get("text", "")
            twt = seg.get("twt")
            if twt:
                for item in twt:
                    char = item.get("text", "")
                    if not char:
                        continue
                    unit_idx = min(int(float(item["start"]) / unit_duration), total_units - 1)
                    unit_chars[unit_idx].append(char)
                    covered_units.add(unit_idx)
            elif text:
                # Linear interpolation of characters across [start_sec, end_sec).
                chars = [c for c in text if not c.isspace()] or list(text)
                span = max(end_sec - start_sec, 1e-6)
                for i, char in enumerate(chars):
                    char_time = start_sec + span * i / len(chars)
                    unit_idx = min(int(char_time / unit_duration), total_units - 1)
                    unit_chars[unit_idx].append(char)
                    covered_units.add(unit_idx)

        # ---- slice the audio into unit chunks and extract features ----
        samples_per_unit = int(round(unit_duration * sr))
        unit_audios = [env_audio[i * samples_per_unit : (i + 1) * samples_per_unit] for i in range(total_units)]
        audio_features: Optional[torch.Tensor] = None
        audio_feature_lens = torch.zeros(0, dtype=torch.long)
        audio_phs: List[str] = []
        if any(len(a) > 0 for a in unit_audios):
            feats, lens_list, phs_list = self.processor.audio_feature_extract([unit_audios], None, False, sr)
            if isinstance(feats, torch.Tensor) and feats.numel() > 0:
                audio_features = feats
            if lens_list:
                audio_feature_lens = torch.as_tensor(lens_list[0], dtype=torch.long)
            if phs_list:
                audio_phs = list(phs_list[0])

        # ---- image features (frames_per_unit frames per unit) ----
        pixel_values, image_sizes, tgt_sizes = [], [], []
        fpu = self.frames_per_unit
        unit_frames = [frames[u * fpu : (u + 1) * fpu] for u in range(total_units)] if frames else [[] for _ in range(total_units)]
        flat_frames = [f for fs in unit_frames for f in fs]
        if flat_frames:
            image_inputs = self.processor.process_image(flat_frames, do_pad=True, max_slice_nums=1)
            pixel_values = image_inputs["pixel_values"]
            image_sizes = image_inputs["image_sizes"]
            tgt_sizes = image_inputs["tgt_sizes"]

        # ---- build the unit sequence ----
        listen_id = tokenizer.convert_tokens_to_ids("<|listen|>")
        speak_id = tokenizer.convert_tokens_to_ids("<|speak|>")
        chunk_tts_eos_id = tokenizer.convert_tokens_to_ids("<|chunk_tts_eos|>")
        unit_open_id = tokenizer.convert_tokens_to_ids("<unit>")
        unit_close_id = tokenizer.convert_tokens_to_ids("</unit>")

        input_ids: List[int] = []
        labels: List[int] = []
        image_bounds: List[torch.Tensor] = []
        audio_bounds: List[torch.Tensor] = []
        spk_bounds: List[torch.Tensor] = []

        def _append(token_str_or_id, supervise: bool):
            if isinstance(token_str_or_id, str):
                token_id = tokenizer.convert_tokens_to_ids(token_str_or_id)
            else:
                token_id = token_str_or_id
            input_ids.append(token_id)
            labels.append(token_id if supervise else -100)

        _append("<|im_start|>", False)
        for tid in tokenizer.encode(f"system\n{DUPLEX_SYSTEM_MESSAGE}<|im_end|>\n", add_special_tokens=False):
            input_ids.append(tid)
            labels.append(-100)

        frame_cursor = 0
        audio_cursor = 0
        for unit_idx in range(total_units):
            offset = len(input_ids)
            _append(unit_open_id, False)

            # vision block: one <image>./</image> per frame of this unit
            for _ in unit_frames[unit_idx]:
                for tid in tokenizer.encode(IMAGE_MARKER, add_special_tokens=False):
                    input_ids.append(tid)
                    labels.append(-100)
                frame_cursor += 1

            # audio block: one placeholder chunk per unit
            if audio_cursor < len(audio_phs):
                for tid in tokenizer.encode(audio_phs[audio_cursor], add_special_tokens=False):
                    input_ids.append(tid)
                    labels.append(-100)
                audio_cursor += 1

            # content block
            if unit_idx in covered_units:
                _append(speak_id, True)
                for tid in tokenizer.encode("".join(unit_chars[unit_idx]), add_special_tokens=False):
                    input_ids.append(tid)
                    labels.append(tid)
                _append(chunk_tts_eos_id, True)
            else:
                _append(listen_id, True)

            _append(unit_close_id, False)

            # bounds for this unit (already absolute)
            unit_tensor = torch.tensor(input_ids[offset:], dtype=torch.long)
            _, unit_image_bounds, unit_audio_bounds, unit_spk_bounds = self._convert(unit_tensor)
            image_bounds.append(unit_image_bounds + offset)
            audio_bounds.append(unit_audio_bounds + offset)
            spk_bounds.append(unit_spk_bounds + offset)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        labels[input_ids == self.tokenizer.unk_token_id] = -100

        cat = lambda bs: torch.cat(bs, dim=0) if bs else torch.zeros(0, 2, dtype=torch.long)
        # process_image returns batched (bs == 1) structures; used as-is.
        return dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            tgt_sizes=tgt_sizes,
            image_bound=[cat(image_bounds)],
            audio_features=[audio_features] if audio_features is not None else [[]],
            audio_feature_lens=[audio_feature_lens],
            audio_bounds=[cat(audio_bounds)],
            spk_bounds=[cat(spk_bounds)],
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _convert(self, input_ids: torch.Tensor):
        """Compute image/audio/spk bounds for a flat id tensor.

        Mirrors ``MiniCPMOProcessor._convert`` but keeps ``<|listen|>``
        tokens (required for duplex) and accepts a pre-built tensor.
        """
        tokenizer = self.tokenizer
        input_ids = input_ids.long()

        start_cond = (input_ids == tokenizer.im_start_id) | (input_ids == tokenizer.slice_start_id)
        end_cond = (input_ids == tokenizer.im_end_id) | (input_ids == tokenizer.slice_end_id)
        image_start_idx = torch.where(start_cond)[0] + 1
        image_end_idx = torch.where(end_cond)[0]
        valid = max(len(image_start_idx), len(image_end_idx))
        image_bounds = torch.hstack(
            [image_start_idx[:valid].unsqueeze(-1), image_end_idx[:valid].unsqueeze(-1)]
        ) if valid else torch.zeros(0, 2, dtype=torch.long)

        audio_start_idx = torch.where(input_ids == tokenizer.audio_start_id)[0]
        audio_end_idx = torch.where(input_ids == tokenizer.audio_end_id)[0]
        assert len(audio_start_idx) == len(audio_end_idx), "unbalanced <|audio_start|>/<|audio_end|>"
        audio_bounds = (
            torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])
            if len(audio_start_idx)
            else torch.zeros(0, 2, dtype=torch.long)
        )

        spk_start_idx = torch.where(input_ids == tokenizer.spk_start_id)[0]
        spk_end_idx = torch.where(input_ids == tokenizer.spk_end_id)[0]
        assert len(spk_start_idx) == len(spk_end_idx), "unbalanced <|spk_bos|>/<|spk_eos|>"
        spk_bounds = (
            torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])
            if len(spk_start_idx)
            else torch.zeros(0, 2, dtype=torch.long)
        )
        return None, image_bounds, audio_bounds, spk_bounds
