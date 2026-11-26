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

"""Collator for MiniCPM-o training batches.

Pads ``input_ids`` / ``labels`` and groups the nested modality structures
expected by ``MiniCPMO.get_vision_embedding`` / ``get_omni_embedding``:

- ``pixel_values``: list (batch) of lists of per-image ``[3, ps, np]`` tensors.
- ``tgt_sizes``: flat list of ``[h, w]`` tensors across the whole batch.
- ``image_bound`` / ``audio_bounds`` / ``spk_bounds``: list (batch) of
  ``[n, 2]`` tensors.
- ``audio_features``: ``(bs, 80, max_total_frames)`` — per-sample chunks are
  concatenated along time (the Whisper encoder uses 1 s chunked attention).
- ``audio_feature_lens``: list (batch) of 1-D chunk-length tensors.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class MiniCPMOCollator:
    processor: object

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]

        inputs = defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        batched_inputs: Dict = {}

        # ---- input_ids / labels / attention_mask ----
        pad_id = self.processor.pad_token_id
        input_ids = inputs.pop("input_ids")
        batched_inputs["input_ids"] = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)

        if "labels" in inputs:
            labels = inputs.pop("labels")
            batched_inputs["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)

        batched_inputs["attention_mask"] = batched_inputs["input_ids"].ne(pad_id).long()

        # ---- nested modality structures ----
        pixel_values = inputs.pop("pixel_values", [])
        tgt_sizes = inputs.pop("tgt_sizes", [])
        image_sizes = inputs.pop("image_sizes", [])
        image_bound = inputs.pop("image_bound", [])
        audio_bounds = inputs.pop("audio_bounds", [])
        spk_bounds = inputs.pop("spk_bounds", [])
        audio_feature_lens = inputs.pop("audio_feature_lens", [])
        audio_features = inputs.pop("audio_features", [])

        if pixel_values:
            batched_inputs["pixel_values"] = [list(per_sample) for per_sample in pixel_values]
            # Flat list across the batch (get_vision_embedding vstacks it).
            batched_inputs["tgt_sizes"] = [t for per_sample in tgt_sizes for t in per_sample]
            batched_inputs["image_sizes"] = [list(per_sample) for per_sample in image_sizes]
            batched_inputs["image_bound"] = [self._as_bound_tensor(b) for b in image_bound]
        if audio_features:
            batched_inputs["audio_features"] = self._collate_audio(audio_features)
            batched_inputs["audio_feature_lens"] = [self._as_lens_tensor(lens) for lens in audio_feature_lens]
            batched_inputs["audio_bounds"] = [self._as_bound_tensor(b) for b in audio_bounds]
            batched_inputs["spk_bounds"] = [self._as_bound_tensor(b) for b in spk_bounds]

        # ---- remaining keys ----
        for key, values in inputs.items():
            if isinstance(values[0], (bool, int, float)) and not torch.is_tensor(values[0]):
                batched_inputs[key] = values[0]
            else:
                batched_inputs[key] = list(values)

        return batched_inputs

    @staticmethod
    def _as_bound_tensor(bound) -> torch.Tensor:
        """Coerce a bounds value (tensor, or list of tensors / ints) to [n, 2]."""
        if torch.is_tensor(bound):
            return bound.long().reshape(-1, 2)
        if isinstance(bound, (list, tuple)):
            if len(bound) == 0:
                return torch.zeros(0, 2, dtype=torch.long)
            if torch.is_tensor(bound[0]):
                return torch.cat([b.long().reshape(-1, 2) for b in bound], dim=0)
        return torch.as_tensor(bound, dtype=torch.long).reshape(-1, 2)

    @staticmethod
    def _as_lens_tensor(lens) -> torch.Tensor:
        """Coerce chunk lengths (tensor, int, or list thereof) to a 1-D tensor."""
        if torch.is_tensor(lens):
            return lens.long().reshape(-1)
        if isinstance(lens, (list, tuple)):
            if len(lens) == 0:
                return torch.zeros(0, dtype=torch.long)
            if torch.is_tensor(lens[0]):
                return torch.cat([torch.as_tensor(l).long().reshape(-1) for l in lens])
        return torch.as_tensor(lens, dtype=torch.long).reshape(-1)

    @staticmethod
    def _collate_audio(audio_features: List) -> torch.Tensor:
        """Stack per-sample chunk tensors into one flat batch.

        Each sample's audio features are a ``[n_chunks, 80, max_T]`` tensor
        (one row per 1-second chunk, as returned by
        ``MiniCPMOProcessor.audio_feature_extract``).  The model's Whisper
        encoder treats each chunk as an independent batch row.

        Returns:
            ``(n_total_chunks, 80, max_T)`` float tensor.
        """
        per_sample = []
        for sample in audio_features:
            if torch.is_tensor(sample):
                t = sample if sample.dim() == 3 else sample.unsqueeze(0)
                per_sample.append(t)
            elif isinstance(sample, (list, tuple)) and len(sample) > 0 and torch.is_tensor(sample[0]):
                first = sample[0]
                if first.dim() == 3:
                    # a single wrapped [n_chunks, 80, T] tensor
                    per_sample.append(first)
                else:
                    # a list of [80, T] chunks
                    per_sample.append(torch.stack(list(sample)))
        if not per_sample:
            return torch.zeros(0)
        max_t = max(t.shape[-1] for t in per_sample)
        n_total = sum(t.shape[0] for t in per_sample)
        batch = torch.zeros(n_total, per_sample[0].shape[1], max_t, dtype=per_sample[0].dtype)
        offset = 0
        for t in per_sample:
            batch[offset : offset + t.shape[0], :, : t.shape[-1]] = t
            offset += t.shape[0]
        return batch
