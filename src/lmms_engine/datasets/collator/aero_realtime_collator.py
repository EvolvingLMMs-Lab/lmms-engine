import collections
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

from .vision_collator import VisionCollator


@dataclass
class AeroRealtimeCollator(VisionCollator):
    """Collator for AeroRealtime that additionally pads ``text_stream_ids``."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        batched_inputs = {}

        if "input_ids" in inputs.keys():
            input_ids = inputs.pop("input_ids")
            input_ids = self.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            )
            batched_inputs["input_ids"] = input_ids

        if "labels" in inputs.keys():
            labels = inputs.pop("labels")
            labels = self.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            )
            batched_inputs["labels"] = labels

        if "text_stream_ids" in inputs.keys():
            text_stream_ids = inputs.pop("text_stream_ids")
            text_stream_ids = self.pad_sequence(
                text_stream_ids,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            )
            batched_inputs["text_stream_ids"] = text_stream_ids

        if "attention_mask" in inputs.keys():
            inputs.pop("attention_mask")

        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id).long()
        batched_inputs["attention_mask"] = attention_mask

        # Remaining keys: concatenate tensors, pass through scalars
        for key, values in inputs.items():
            if isinstance(values[0], bool) or (
                isinstance(values[0], (int, float)) and not isinstance(values[0], torch.Tensor)
            ):
                batched_inputs[key] = values[0]
            else:
                # Convert numpy arrays to tensors if needed
                values = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in values]
                # Audio tensors from different samples may have different
                # padded mel time lengths (variable across the outer batch).
                # Right-pad along the last dim before concatenating on dim 0.
                if key == "input_features":
                    batched_inputs[key] = self._concat_pad_last_dim(values, pad_value=0.0)
                elif key == "audio_attention_mask":
                    batched_inputs[key] = self._concat_pad_last_dim(values, pad_value=0)
                else:
                    batched_inputs[key] = torch.concatenate(values, dim=0)
        return batched_inputs

    @staticmethod
    def _concat_pad_last_dim(tensors, pad_value):
        """Right-pad each tensor's last dim to the batch max, then concat on dim 0."""
        max_len = max(t.shape[-1] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[-1] < max_len:
                pad_shape = list(t.shape)
                pad_shape[-1] = max_len - t.shape[-1]
                pad = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad], dim=-1)
            padded.append(t)
        return torch.cat(padded, dim=0)
