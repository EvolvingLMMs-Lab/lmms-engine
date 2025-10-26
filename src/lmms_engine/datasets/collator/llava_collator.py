import collections
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

from ...protocol import Processable
from ...utils.train_utils import TrainUtilities
from .vision_collator import VisionCollator


@dataclass
class LLaVACollator(VisionCollator):
    processor: Processable

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

        if "attention_mask" in inputs.keys():
            inputs.pop("attention_mask")

        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id).long()
        batched_inputs["attention_mask"] = attention_mask

        # for the other keys
        for key, values in inputs.items():
            # Handle scalar/boolean values ( use_audio_in_video)
            if isinstance(values[0], bool) or (
                isinstance(values[0], (int, float)) and not isinstance(values[0], torch.Tensor)
            ):
                batched_inputs[key] = values[0]
            elif key == "pixel_values":
                padded_pixel_values = self._pad_for_batching(values)
                batched_inputs[key] = torch.concatenate(padded_pixel_values, dim=0)
            else:
                batched_inputs[key] = torch.concatenate(values, dim=0)
        return batched_inputs

    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)

    def _pad_for_batching(
        self,
        pixel_values: list["torch.Tensor"],
    ) -> list["torch.Tensor"]:
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`list[torch.Tensor]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)

        Returns:
            list[`torch.Tensor`]: The padded images.
        """
        max_patch = max(x.shape[1] for x in pixel_values)
        pixel_values = [
            torch.nn.functional.pad(image, pad=[0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[1]])
            for image in pixel_values
        ]

        return pixel_values
