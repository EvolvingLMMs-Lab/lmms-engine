from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

from lmms_engine.models.blip3o.constants import Blip3oConstants

from ...protocol import Processable


@dataclass
class Blip3oCollator:
    processor: Processable

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        return self.processor.tokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(
        self, instances: Sequence[Dict] | Sequence[Sequence[Dict]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], Sequence):
            instances = [inst for instance in instances for inst in instance]
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [
            _input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids
        ]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                0  # This gets the best result. Don't know why.
            )
        input_ids = self.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = self.pad_sequence(
            labels, batch_first=True, padding_value=Blip3oConstants.IGNORE_INDEX
        )
        batch = dict(
            input_ids=input_ids,
            labels=labels.long() if labels.dtype == torch.int32 else labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            batch["images"] = images

            target_images = [instance["target_image"][0] for instance in instances]
            target_images = torch.stack(target_images, dim=0) if target_images else None
            batch["target_images"] = target_images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]
        return batch

    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
