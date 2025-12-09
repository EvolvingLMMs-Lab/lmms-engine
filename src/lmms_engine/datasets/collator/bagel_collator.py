import collections
from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from ...protocol import Processable


@dataclass
class BagelCollator:
    processor: Processable

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.processor.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.processor.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        current_lens = 0
        for instance in instances:
            for key, values in instance.items():
                if key.endswith("_indexes"):
                    if isinstance(values, torch.Tensor):
                        values = values + current_lens
                    elif isinstance(values, Sequence):
                        values = type(values)(map(lambda x: x + current_lens, values))
                    else:
                        values = values + current_lens
                inputs[key].append(values)
            current_lens += instance["sequence_length"]

        batched_inputs = collections.defaultdict(list)
        sequence_length = inputs.pop("sequence_length")
        sample_lens = inputs.pop("sample_lens")
        for keys, values in inputs.items():
            if keys == "padded_images":
                num_images = sum([item.shape[0] for item in values])
                image_width = [item.shape[2] for item in values]
                image_height = [item.shape[3] for item in values]
                max_image_width = max(image_width)
                max_image_height = max(image_height)
                padded_images = torch.zeros((num_images, 3, max_image_width, max_image_height), dtype=values[0].dtype)
                index = 0
                for image in values:
                    padded_images[index : index + image.shape[0], :, : image.shape[2], : image.shape[3]] = image
                    index += image.shape[0]
                batched_inputs[keys] = padded_images
            elif isinstance(values[0], torch.Tensor):
                batched_inputs[keys] = torch.concatenate(values, dim=0)
            elif isinstance(values[0], list):
                for value in values:
                    batched_inputs[keys].extend(value)
            else:
                batched_inputs[keys].append(values)

        batched_inputs["sequence_length"] = torch.tensor(sequence_length).flatten().sum()
        batched_inputs["sample_lens"] = torch.tensor(sample_lens)
        batched_inputs.pop("curr")
        batched_inputs["patchified_vae_latent_shapes"] = torch.tensor(batched_inputs["patchified_vae_latent_shapes"])

        # Fake input ids for packing
        batched_inputs["input_ids"] = torch.zeros((1, batched_inputs["sequence_length"]), dtype=torch.long)
        batched_inputs["attention_mask"] = torch.ones((1, batched_inputs["sequence_length"]), dtype=torch.long)

        # Make batched input a dict to send to device
        return dict(batched_inputs)

    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
