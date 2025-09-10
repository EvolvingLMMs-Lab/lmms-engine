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
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.processor.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        batched_inputs = collections.defaultdict(list)
        for keys, values in inputs.items():
            if isinstance(values[0], torch.Tensor):
                batched_inputs[keys] = torch.stack(values, dim=0)
            elif isinstance(values[0], list):
                for value in values:
                    batched_inputs[keys].extend(value)
            else:
                batched_inputs[keys].append(values)

        # Fake input ids for packing
        # seq_len = sum(batched_inputs['sample_lens'])
        # text_length = batched_inputs['packed_text_ids'].shape[0]
        # vae_length = getattr(batched_inputs, 'packed_vae_token_indexes', torch.tensor(0)).shape[0]
        # vit_length = getattr(batched_inputs, 'packed_vit_token_indexes', torch.tensor(0)).shape[0]
        # assert seq_len == (text_length + vae_length + vit_length)
        return batched_inputs

    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
