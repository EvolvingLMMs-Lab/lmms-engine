import os
import torch
from lmms_engine.datasets.naive.multimodal_dataset import MultiModalDataset
from lmms_engine.mapping_func import register_dataset
from lmms_engine.datasets.collator import LLaVACollator, Blip3oNextRLCollator
from typing import Dict


QUESTION_TEMPLATE_1 = "Please generate the image given the following information: \n caption: {Question} \n metadata: {Metadata}"

QUESTION_TEMPLATE_2 = "Please extend the caption based on the metadata provided, making it more detailed and specific: \ncaption: {Question} \nmetadata: {Metadata}"

@register_dataset("t2i_unirl")
class T2IUniRLDataset(MultiModalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        caption = data.pop("prompt")
        text_1 = QUESTION_TEMPLATE_1.format(Question=caption, Metadata=data)
        text_2 = QUESTION_TEMPLATE_2.format(Question=caption, Metadata=data)
        prompt = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text_2}],
            }
        ]
        inputs = self.processor.process(images=None, hf_messages=prompt, videos=None)
        inputs['caption'] = caption
        return inputs
        
    def estimate_data_tokens_per_row(self, row):
        length = len(str(row).split())
        return length

    def get_collator(self):
        return Blip3oNextRLCollator(self.processor)