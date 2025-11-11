import os
import torch
from lmms_engine.datasets.naive.multimodal_dataset import MultiModalDataset
from lmms_engine.mapping_func import register_dataset
from lmms_engine.datasets.collator import LLaVACollator, VisionCollator
from typing import Dict


QUESTION_TEMPLATE = "Please generate the image given the following information: \n caption: {Question} \n metadata: {Metadata}"

@register_dataset("t2i_unirl")
class T2IUniRLDataset(MultiModalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        caption = data.pop("prompt")
        prompt = [
            {
                "role": "user",
                "content": [{"type": "text", "text": QUESTION_TEMPLATE.format(Question=caption, Metadata=data)}],
            }
        ]
        inputs = self.processor.process(images=None, hf_messages=prompt, videos=None)
        return inputs
        
    def estimate_data_tokens_per_row(self, row):
        length = len(str(row).split())
        return length

    def get_collator(self):
        if self.processor_config.processor_type == "llava":
            return LLaVACollator(self.processor)
        else:
            return VisionCollator(self.processor)