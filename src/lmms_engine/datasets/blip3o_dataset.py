from typing import override

from lmms_engine.mapping_func import register_dataset

from .collator import Blip3oCollator
from .vision_dataset import VisionSFTDataset


@register_dataset("blip3o")
class Blip3oSFTDataset(VisionSFTDataset):
    @override
    def get_collator(self):
        return Blip3oCollator(self.processor)
