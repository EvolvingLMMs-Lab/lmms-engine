from typing import override

from lmms_engine.mapping_func import register_dataset

from .collator import BagelCollator
from .vision_dataset import VisionSFTDataset


@register_dataset("bagel")
class BagelSFTDataset(VisionSFTDataset):
    @override
    def get_collator(self):
        return BagelCollator(self.processor)
