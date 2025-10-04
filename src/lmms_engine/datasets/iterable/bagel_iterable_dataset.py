from lmms_engine.datasets.collator import BagelCollator
from lmms_engine.mapping_func import register_dataset

from .vision_iterable_dataset import VisionSFTIterableDataset


@register_dataset("bagel_iterable")
class BagelIterableDataset(VisionSFTIterableDataset):
    def get_collator(self):
        return BagelCollator(self.processor)
