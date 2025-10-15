from datasets import load_dataset

from lmms_engine.datasets.config import DatasetConfig
from lmms_engine.mapping_func import register_dataset

from .base_dataset import BaseDataset


@register_dataset("sit")
class SitDataset(BaseDataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def _build_from_config(self):
        self.dataset = load_dataset(self.config.dataset_path, split="train")

    def __getitem__(self, index):
        x = self.dataset[index]["image"].convert("RGB")
        y = self.dataset[index]["label"]
        x, y = self.processor.process(x, y)
        return dict(x=x, y=y)

    def __len__(self):
        return len(self.dataset)

    # Use pytorch default collator
    def get_collator(self):
        return None
