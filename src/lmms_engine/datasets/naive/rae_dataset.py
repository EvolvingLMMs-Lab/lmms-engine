from datasets import load_dataset

from lmms_engine.datasets.config import DatasetConfig
from lmms_engine.datasets.naive.vision_dataset import VisionSFTDataset
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("rae")
class RaeDataset(VisionSFTDataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def _build_from_config(self):
        self.dataset = load_dataset(self.config.dataset_path, split="train")

    def __getitem__(self, index):
        images = [self.dataset[index]["image"]]
        hf_messages = None
        return self.processor.process(images=images, hf_messages=hf_messages)

    def __len__(self):
        return len(self.dataset)
