from lmms_engine.datasets.config import DatasetConfig
from lmms_engine.mapping_func import register_dataset
from datasets import load_dataset

@register_dataset("nit")
class NitDataset:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.processor = NitDataProcessor(config)
    
    def _build_from_config(self):
        # A bit ugly, but it seems that I cannot merge with multimodal dataset
        if self.config.dataset_format == "hf_dataset":
            self.dataset = load_dataset(self.config.dataset_path, split="train")
        else:
            raise NotImplementedError("Only hf_dataset is supported for now")
        
        self.dataset = self.dataset.map(self.processor.process, num_proc=self.config.processor_workers)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

