import os
import pathlib
import unittest

from torch.utils.data import DataLoader

from lmms_engine.datasets import DatasetConfig, DatasetFactory
from lmms_engine.utils.train_utils import TrainUtilities

current_dir = pathlib.Path().resolve()
data_folder = current_dir.parent.parent / "examples" / "sample_json_data"


class TestVisionDataset(unittest.TestCase):
    def test_sft_dataset(self):
        config = {
            "dataset_type": "vision",
            "dataset_format": "json",
            "dataset_path": os.path.join(str(data_folder), "lmms_engine.json"),
            "chat_template": "qwen",
            "processor_config": {
                "processor_name": "Qwen/Qwen2-VL-7B-Instruct",
                "processor_modality": "vision",
                "processor_type": "qwen2_vl",
            },
        }

        dataset_config = DatasetConfig(**config)
        dataset = DatasetFactory.create_dataset(dataset_config)
        dataset.build()
        collator = dataset.get_collator()
        dataLoader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collator)
        for data in dataLoader:
            # TrainUtilities.sanity_check_labels(dataset.processor.processor, data["input_ids"], data["labels"])
            print([f"{key}: {value.shape}" for key, value in data.items()])
            break

    def test_load_hf_dataset(self):
        config = {
            "dataset_type": "vision",
            "dataset_format": "hf_dataset",
            "dataset_path": "kcz358/LLaVA-NeXT-20k",
            "chat_template": "qwen",
            "processor_config": {
                "processor_name": "Qwen/Qwen2-VL-7B-Instruct",
                "processor_modality": "vision",
                "processor_type": "qwen2_vl",
            },
        }

        dataset_config = DatasetConfig(**config)
        dataset = DatasetFactory.create_dataset(dataset_config)
        dataset.build()
        collator = dataset.get_collator()
        dataLoader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collator)
        for data in dataLoader:
            # TrainUtilities.sanity_check_labels(dataset.processor.processor, data["input_ids"], data["labels"])
            print([f"{key}: {value.shape}" for key, value in data.items()])
            break


if __name__ == "__main__":
    unittest.main()
