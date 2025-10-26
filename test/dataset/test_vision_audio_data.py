import os
import pathlib
import unittest

from torch.utils.data import DataLoader

from lmms_engine.datasets import DatasetConfig, DatasetFactory
from lmms_engine.utils.train_utils import TrainUtilities

current_dir = pathlib.Path().resolve()
data_folder = current_dir.parent.parent / "examples" / "sample_jsonl_data"


class TestVisionDataset(unittest.TestCase):
    def test_sft_dataset(self):
        config = {
            "dataset_type": "vision_audio",
            "dataset_format": "jsonl",
            "dataset_path": os.path.join(str(data_folder), "voice_assis", "voice_assis.jsonl"),
            "processor_config": {
                "processor_name": "Evo-LMM/kino-7b-init",
                "processor_modality": "vision_audio",
                "processor_type": "kino",
            },
        }

        dataset_config = DatasetConfig(**config)
        dataset = DatasetFactory.create_dataset(dataset_config)
        dataset.build()
        collator = dataset.get_collator()
        outputs = dataset.load_from_json(dataset.data_list[0], data_folder=str(current_dir.parent.parent))
        # dataLoader = DataLoader(
        # dataset, batch_size=4, shuffle=False, collate_fn=collator
        # )
        # for data in dataLoader:
        # # TrainUtilities.sanity_check_labels(dataset.processor.processor, data["input_ids"], data["labels"])
        # print([f"{key}: {value.shape}" for key, value in data.items()])
        # break

    def test_kino_qwen2_5_dataset(self):
        config = {
            "dataset_type": "vision_audio",
            "dataset_format": "jsonl",
            "dataset_path": os.path.join(str(data_folder), "voice_assis", "voice_assis.jsonl"),
            "processor_config": {
                "processor_name": "Evo-LMM/kino_qwen2_5_vl_init",
                "processor_modality": "vision_audio",
                "processor_type": "kino_qwen2_5",
            },
        }

        dataset_config = DatasetConfig(**config)
        dataset = DatasetFactory.create_dataset(dataset_config)
        dataset.build()
        collator = dataset.get_collator()
        outputs = dataset.load_from_json(dataset.data_list[0], data_folder=str(current_dir.parent.parent))


if __name__ == "__main__":
    unittest.main()
