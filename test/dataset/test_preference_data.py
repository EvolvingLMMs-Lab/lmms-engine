import os
import pathlib
import unittest

from trl.trainer.dpo_trainer import PreferenceCollator

from lmms_engine.datasets import DatasetConfig, VisionPreferenceDataset

current_dir = pathlib.Path().resolve()
data_folder = current_dir.parent.parent / "examples" / "sample_preference_data"


class TestPreferenceData(unittest.TestCase):
    def test_preference_data(self):
        config = {
            "dataset_type": "vision_preference",
            "dataset_format": "jsonl",
            "dataset_path": os.path.join(str(data_folder), "rlaif-v.jsonl"),
            "processor_config": {
                "processor_name": "Evo-LMM/kino-maas-7B_v12_18000_init",
                "processor_modality": "vision_audio",
                "processor_type": "kino",
            },
        }
        config = DatasetConfig(**config)
        dataset = VisionPreferenceDataset(config)
        dataset.build()
        collator = PreferenceCollator(pad_token_id=0)
        inputs = dataset.load_from_json(dataset.data_list[0], data_folder=str(data_folder / "image"))


if __name__ == "__main__":
    unittest.main()
