import unittest
from unittest import TestCase

from utils import with_temp_dir

from lmms_engine.launch.cli import create_train_task


class TestQwen2_5_VL(TestCase):
    @with_temp_dir
    def test_text_train_fsdp2(self, temp_dir):
        cfg = {
            "trainer_type": "fsdp2_trainer",
            "dataset_config": {
                "dataset_type": "vision",
                "dataset_format": "yaml",
                "datasets": [
                    {
                        "path": "data/open_thoughts_debug",
                        "data_folder": "",
                        "data_type": "arrow",
                    }
                ],
                "processor_config": {
                    "processor_name": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "processor_type": "qwen2_5_vl",
                },
                "packing": False,
                "video_backend": "qwen_vl_utils",
            },
            "model_config": {
                "load_from_pretrained_path": "Qwen/Qwen2.5-VL-3B-Instruct",
                "attn_implementation": "flash_attention_2",
            },
            "per_device_train_batch_size": 1,
            "gradient_checkpointing": True,
            "num_train_epochs": 1,
            "max_steps": 10,
            "report_to": "none",
            "output_dir": temp_dir,
            "warmup_ratio": 0.0,
            "eval_strategy": "no",
            "dataloader_num_workers": 8,
            "bf16": True,
            "lr_scheduler_type": "cosine",
            "group_by_length": True,
            "use_liger_kernel": True,
            "use_rmpad": True,
            "fsdp2": True,
            "fsdp_config": {
                "transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer"],
                "reshard_after_forward": False,
            },
            "sp_ulysses_degree": 1,
        }

        train_task = create_train_task(cfg)
        train_task.build()
        train_task.run()


if __name__ == "__main__":
    unittest.main()
