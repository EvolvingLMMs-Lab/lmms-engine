import os
import unittest
from unittest import TestCase

from utils import launch_torchrun_training, with_multi_gpu_training, with_temp_dir


class Test_Bagel(TestCase):
    @with_temp_dir
    @with_multi_gpu_training
    def test_text_train_fsdp2(self, temp_dir, nproc_per_node):
        """Test Qwen3-VL training with FSDP2 using torchrun subprocess."""

        # Path to the training script
        script_path = os.path.join(os.path.dirname(__file__), "train_bagel.py")

        # Launch training using torchrun
        result = launch_torchrun_training(
            script_path=script_path,
            output_dir=temp_dir,
            nproc_per_node=nproc_per_node,
            timeout=600,  # 10 minutes timeout
        )

        # Assert training completed successfully
        self.assertIsNotNone(result, "Training process should not be None")
        self.assertEqual(
            result.returncode,
            0,
            f"Training failed with return code {result.returncode}",
        )

        # Print training output for debugging
        if result.stdout:
            print("Training stdout:", result.stdout)
        if result.stderr:
            print("Training stderr:", result.stderr)


if __name__ == "__main__":
    unittest.main()
