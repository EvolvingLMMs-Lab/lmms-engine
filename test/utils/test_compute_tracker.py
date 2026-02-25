"""Tests for ComputeTracker, format_flops, get_device_tdp, and FlopsCounter 3-tuple return."""

import json
import math
import os
import tempfile
import time
import unittest

from lmms_engine.utils.compute_tracker import ComputeTracker, ComputeSummary
from lmms_engine.utils.train_utils import TrainUtilities


class TestFormatFlops(unittest.TestCase):
    """Tests for TrainUtilities.format_flops()."""

    def test_zero(self):
        self.assertEqual(TrainUtilities.format_flops(0), "0 FLOPS")

    def test_none(self):
        self.assertEqual(TrainUtilities.format_flops(None), "0 FLOPS")

    def test_small_value(self):
        self.assertEqual(TrainUtilities.format_flops(500), "500 FLOPS")

    def test_megaflops(self):
        self.assertEqual(TrainUtilities.format_flops(1.23e6), "1.23 MFLOPS")

    def test_gigaflops(self):
        self.assertEqual(TrainUtilities.format_flops(1.23e9), "1.23 GFLOPS")

    def test_teraflops(self):
        self.assertEqual(TrainUtilities.format_flops(1.23e12), "1.23 TFLOPS")

    def test_petaflops(self):
        self.assertEqual(TrainUtilities.format_flops(1.23e15), "1.23 PFLOPS")

    def test_exaflops(self):
        self.assertEqual(TrainUtilities.format_flops(1.23e18), "1.23 EFLOPS")

    def test_large_exaflops(self):
        # 999 EFLOPS should still use EFLOPS
        self.assertEqual(TrainUtilities.format_flops(999e18), "999.00 EFLOPS")

    def test_negative(self):
        self.assertEqual(TrainUtilities.format_flops(-1.23e12), "-1.23 TFLOPS")

    def test_inf(self):
        self.assertEqual(TrainUtilities.format_flops(float("inf")), "0 FLOPS")

    def test_nan(self):
        self.assertEqual(TrainUtilities.format_flops(float("nan")), "0 FLOPS")


class TestGetDeviceTdp(unittest.TestCase):
    """Tests for TrainUtilities.get_device_tdp()."""

    def test_returns_float(self):
        tdp = TrainUtilities.get_device_tdp()
        self.assertIsInstance(tdp, float)
        self.assertGreaterEqual(tdp, 0.0)


class TestComputeTracker(unittest.TestCase):
    """Tests for ComputeTracker class."""

    def test_init_defaults(self):
        tracker = ComputeTracker()
        self.assertEqual(tracker.num_gpus, 1)
        self.assertAlmostEqual(tracker.carbon_intensity, 0.475)
        self.assertEqual(tracker.gpu_tdp_watts, 0.0)
        self.assertEqual(tracker.gpu_name, "unknown")
        self.assertEqual(tracker._total_flops, 0.0)
        self.assertIsNone(tracker._start_time)

    def test_init_custom(self):
        tracker = ComputeTracker(
            num_gpus=8,
            carbon_intensity=0.581,
            gpu_tdp_watts=700.0,
            gpu_name="NVIDIA H100",
        )
        self.assertEqual(tracker.num_gpus, 8)
        self.assertAlmostEqual(tracker.carbon_intensity, 0.581)
        self.assertEqual(tracker.gpu_tdp_watts, 700.0)
        self.assertEqual(tracker.gpu_name, "NVIDIA H100")

    def test_start(self):
        tracker = ComputeTracker()
        tracker.start()
        self.assertIsNotNone(tracker._start_time)
        self.assertAlmostEqual(tracker._start_time, time.time(), delta=1.0)

    def test_accumulate_flops(self):
        tracker = ComputeTracker()
        tracker.accumulate_flops(1e15)
        self.assertEqual(tracker._total_flops, 1e15)
        tracker.accumulate_flops(2e15)
        self.assertEqual(tracker._total_flops, 3e15)
        tracker.accumulate_flops(0)
        self.assertEqual(tracker._total_flops, 3e15)

    def test_state_dict_and_load(self):
        tracker = ComputeTracker(num_gpus=4, gpu_tdp_watts=400.0)
        tracker.start()
        tracker.accumulate_flops(5e15)

        sd = tracker.state_dict()
        self.assertIn("total_flops", sd)
        self.assertIn("start_time", sd)
        self.assertEqual(sd["total_flops"], 5e15)

        # Load into a fresh tracker
        tracker2 = ComputeTracker(num_gpus=4, gpu_tdp_watts=400.0)
        tracker2.load_state_dict(sd)
        self.assertEqual(tracker2._total_flops, 5e15)
        self.assertEqual(tracker2._start_time, sd["start_time"])

    def test_finish_summary(self):
        tracker = ComputeTracker(
            num_gpus=8,
            carbon_intensity=0.475,
            gpu_tdp_watts=700.0,
            gpu_name="NVIDIA H100",
        )
        tracker.start()
        tracker.accumulate_flops(1.23e18)

        summary = tracker.finish()
        self.assertIsInstance(summary, ComputeSummary)
        self.assertEqual(summary.total_flops, 1.23e18)
        self.assertEqual(summary.total_flops_formatted, "1.23 EFLOPS")
        self.assertEqual(summary.gpu_name, "NVIDIA H100")
        self.assertEqual(summary.gpu_tdp_watts, 700.0)
        self.assertEqual(summary.num_gpus, 8)
        self.assertAlmostEqual(summary.carbon_intensity_kg_per_kwh, 0.475)
        self.assertGreaterEqual(summary.training_duration_seconds, 0.0)
        self.assertGreaterEqual(summary.energy_kwh, 0.0)
        self.assertGreaterEqual(summary.co2_kg, 0.0)
        self.assertIn("CO2", summary.co2_formatted)

    def test_finish_with_zero_tdp(self):
        """Unknown GPU should give 0 energy and 0 CO2."""
        tracker = ComputeTracker(
            num_gpus=1,
            carbon_intensity=0.475,
            gpu_tdp_watts=0.0,
            gpu_name="unknown",
        )
        tracker.start()
        tracker.accumulate_flops(1e15)

        summary = tracker.finish()
        self.assertEqual(summary.energy_kwh, 0.0)
        self.assertEqual(summary.co2_kg, 0.0)
        self.assertEqual(summary.co2_formatted, "0 g CO2")

    def test_save_summary_creates_json(self):
        summary = ComputeSummary(
            total_flops=1.23e18,
            total_flops_formatted="1.23 EFLOPS",
            training_duration_seconds=9015.0,
            training_duration_formatted="2h 30m 15s",
            gpu_name="NVIDIA H100 80GB HBM3",
            gpu_tdp_watts=700.0,
            num_gpus=8,
            energy_kwh=14.02,
            carbon_intensity_kg_per_kwh=0.475,
            co2_kg=6.66,
            co2_formatted="6.66 kg CO2",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ComputeTracker.save_summary(tmpdir, summary)
            path = os.path.join(tmpdir, "compute_summary.json")
            self.assertTrue(os.path.exists(path))

            with open(path) as f:
                data = json.load(f)

            self.assertEqual(data["total_flops"], 1.23e18)
            self.assertEqual(data["total_flops_formatted"], "1.23 EFLOPS")
            self.assertEqual(data["training_duration_seconds"], 9015.0)
            self.assertEqual(data["training_duration_formatted"], "2h 30m 15s")
            self.assertEqual(data["gpu_name"], "NVIDIA H100 80GB HBM3")
            self.assertEqual(data["gpu_tdp_watts"], 700.0)
            self.assertEqual(data["num_gpus"], 8)
            self.assertAlmostEqual(data["energy_kwh"], 14.02)
            self.assertAlmostEqual(data["carbon_intensity_kg_per_kwh"], 0.475)
            self.assertAlmostEqual(data["co2_kg"], 6.66)
            self.assertEqual(data["co2_formatted"], "6.66 kg CO2")

    def test_save_summary_creates_directory(self):
        """save_summary should create the output directory if it doesn't exist."""
        summary = ComputeSummary(
            total_flops=0,
            total_flops_formatted="0 FLOPS",
            training_duration_seconds=0,
            training_duration_formatted="0s",
            gpu_name="test",
            gpu_tdp_watts=0,
            num_gpus=1,
            energy_kwh=0,
            carbon_intensity_kg_per_kwh=0.475,
            co2_kg=0,
            co2_formatted="0.00 kg CO2",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "dir")
            ComputeTracker.save_summary(nested, summary)
            self.assertTrue(os.path.exists(os.path.join(nested, "compute_summary.json")))


class TestFormatDuration(unittest.TestCase):
    """Tests for ComputeTracker._format_duration()."""

    def test_seconds_only(self):
        self.assertEqual(ComputeTracker._format_duration(45), "45s")

    def test_minutes_and_seconds(self):
        self.assertEqual(ComputeTracker._format_duration(125), "2m 5s")

    def test_hours_minutes_seconds(self):
        self.assertEqual(ComputeTracker._format_duration(9015), "2h 30m 15s")

    def test_zero(self):
        self.assertEqual(ComputeTracker._format_duration(0), "0s")

    def test_exact_hour(self):
        self.assertEqual(ComputeTracker._format_duration(3600), "1h 0s")

    def test_exact_minute(self):
        self.assertEqual(ComputeTracker._format_duration(60), "1m 0s")


class TestFlopsCounterThreeTuple(unittest.TestCase):
    """Test that FlopsCounter.estimate_flops() returns a 3-tuple."""

    def test_estimate_flops_returns_three_values(self):
        from transformers import AutoConfig

        from lmms_engine.models.utils import FlopsCounter

        # Use a minimal Qwen2 config
        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
        counter = FlopsCounter(config)

        batch_seqlens = [128, 256]
        delta_time = 1.0
        result = counter.estimate_flops(batch_seqlens, delta_time)

        self.assertEqual(len(result), 3, "estimate_flops should return a 3-tuple")
        estimated_flops, promised_flops, raw_flops = result
        self.assertIsInstance(estimated_flops, (int, float))
        self.assertIsInstance(promised_flops, (int, float))
        self.assertIsInstance(raw_flops, (int, float))
        self.assertGreater(raw_flops, 0, "raw_flops should be positive for a known model")
        self.assertGreater(estimated_flops, 0, "estimated_flops should be positive")

    def test_unknown_model_returns_zeros(self):
        from unittest.mock import MagicMock

        from lmms_engine.models.utils import FlopsCounter

        # Create a mock config with an unknown model type
        mock_config = MagicMock()
        mock_config.model_type = "some_unknown_model"
        counter = FlopsCounter(mock_config)

        batch_seqlens = [128]
        delta_time = 1.0
        result = counter.estimate_flops(batch_seqlens, delta_time)

        self.assertEqual(len(result), 3)
        estimated_flops, promised_flops, raw_flops = result
        self.assertEqual(estimated_flops, 0)
        self.assertEqual(raw_flops, 0)


class TestCarbonIntensityConfig(unittest.TestCase):
    """Test that the carbon_intensity config field works."""

    def test_default_value(self):
        from lmms_engine.train.config import TrainingArguments

        args = TrainingArguments(output_dir="/tmp/test")
        self.assertAlmostEqual(args.carbon_intensity, 0.475)

    def test_custom_value(self):
        from lmms_engine.train.config import TrainingArguments

        args = TrainingArguments(output_dir="/tmp/test", carbon_intensity=0.581)
        self.assertAlmostEqual(args.carbon_intensity, 0.581)


class TestCheckpointRoundTrip(unittest.TestCase):
    """Test that accumulation is preserved across checkpoint save/load."""

    def test_roundtrip(self):
        tracker1 = ComputeTracker(num_gpus=4, gpu_tdp_watts=700.0, gpu_name="test")
        tracker1.start()
        tracker1.accumulate_flops(1e15)
        tracker1.accumulate_flops(2e15)

        sd = tracker1.state_dict()
        self.assertEqual(sd["total_flops"], 3e15)

        tracker2 = ComputeTracker(num_gpus=4, gpu_tdp_watts=700.0, gpu_name="test")
        tracker2.load_state_dict(sd)
        # Continue accumulating after "resume"
        tracker2.accumulate_flops(4e15)
        self.assertEqual(tracker2._total_flops, 7e15)

        summary = tracker2.finish()
        self.assertEqual(summary.total_flops, 7e15)
        self.assertEqual(summary.total_flops_formatted, "7.00 PFLOPS")


if __name__ == "__main__":
    unittest.main()
