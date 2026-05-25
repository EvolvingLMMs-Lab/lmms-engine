import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILER_PATH = REPO_ROOT / "src" / "lmms_engine" / "utils" / "profiler.py"
spec = importlib.util.spec_from_file_location("lmms_engine_utils_profiler", PROFILER_PATH)
profiler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler_module)
CudaEventProfiler = profiler_module.CudaEventProfiler


class FakeCudaEvent:
    def record(self):
        return None

    def query(self):
        return True

    def synchronize(self):
        return None

    def elapsed_time(self, other):
        return 1.25


class TestCudaEventProfiler(unittest.TestCase):
    def test_disabled_profiler_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = CudaEventProfiler(enable=False, directory=tmpdir, rank=0)
            with profiler.record("training_step", step=0):
                pass
            profiler.close()
            self.assertEqual(list(Path(tmpdir).glob("*.jsonl")), [])

    def test_writes_completed_cuda_events_as_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.Event", side_effect=lambda enable_timing: FakeCudaEvent()
            ):
                profiler = CudaEventProfiler(
                    enable=True,
                    directory=tmpdir,
                    rank=3,
                    profiler_config={"record_every_n_steps": 1},
                )
                with profiler.record("training_step", step=7, micro_step=2):
                    pass
                profiler.flush()
                profiler.close()

            output_file = Path(tmpdir) / "cuda_events_rank_3.jsonl"
            records = [json.loads(line) for line in output_file.read_text().splitlines()]
            self.assertEqual(
                records,
                [
                    {
                        "duration_ms": 1.25,
                        "micro_step": 2,
                        "name": "training_step",
                        "rank": 3,
                        "step": 7,
                    }
                ],
            )

    def test_respects_recording_window_and_stride(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.Event", side_effect=lambda enable_timing: FakeCudaEvent()
            ):
                profiler = CudaEventProfiler(
                    enable=True,
                    directory=tmpdir,
                    rank=0,
                    profiler_config={"start_step": 2, "end_step": 5, "record_every_n_steps": 2},
                )
                for step in range(7):
                    with profiler.record("training_step", step=step):
                        pass
                profiler.close()

            output_file = Path(tmpdir) / "cuda_events_rank_0.jsonl"
            records = [json.loads(line) for line in output_file.read_text().splitlines()]
            self.assertEqual([record["step"] for record in records], [2, 4])

    def test_defaults_to_sample_every_ten_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.Event", side_effect=lambda enable_timing: FakeCudaEvent()
            ):
                profiler = CudaEventProfiler(enable=True, directory=tmpdir, rank=0)
                for step in range(21):
                    with profiler.record("training_step", step=step):
                        pass
                profiler.close()

            output_file = Path(tmpdir) / "cuda_events_rank_0.jsonl"
            records = [json.loads(line) for line in output_file.read_text().splitlines()]
            self.assertEqual([record["step"] for record in records], [0, 10, 20])

    def test_rank_filter_skips_unselected_ranks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.Event", side_effect=lambda enable_timing: FakeCudaEvent()
            ):
                profiler = CudaEventProfiler(
                    enable=True,
                    directory=tmpdir,
                    rank=3,
                    profiler_config={"ranks": [0, 1]},
                )
                with profiler.record("training_step", step=0):
                    pass
                profiler.close()

            self.assertEqual(list(Path(tmpdir).glob("*.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
