import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

from lmms_engine.utils.train_utils import TrainUtilities


@dataclass
class ComputeSummary:
    total_flops: float
    total_flops_formatted: str
    training_duration_seconds: float
    training_duration_formatted: str
    gpu_name: str
    gpu_tdp_watts: float
    num_gpus: int
    energy_kwh: float
    carbon_intensity_kg_per_kwh: float
    co2_kg: float
    co2_formatted: str


class ComputeTracker:
    """Accumulates raw FLOPS across an entire training run, tracks wall-clock
    duration, and estimates energy consumption and CO2 emissions.

    Usage:
        tracker = ComputeTracker(num_gpus=8, carbon_intensity=0.475,
                                 gpu_tdp_watts=700.0, gpu_name="NVIDIA H100")
        tracker.start()
        # ... in training loop ...
        tracker.accumulate_flops(raw_flops)
        # ... at end of training ...
        summary = tracker.finish()
        tracker.save_summary(output_dir, summary)
    """

    def __init__(
        self,
        num_gpus: int = 1,
        carbon_intensity: float = 0.475,
        gpu_tdp_watts: float = 0.0,
        gpu_name: str = "unknown",
    ):
        self.num_gpus = num_gpus
        self.carbon_intensity = carbon_intensity
        self.gpu_tdp_watts = gpu_tdp_watts
        self.gpu_name = gpu_name
        self._total_flops: float = 0.0
        self._start_time: Optional[float] = None

    def start(self):
        """Record the wall-clock start time."""
        self._start_time = time.time()

    def accumulate_flops(self, raw_flops: float):
        """Add raw FLOPS from one training step (per-rank)."""
        self._total_flops += raw_flops

    def state_dict(self) -> dict:
        """Return checkpoint-friendly state."""
        return {
            "total_flops": self._total_flops,
            "start_time": self._start_time,
        }

    def load_state_dict(self, state: dict):
        """Restore from checkpoint."""
        self._total_flops = state.get("total_flops", 0.0)
        self._start_time = state.get("start_time", self._start_time)

    def finish(self) -> ComputeSummary:
        """Compute the final summary.  Call once at the end of training."""
        end_time = time.time()
        duration_s = end_time - self._start_time if self._start_time is not None else 0.0
        training_hours = duration_s / 3600.0

        # Energy = num_gpus * TDP_watts * hours / 1000  (kWh)
        energy_kwh = self.num_gpus * self.gpu_tdp_watts * training_hours / 1000.0

        # CO2 = energy * carbon intensity
        co2_kg = energy_kwh * self.carbon_intensity

        return ComputeSummary(
            total_flops=self._total_flops,
            total_flops_formatted=TrainUtilities.format_flops(self._total_flops),
            training_duration_seconds=round(duration_s, 1),
            training_duration_formatted=self._format_duration(duration_s),
            gpu_name=self.gpu_name,
            gpu_tdp_watts=self.gpu_tdp_watts,
            num_gpus=self.num_gpus,
            energy_kwh=round(energy_kwh, 4),
            carbon_intensity_kg_per_kwh=self.carbon_intensity,
            co2_kg=round(co2_kg, 4),
            co2_formatted=self._format_co2(co2_kg),
        )

    @staticmethod
    def _format_co2(co2_kg: float) -> str:
        """Auto-scale CO2 to g / kg / t."""
        if co2_kg <= 0:
            return "0 g CO2"
        if co2_kg >= 1000:
            return f"{co2_kg / 1000:.2f} t CO2"
        if co2_kg >= 1:
            return f"{co2_kg:.2f} kg CO2"
        return f"{co2_kg * 1000:.2f} g CO2"

    @staticmethod
    def save_summary(output_dir: str, summary: ComputeSummary):
        """Write compute_summary.json to *output_dir*."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "compute_summary.json")
        with open(path, "w") as f:
            json.dump(asdict(summary), f, indent=2)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        parts = []
        if h > 0:
            parts.append(f"{h}h")
        if m > 0:
            parts.append(f"{m}m")
        parts.append(f"{s}s")
        return " ".join(parts)
