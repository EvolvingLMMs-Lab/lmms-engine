import math
import os
from dataclasses import dataclass


@dataclass
class MegatronConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
    seq_length: int = 4096
    online_packing: bool = True
    attention_backend: str = "auto"
    recompute_granularity: str | None = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    gpu_peak_tflops: float | None = None

    def validate(self):
        if self.attention_backend not in {"auto", "flash", "fused", "unfused", "local"}:
            raise ValueError("attention_backend must be auto, flash, fused, unfused, or local")
        if self.gpu_peak_tflops is not None and (not math.isfinite(self.gpu_peak_tflops) or self.gpu_peak_tflops <= 0):
            raise ValueError("gpu_peak_tflops must be a finite positive dense BF16 peak per GPU")
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "seq_length",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if int(os.environ.get("WORLD_SIZE", "1")) % self.model_parallel_size:
            raise ValueError("WORLD_SIZE must be divisible by TP * PP * CP")
        if self.sequence_parallel and self.tensor_model_parallel_size == 1:
            raise ValueError("sequence_parallel requires TP > 1")
        if self.seq_length % self.alignment:
            raise ValueError(f"seq_length must be divisible by {self.alignment}")
        if self.recompute_granularity not in {None, "full", "selective"}:
            raise ValueError("recompute_granularity must be null, full, or selective")
        if self.recompute_method not in {"uniform", "block"} or self.recompute_num_layers < 1:
            raise ValueError("Use a positive recompute_num_layers and uniform or block recompute_method")

    @property
    def model_parallel_size(self):
        return self.tensor_model_parallel_size * self.pipeline_model_parallel_size * self.context_parallel_size

    @property
    def alignment(self):
        cp = self.context_parallel_size
        return math.lcm(2 * cp if cp > 1 else 1, self.tensor_model_parallel_size * cp if self.sequence_parallel else 1)
