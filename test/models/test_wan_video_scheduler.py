import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Load the module by path so the test does not need the full model stack.
_SCHEDULER_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "lmms_engine" / "models" / "wanvideo" / "wan_video_scheduler.py"
)
_spec = importlib.util.spec_from_file_location("wan_video_scheduler_under_test", _SCHEDULER_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
FlowMatchScheduler = _module.FlowMatchScheduler


def test_exponential_shift_with_fixed_mu_uses_math():
    mu = 0.5
    scheduler = FlowMatchScheduler(num_inference_steps=10, exponential_shift=True, exponential_shift_mu=mu)
    assert scheduler.sigmas.shape == (10,)
    assert torch.isfinite(scheduler.sigmas).all()

    base = torch.linspace(1.0, scheduler.sigma_min, 10)
    expected = torch.exp(torch.tensor(mu)) / (torch.exp(torch.tensor(mu)) + (1 / base - 1))
    torch.testing.assert_close(scheduler.sigmas, expected, rtol=1e-4, atol=1e-6)


def test_exponential_shift_with_dynamic_shift_len():
    scheduler = FlowMatchScheduler(num_inference_steps=5, exponential_shift=True, exponential_shift_mu=0.0)
    scheduler.set_timesteps(5, dynamic_shift_len=1024)
    assert torch.isfinite(scheduler.sigmas).all()
