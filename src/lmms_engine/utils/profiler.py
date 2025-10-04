import os
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Optional

import torch
from loguru import logger
from torch import profiler as torch_profiler


class StepProfiler:
    def __init__(
        self,
        enable: bool,
        directory: str,
        rank: int = 0,
        profiler_config: Optional[Dict[str, Any]] = None,
    ):
        self.enable = enable
        if not self.enable:
            self.prof = None
            self.skip_prof = True
            self.rank = rank
            return
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        self.activities = activities
        self.profiler_config = profiler_config or {}
        # Default to profile 10 steps from start to end
        self.start_step = self.profiler_config.get("start_step", 0)
        self.end_step = self.profiler_config.get("end_step", 5)
        self.prof = torch_profiler.profile(
            activities=activities,
            schedule=torch_profiler.schedule(
                wait=self.start_step, warmup=1, active=self.end_step - self.start_step
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.rank = rank
        self.skip_prof = False

    def check(self):
        return self.prof is not None and not self.skip_prof

    def start(self):
        if self.check():
            logger.info(f"[Profiler] started for rank {self.rank}")
            self.prof.start()

    def step(self):
        if self.check():
            self.prof.step()

    def stop(self):
        if self.check():
            logger.info(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()

    def save(self):
        if self.prof is not None:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            save_file_name = f"/prof_start_{self.start_step}_end_{self.end_step}_rank_{self.rank}.json"
            logger.info(f"[Profiler] Saving trace to {self.directory + save_file_name}")
            self.prof.export_chrome_trace(self.directory + save_file_name)
            self.skip_prof = True

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()

    def should_save(self, step: int):
        if self.check():
            return step >= self.start_step and step == self.end_step
        else:
            return False

    def stop_trace(self):
        if self.check():
            logger.info(f"[Profiler] Trace stopped for rank {self.rank}")
            self.skip_prof = True
