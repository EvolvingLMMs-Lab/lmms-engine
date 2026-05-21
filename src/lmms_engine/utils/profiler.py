import os
import time
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
            schedule=torch_profiler.schedule(wait=self.start_step, warmup=1, active=self.end_step - self.start_step),
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


class MemorySnapshotProfiler:
    """CUDA memory snapshot profiler with automatic OOM capture.

    When enabled, records every CUDA alloc/free event (with Python stack
    traces) into an in-memory ring buffer via
    ``torch.cuda.memory._record_memory_history``. On a ``CUDA OOM``, an
    out-of-memory observer dumps the buffer to a ``.pickle`` file that can
    be loaded into https://pytorch.org/memory_viz for visualization.

    Independent of ``StepProfiler`` — both can be enabled together.

    Config keys (under ``memory_snapshot_config``):
        - ``max_entries`` (int, default 100000): ring buffer size. One alloc
          or free event = one entry. 100k covers ~a few training steps.
        - ``stop_step`` (int, optional): if set, stop recording and dump a
          final snapshot at this global step (useful for inspecting steady
          state without OOM).
    """

    def __init__(
        self,
        enable: bool,
        directory: str,
        rank: int = 0,
        memory_snapshot_config: Optional[Dict[str, Any]] = None,
    ):
        self.enable = enable and torch.cuda.is_available()
        self.rank = rank
        self.directory = directory
        self.config = memory_snapshot_config or {}
        self.max_entries = int(self.config.get("max_entries", 100000))
        self.stop_step = self.config.get("stop_step", None)
        self.started = False
        self.stopped = False

    def _dump(self, filename: str, force: bool = False):
        if not self.enable or (self.stopped and not force):
            return
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, filename)
        try:
            torch.cuda.memory._dump_snapshot(path)
            logger.info(f"[MemSnapshot] dumped snapshot to {path} (rank {self.rank})")
        except Exception:
            logger.exception(f"[MemSnapshot] failed to dump snapshot to {path}")

    def dump_on_exception(self, reason: str):
        timestamp = int(time.time())
        self._dump(f"snapshot_{reason}_rank{self.rank}_pid{os.getpid()}_{timestamp}.pickle", force=True)

    def _oom_observer(self, device, alloc, device_alloc, device_free):
        # Called by PyTorch BEFORE raising CUDA OOM. Dump current snapshot.
        logger.error(
            f"[MemSnapshot] CUDA OOM on rank {self.rank} device {device}: "
            f"attempted to alloc {alloc} bytes "
            f"(device_alloc={device_alloc}, device_free={device_free})"
        )
        self.dump_on_exception("oom_observer")
        # Mark stopped so we don't try to dump again on re-raise paths.
        self.stopped = True

    def start(self):
        if not self.enable or self.started:
            return
        os.makedirs(self.directory, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
        try:
            torch._C._cuda_attach_out_of_memory_observer(self._oom_observer)
        except AttributeError:
            logger.warning(
                "[MemSnapshot] OOM observer API not available in this torch version; "
                "snapshot will only dump on explicit stop_and_save()."
            )
        self.started = True
        logger.info(
            f"[MemSnapshot] recording started on rank {self.rank} "
            f"(max_entries={self.max_entries}, dir={self.directory})"
        )

    def step(self, global_step: int):
        """Mark step boundary in the snapshot timeline; optionally auto-stop."""
        if not self.enable or self.stopped:
            return
        # NVTX marker → shows up as a vertical line in memory_viz timeline.
        torch.cuda.nvtx.range_push(f"step_{global_step}")
        torch.cuda.nvtx.range_pop()
        if self.stop_step is not None and global_step >= self.stop_step:
            self.stop_and_save(reason="stop_step")

    def stop_and_save(self, reason: str = "manual"):
        if not self.enable or self.stopped:
            return
        self._dump(f"snapshot_{reason}_rank{self.rank}.pickle")
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception:
            pass
        self.stopped = True
        logger.info(f"[MemSnapshot] recording stopped (reason={reason}, rank {self.rank})")
