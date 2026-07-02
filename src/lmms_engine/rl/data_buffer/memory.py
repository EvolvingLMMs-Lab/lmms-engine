from __future__ import annotations

import asyncio
import uuid
from collections import deque
from typing import Deque

from lmms_engine.rl.config import DataBufferConfig
from lmms_engine.rl.core.interfaces import DataBuffer
from lmms_engine.rl.protocol import BufferStats, RewardedTrajectory, TrainBatch


class InMemoryDataBuffer(DataBuffer):
    """Small independent producer-consumer buffer for the RL MVP.

    This is intentionally process-local scaffolding. A Ray actor wrapper or a
    durable queue can later use the same push/pop/stats contract.
    """

    def __init__(self, config: DataBufferConfig | None = None) -> None:
        self.config = config or DataBufferConfig()
        self._queue: Deque[RewardedTrajectory] = deque()
        self._pending_steps = 0
        self._paused = False
        self._lock = asyncio.Lock()

    def push(self, trajectory: RewardedTrajectory) -> BufferStats:
        if len(self._queue) >= self.config.max_trajectories:
            raise BufferError("DataBuffer is full")
        self._queue.append(trajectory)
        self._pending_steps += len(trajectory.steps)
        self._refresh_pause_state()
        return self.stats()

    async def push_async(self, trajectory: RewardedTrajectory) -> BufferStats:
        async with self._lock:
            return self.push(trajectory)

    def pop_train_batch(self) -> TrainBatch | None:
        if len(self._queue) < self.config.min_trajectories_per_batch:
            return None

        trajectories: list[RewardedTrajectory] = []
        while self._queue and len(trajectories) < self.config.train_batch_size:
            trajectory = self._queue.popleft()
            trajectories.append(trajectory)
            self._pending_steps -= len(trajectory.steps)

        self._refresh_pause_state()
        model_version = trajectories[0].model_version if trajectories else None
        return TrainBatch(
            batch_id=str(uuid.uuid4()),
            model_version=model_version,
            trajectories=trajectories,
            global_batch_size=self.config.train_batch_size,
            metadata={"source": "in_memory_data_buffer"},
        )

    async def pop_train_batch_async(self) -> TrainBatch | None:
        async with self._lock:
            return self.pop_train_batch()

    def should_pause_rollout(self) -> bool:
        return self._paused

    def stats(self) -> BufferStats:
        return BufferStats(
            pending_trajectories=len(self._queue),
            pending_steps=self._pending_steps,
            high_watermark=self.config.high_watermark,
            low_watermark=self.config.low_watermark,
            paused=self._paused,
        )

    def _refresh_pause_state(self) -> None:
        if len(self._queue) >= self.config.high_watermark:
            self._paused = True
        elif len(self._queue) <= self.config.low_watermark:
            self._paused = False
