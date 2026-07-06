import unittest

from lmms_engine.rl.config import DataBufferConfig, RLRunConfig
from lmms_engine.rl.core.interfaces import BatchBuilder, RolloutManager, WeightSyncClient
from lmms_engine.rl.core.orchestrator import RLOrchestrator
from lmms_engine.rl.data_buffer import InMemoryDataBuffer
from lmms_engine.rl.protocol import ModelVersion, RewardedTrajectory, RolloutTask, TrainBatch


class FakeRolloutManager(RolloutManager):
    def __init__(self, completed: list[RewardedTrajectory]) -> None:
        self.completed = completed
        self.started = False
        self.paused = False
        self.last_max_trajectories = None

    def start(self) -> None:
        self.started = True

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def submit(self, task: RolloutTask) -> bool:
        return not self.paused

    def poll_completed(
        self,
        timeout_s: float | None = None,
        max_trajectories: int | None = None,
    ) -> list[RewardedTrajectory]:
        self.last_max_trajectories = max_trajectories
        completed = self.completed
        self.completed = []
        return completed

    @property
    def inflight(self) -> int:
        return len(self.completed)


class FakeBatchBuilder(BatchBuilder):
    def build(self, trajectories, model_version):
        return TrainBatch(
            batch_id="fake",
            model_version=model_version,
            trajectories=list(trajectories),
            global_batch_size=len(trajectories),
        )


class FakeWeightSync(WeightSyncClient):
    def reload_weights(self, model_version: ModelVersion) -> dict:
        return {}


def _trajectory(index: int) -> RewardedTrajectory:
    return RewardedTrajectory(
        trajectory_id=f"trajectory-{index}",
        task_id=f"task-{index}",
        model_version=ModelVersion(version_id=0),
    )


class TestRLOrchestratorBackpressure(unittest.TestCase):
    def test_completed_rollouts_backpressure_without_buffer_error(self):
        data_buffer_config = DataBufferConfig(
            max_trajectories=2,
            high_watermark=2,
            low_watermark=0,
            global_train_batch_size=1,
            min_trajectories_per_batch=1,
        )
        config = RLRunConfig(data_buffer=data_buffer_config)
        rollout_manager = FakeRolloutManager([_trajectory(0), _trajectory(1), _trajectory(2)])
        orchestrator = RLOrchestrator(
            config=config,
            rollout_manager=rollout_manager,
            data_buffer=InMemoryDataBuffer(data_buffer_config),
            batch_builder=FakeBatchBuilder(),
            weight_sync=FakeWeightSync(),
        )

        accepted = orchestrator.drain_rollouts(timeout_s=0.0)

        self.assertEqual(accepted, 2)
        self.assertEqual(rollout_manager.last_max_trajectories, 2)
        self.assertEqual(orchestrator.data_buffer.stats().pending_trajectories, 2)
        self.assertTrue(rollout_manager.paused)
        self.assertFalse(
            orchestrator.submit_rollout(
                RolloutTask(
                    task_id="blocked",
                    payload=None,
                    model_version=ModelVersion(version_id=0),
                )
            )
        )

        self.assertIsNotNone(orchestrator.next_train_batch())
        self.assertEqual(orchestrator.drain_rollouts(timeout_s=0.0), 1)
        self.assertEqual(orchestrator.data_buffer.stats().pending_trajectories, 2)

        self.assertIsNotNone(orchestrator.next_train_batch())
        self.assertIsNotNone(orchestrator.next_train_batch())
        orchestrator.drain_rollouts(timeout_s=0.0)

        self.assertEqual(orchestrator.data_buffer.stats().pending_trajectories, 0)
        self.assertFalse(rollout_manager.paused)


if __name__ == "__main__":
    unittest.main()
