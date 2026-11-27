import unittest

from lmms_engine.rl.config import DataBufferConfig, RLRunConfig
from lmms_engine.rl.core.interfaces import (
    BatchBuilder,
    RolloutManager,
    WeightSyncClient,
)
from lmms_engine.rl.core.orchestrator import RLOrchestrator
from lmms_engine.rl.data_buffer import InMemoryDataBuffer
from lmms_engine.rl.protocol import (
    ModelVersion,
    RewardedTrajectory,
    RolloutTask,
    TrainBatch,
    TrajectoryStep,
)
from lmms_engine.rl.trajectory_annotation import ReferenceLogprobAnnotator


class FakeScorer:
    def __init__(self):
        self.calls = []

    def score_logprobs(self, requests, responses):
        self.calls.append((list(requests), list(responses)))
        return [
            {
                "mean_logprob": -0.25 - index,
                "sequence_logprob": -0.5 - index,
                "num_tokens": 2,
            }
            for index, _request in enumerate(requests)
        ]


class FakeRolloutManager(RolloutManager):
    def __init__(self, completed):
        self.completed = list(completed)

    def start(self):
        return None

    def pause(self):
        return None

    def resume(self):
        return None

    def submit(self, task):
        return True

    def poll_completed(self, timeout_s=None, max_trajectories=None):
        completed = self.completed
        self.completed = []
        return completed

    @property
    def inflight(self):
        return len(self.completed)


class FakeBatchBuilder(BatchBuilder):
    def build(self, trajectories, model_version):
        return TrainBatch(
            batch_id="batch",
            model_version=model_version,
            trajectories=list(trajectories),
            global_batch_size=len(trajectories),
        )


class FakeWeightSync(WeightSyncClient):
    def reload_weights(self, model_version):
        return {}


def _trajectory(num_steps=2):
    return RewardedTrajectory(
        trajectory_id="trajectory",
        task_id="task",
        model_version=ModelVersion(version_id=0),
        steps=[TrajectoryStep(request=f"prompt-{index}", response=f"response-{index}") for index in range(num_steps)],
    )


class TestTrajectoryAnnotation(unittest.TestCase):
    def test_reference_logprob_annotator_writes_step_metadata(self):
        scorer = FakeScorer()
        trajectory = _trajectory()
        annotator = ReferenceLogprobAnnotator(scorer, max_batch_size=8)

        result = annotator.annotate([trajectory])

        self.assertIs(result[0], trajectory)
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(trajectory.steps[0].metadata["logprobs"]["reference"]["mean_logprob"], -0.25)
        self.assertEqual(trajectory.steps[0].metadata["reference_logprob_mean"], -0.25)
        self.assertEqual(trajectory.steps[1].metadata["logprobs"]["reference"]["mean_logprob"], -1.25)

    def test_reference_logprob_annotator_can_score_chunks_concurrently(self):
        scorer = FakeScorer()
        trajectory = _trajectory(num_steps=4)
        annotator = ReferenceLogprobAnnotator(scorer, max_batch_size=2, max_workers=2)

        annotator.annotate([trajectory])

        self.assertEqual(len(scorer.calls), 2)
        self.assertTrue(all(len(requests) == 2 for requests, _responses in scorer.calls))
        self.assertIn("reference_logprob_mean", trajectory.steps[3].metadata)

    def test_orchestrator_annotates_before_buffer_push(self):
        scorer = FakeScorer()
        config = RLRunConfig(
            data_buffer=DataBufferConfig(
                max_trajectories=4,
                high_watermark=4,
                low_watermark=0,
                global_train_batch_size=1,
                min_trajectories_per_batch=1,
            )
        )
        orchestrator = RLOrchestrator(
            config=config,
            rollout_manager=FakeRolloutManager([_trajectory()]),
            data_buffer=InMemoryDataBuffer(config.data_buffer),
            batch_builder=FakeBatchBuilder(),
            trajectory_annotator=ReferenceLogprobAnnotator(scorer),
            weight_sync=FakeWeightSync(),
        )

        self.assertEqual(orchestrator.drain_rollouts(timeout_s=0.0), 1)
        batch = orchestrator.next_train_batch()

        self.assertIsNotNone(batch)
        step = batch.trajectories[0].steps[0]
        self.assertEqual(step.metadata["reference_logprob_mean"], -0.25)


if __name__ == "__main__":
    unittest.main()
