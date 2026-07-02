from __future__ import annotations

from typing import Any

from lmms_engine.rl.core.interfaces import TrajectoryAdapter
from lmms_engine.rl.protocol import (
    ModelVersion,
    RewardedTrajectory,
    RolloutTask,
    TrajectoryStep,
)


class LMMSEvalTrajectoryAdapter(TrajectoryAdapter):
    """Default adapter from lmms-eval EpisodeResult to RewardedTrajectory."""

    def from_episode(self, task: RolloutTask, episode: Any) -> RewardedTrajectory:
        return trajectory_from_lmms_eval_episode(task, episode)


def trajectory_from_lmms_eval_episode(task: RolloutTask, episode: Any) -> RewardedTrajectory:
    """Convert an lmms-eval EpisodeResult into engine-side RL data.

    lmms-eval remains the owner of env/session semantics. This adapter only
    normalizes the payload for buffering and training.
    """

    steps = []
    for step in getattr(episode, "steps", []) or []:
        result = getattr(step, "result", None)
        parsed_action = getattr(step, "parsed_action", None)
        state = getattr(step, "state", None)
        steps.append(
            TrajectoryStep(
                observation=getattr(state, "observation", None),
                request=getattr(step, "request", None),
                response=getattr(step, "raw_output", None),
                action=getattr(parsed_action, "action", None),
                reward=getattr(result, "reward", None),
                done=bool(getattr(result, "done", False)) if result is not None else False,
                info=dict(getattr(result, "info", {}) or {}),
                metadata={
                    "step_idx": getattr(state, "step_idx", None),
                    "parse_error": getattr(parsed_action, "error", None),
                },
            )
        )

    model_version = getattr(task, "model_version", None)
    if model_version is None:
        model_version = ModelVersion(version_id=0)

    return RewardedTrajectory(
        trajectory_id=f"{task.task_id}:{len(steps)}",
        task_id=task.task_id,
        model_version=model_version,
        steps=steps,
        metrics=dict(getattr(episode, "metrics", {}) or {}),
        final_state=getattr(episode, "final_state", None),
        metadata={
            "lmms_eval_metadata": dict(getattr(episode, "metadata", {}) or {}),
            "success": getattr(episode, "success", None),
            **task.metadata,
        },
    )
