from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from lmms_engine.rl.core.interfaces import TrajectoryAnnotator
from lmms_engine.rl.protocol import RewardedTrajectory, TrajectoryStep


@dataclass(slots=True)
class ReferenceLogprobAnnotator(TrajectoryAnnotator):
    """Annotate trajectories with reference-model response logprobs."""

    model_server: Any
    role: str = "reference"
    max_batch_size: int = 8
    max_workers: int = 1

    def annotate(self, trajectories: list[RewardedTrajectory]) -> list[RewardedTrajectory]:
        items = [
            (trajectory, step)
            for trajectory in trajectories
            for step in trajectory.steps
            if _has_scoreable_pair(step)
        ]
        if not items:
            return trajectories
        score_method = getattr(self.model_server, "score_logprobs", None)
        if score_method is None:
            raise NotImplementedError(
                f"Reference model server {type(self.model_server).__name__} does not implement score_logprobs()."
            )

        batch_size = max(1, int(self.max_batch_size))
        chunks = [items[start : start + batch_size] for start in range(0, len(items), batch_size)]

        def score_chunk(chunk: list[tuple[RewardedTrajectory, TrajectoryStep]]) -> list[dict[str, Any]]:
            requests = [step.request for _trajectory, step in chunk]
            responses = [step.response for _trajectory, step in chunk]
            scores = score_method(requests, responses)
            if len(scores) != len(chunk):
                raise RuntimeError(f"Reference scorer returned {len(scores)} scores for {len(chunk)} steps.")
            return [dict(score) for score in scores]

        workers = min(max(1, int(self.max_workers)), len(chunks))
        if workers == 1:
            chunk_scores = [score_chunk(chunk) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                chunk_scores = list(executor.map(score_chunk, chunks))

        for chunk, scores in zip(chunks, chunk_scores, strict=True):
            for (_trajectory, step), score in zip(chunk, scores, strict=True):
                _write_reference_score(step, self.role, score)
        return trajectories


def _has_scoreable_pair(step: TrajectoryStep) -> bool:
    return step.request is not None and step.response is not None and _first_text(step.response) not in (None, "")


def _write_reference_score(step: TrajectoryStep, role: str, score: dict[str, Any]) -> None:
    metadata = step.metadata
    logprobs = metadata.setdefault("logprobs", {})
    if not isinstance(logprobs, dict):
        raise TypeError(f"TrajectoryStep.metadata['logprobs'] must be a dict, got {type(logprobs).__name__}.")
    logprobs[role] = score
    if score.get("mean_logprob") is not None:
        metadata[f"{role}_logprob_mean"] = float(score["mean_logprob"])


def _first_text(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "first_text"):
        text = value.first_text()
        return None if text is None else str(text)
    if isinstance(value, str):
        return value
    return str(value)
