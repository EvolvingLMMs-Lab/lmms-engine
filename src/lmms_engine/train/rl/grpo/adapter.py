from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lmms_engine.datasets.collator import VisionCollator
from lmms_engine.rl.core.interfaces import TrainBatchAdapter
from lmms_engine.rl.protocol import RewardedTrajectory, TrainBatch, TrajectoryStep


@dataclass(slots=True)
class GRPOConfig:
    """GRPO trainer-side adapter knobs.

    This is intentionally only the boundary config. Tensorization, advantage
    computation, old-logprob plumbing, and model-specific loss details should be
    implemented behind `GRPOBatchAdapter`.
    """

    group_size: int = 1
    beta: float = 0.0
    clip_range_low: float = 0.2
    clip_range_high: float = 0.2
    normalize_advantages: bool = True
    advantage_clip: float | None = None
    advantage_epsilon: float = 1.0e-6
    reward_key: str = "total_reward"
    fallback_to_total_reward: bool = True
    use_step_rewards: bool = False
    system_message: str = "You are a helpful agent."
    add_system_prompt: bool = True
    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    max_steps_per_trajectory: int | None = None
    step_sampling_strategy: str = "uniform"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GRPOPayload:
    """Algorithm-specific payload passed from TrainBatchAdapter to trainer."""

    trajectories: list[RewardedTrajectory]
    config: GRPOConfig
    tensors: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GRPOBatchAdapter(TrainBatchAdapter):
    """Convert engine TrainBatch into a GRPO trainer payload."""

    def __init__(self, config: GRPOConfig | None = None, processor: Any | None = None) -> None:
        self.config = config or GRPOConfig()
        self.processor = processor
        self.collator = VisionCollator(processor) if processor is not None else None

    def to_trainer_batch(self, batch: TrainBatch) -> GRPOPayload:
        tensors = self._tensorize(batch) if self.processor is not None else None
        return GRPOPayload(
            trajectories=batch.trajectories,
            config=self.config,
            tensors=tensors,
            metadata={
                **batch.metadata,
                "algorithm": "grpo",
                "batch_id": batch.batch_id,
                "model_version": batch.model_version.version_id if batch.model_version else None,
            },
        )

    def _tensorize(self, batch: TrainBatch) -> dict[str, Any]:
        samples: list[dict[str, Any]] = []
        rewards: list[float] = []
        sample_metadata: list[dict[str, Any]] = []

        for trajectory in batch.trajectories:
            trajectory_reward = self._trajectory_reward(trajectory)
            for step_idx, step in self._selected_steps(trajectory):
                sample = self._step_to_processor_sample(step)
                if sample is None:
                    continue
                samples.append(sample)
                rewards.append(self._step_reward(step, trajectory_reward))
                ref_logprob = _step_mean_logprob(step, "reference")
                if ref_logprob is not None:
                    sample["ref_logprobs"] = torch.tensor([ref_logprob], dtype=torch.float32)
                old_logprob = _step_mean_logprob(step, "policy_old")
                if old_logprob is not None:
                    sample["old_logprobs"] = torch.tensor([old_logprob], dtype=torch.float32)
                sample_metadata.append(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "task_id": trajectory.task_id,
                        "step_idx": step_idx,
                    }
                )

        if not samples:
            raise ValueError("GRPOBatchAdapter could not build any train samples from the rollout trajectories.")

        advantages = self._advantages(rewards)
        for sample, reward, advantage in zip(samples, rewards, advantages, strict=True):
            sample["rewards"] = torch.tensor([reward], dtype=torch.float32)
            sample["advantages"] = torch.tensor([advantage], dtype=torch.float32)

        tensors = self.collator(samples)
        tensors["sample_rewards"] = tensors.pop("rewards")
        tensors["sample_advantages"] = tensors.pop("advantages")
        if "ref_logprobs" in tensors:
            tensors["sample_ref_logprobs"] = tensors.pop("ref_logprobs")
        if "old_logprobs" in tensors:
            tensors["sample_old_logprobs"] = tensors.pop("old_logprobs")
        tensors["sample_metadata"] = sample_metadata
        return tensors

    def _step_to_processor_sample(self, step: TrajectoryStep) -> dict[str, Any] | None:
        request = step.request
        response_text = _first_text(step.response)
        if request is None or response_text is None or response_text == "":
            return None

        user_content, images, videos = _agent_input_to_hf_content(request)
        if not user_content:
            return None

        request_metadata = getattr(request, "metadata", None) or {}
        hf_messages, history_images, history_videos = _history_to_hf_messages(
            request_metadata.get("conversation_history")
        )
        hf_messages.extend(
            [
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response_text}],
                },
            ]
        )
        return self.processor.process(
            images=[*history_images, *images] or None,
            hf_messages=hf_messages,
            videos=[*history_videos, *videos] or None,
            system_message=self.config.system_message,
            add_system_prompt=self.config.add_system_prompt,
            **self.config.processor_kwargs,
        )

    def _selected_steps(self, trajectory: RewardedTrajectory) -> list[tuple[int, TrajectoryStep]]:
        indexed_steps = list(enumerate(trajectory.steps))
        limit = self.config.max_steps_per_trajectory
        if limit is None or int(limit) <= 0 or len(indexed_steps) <= int(limit):
            return indexed_steps

        limit = int(limit)
        strategy = self.config.step_sampling_strategy.lower()
        if strategy == "first":
            return indexed_steps[:limit]
        if strategy == "last":
            return indexed_steps[-limit:]
        if strategy != "uniform":
            raise ValueError(
                "GRPOConfig.step_sampling_strategy must be one of "
                f"'uniform', 'first', or 'last', got {self.config.step_sampling_strategy!r}."
            )

        if limit == 1:
            return [indexed_steps[-1]]
        last = len(indexed_steps) - 1
        indices = [round(position * last / (limit - 1)) for position in range(limit)]
        return [indexed_steps[index] for index in indices]

    def _trajectory_reward(self, trajectory: RewardedTrajectory) -> float:
        if self.config.reward_key in trajectory.metrics:
            return float(trajectory.metrics[self.config.reward_key])
        if self.config.fallback_to_total_reward:
            return float(trajectory.total_reward)
        return 0.0

    def _step_reward(self, step: TrajectoryStep, trajectory_reward: float) -> float:
        if self.config.use_step_rewards and isinstance(step.reward, (float, int)):
            return float(step.reward)
        return trajectory_reward

    def _advantages(self, rewards: list[float]) -> list[float]:
        values = torch.tensor(rewards, dtype=torch.float32)
        if self.config.normalize_advantages and len(rewards) > 1:
            mean = values.mean()
            std = values.std(unbiased=False)
            if float(std) > self.config.advantage_epsilon:
                values = (values - mean) / (std + self.config.advantage_epsilon)
            else:
                values = values
        if self.config.advantage_clip is not None:
            clip = float(self.config.advantage_clip)
            values = values.clamp(min=-clip, max=clip)
        return [float(item) for item in values.tolist()]


def _first_text(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "first_text"):
        text = value.first_text()
        return None if text is None else str(text)
    if isinstance(value, str):
        return value
    return str(value)


def _agent_input_to_hf_content(
    request: Any,
) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
    return _content_blocks_to_hf_content(getattr(request, "content", []) or [])


def _content_blocks_to_hf_content(
    blocks: Any,
) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
    content: list[dict[str, Any]] = []
    images: list[Any] = []
    videos: list[Any] = []

    for block in blocks or []:
        block_type = getattr(block, "type", None)
        data = getattr(block, "data", None)
        if block_type == "text" and data is not None:
            content.append({"type": "text", "text": str(data)})
        elif block_type in {"image", "image_url"} and data is not None:
            images.append(_media_payload(data, "image_url"))
            content.append({"type": "image"})
        elif block_type in {"video", "video_url"} and data is not None:
            videos.append(_media_payload(data, "video_url"))
            content.append({"type": "video"})

    return content, images, videos


def _history_to_hf_messages(
    history: Any,
) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
    messages: list[dict[str, Any]] = []
    images: list[Any] = []
    videos: list[Any] = []
    if not isinstance(history, list):
        return messages, images, videos

    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "user"))
        turn_content = turn.get("content", "")
        if role == "assistant":
            messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": _history_text(turn_content)}],
                }
            )
            continue
        if hasattr(turn_content, "content"):
            content, turn_images, turn_videos = _content_blocks_to_hf_content(turn_content.content)
        elif _is_content_block_list(turn_content):
            content, turn_images, turn_videos = _content_blocks_to_hf_content(turn_content)
        elif isinstance(turn_content, str):
            content, turn_images, turn_videos = (
                [{"type": "text", "text": turn_content}],
                [],
                [],
            )
        elif isinstance(turn_content, list):
            content, turn_images, turn_videos = turn_content, [], []
        else:
            content, turn_images, turn_videos = (
                [{"type": "text", "text": str(turn_content)}],
                [],
                [],
            )
        messages.append({"role": role, "content": content})
        images.extend(turn_images)
        videos.extend(turn_videos)

    return messages, images, videos


def _is_content_block_list(value: Any) -> bool:
    return isinstance(value, list) and all(hasattr(item, "type") and hasattr(item, "data") for item in value)


def _history_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if hasattr(content, "first_text"):
        text = content.first_text()
        return "" if text is None else str(text)
    if isinstance(content, list):
        parts = []
        for item in content:
            if hasattr(item, "type") and getattr(item, "type") == "text" and getattr(item, "data", None) is not None:
                parts.append(str(item.data))
            elif isinstance(item, dict) and item.get("type") == "text" and item.get("text") is not None:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _step_mean_logprob(step: TrajectoryStep, role: str) -> float | None:
    metadata = step.metadata or {}
    direct_keys = (
        f"{role}_logprob_mean",
        f"{role}_mean_logprob",
        f"{role}_avg_logprob",
    )
    for key in direct_keys:
        if key in metadata:
            return float(metadata[key])

    logprobs = metadata.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    value = logprobs.get(role)
    if isinstance(value, (float, int)):
        return float(value)
    if not isinstance(value, dict):
        return None
    for key in ("mean_logprob", "logprob_mean", "avg_logprob", "sequence_logprob"):
        if key in value and value[key] is not None:
            return float(value[key])
    return None


def _media_payload(data: Any, nested_key: str) -> Any:
    if isinstance(data, dict):
        if "url" in data:
            return data["url"]
        nested = data.get(nested_key)
        if isinstance(nested, dict) and "url" in nested:
            return nested["url"]
        if nested is not None:
            return nested
    return data
