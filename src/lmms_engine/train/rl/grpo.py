from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lmms_engine.datasets.collator import VisionCollator
from lmms_engine.rl.core.interfaces import TrainBatchAdapter
from lmms_engine.rl.protocol import RewardedTrajectory, TrajectoryStep, TrainBatch


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
            for step_idx, step in enumerate(trajectory.steps):
                sample = self._step_to_processor_sample(step)
                if sample is None:
                    continue
                samples.append(sample)
                rewards.append(self._step_reward(step, trajectory_reward))
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

        hf_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
        ]
        return self.processor.process(
            images=images or None,
            hf_messages=hf_messages,
            videos=videos or None,
            system_message=self.config.system_message,
            add_system_prompt=self.config.add_system_prompt,
            **self.config.processor_kwargs,
        )

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


def _agent_input_to_hf_content(request: Any) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
    content: list[dict[str, Any]] = []
    images: list[Any] = []
    videos: list[Any] = []

    for block in getattr(request, "content", []) or []:
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
