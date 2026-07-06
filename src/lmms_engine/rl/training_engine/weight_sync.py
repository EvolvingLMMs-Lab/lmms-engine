from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from lmms_engine.rl.config import VLLMServerConfig
from lmms_engine.rl.core.interfaces import WeightSyncClient
from lmms_engine.rl.protocol import ModelVersion


def _require_ray():
    try:
        import ray
    except ImportError as exc:
        raise ImportError("Ray is required for ray_actor_pool policy weight sync.") from exc
    return ray


class VLLMWeightSyncClient(WeightSyncClient):
    """Small HTTP boundary for policy weight reloads.

    The concrete vLLM endpoint can evolve independently. The RL loop only needs
    a versioned acknowledgement after the trainer publishes new weights.
    """

    def __init__(self, config: VLLMServerConfig | None = None) -> None:
        self.config = config or VLLMServerConfig()

    def reload_weights(self, model_version: ModelVersion) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}{self.config.reload_endpoint}"
        payload = {
            "model": self.config.model,
            "version_id": model_version.version_id,
            "checkpoint_path": model_version.checkpoint_path,
            "metadata": model_version.metadata,
            **self.config.extra_kwargs,
        }
        response = requests.post(url, json=payload, timeout=self.config.timeout_s)
        response.raise_for_status()
        return dict(response.json())


@dataclass(slots=True)
class RayActorWeightSyncClient(WeightSyncClient):
    """Reload policy weights on a Ray pool of vLLM model-server actors."""

    actor_handles: list[Any] = field(default_factory=list)
    actor_names: list[str] = field(default_factory=list)
    namespace: str | None = None
    timeout_s: float | None = 600.0
    preflight_weight_path: bool = True
    require_hf_checkpoint: bool = True
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def reload_weights(self, model_version: ModelVersion) -> dict[str, Any]:
        if not model_version.checkpoint_path:
            raise ValueError("RayActorWeightSyncClient requires ModelVersion.checkpoint_path.")
        if not Path(model_version.checkpoint_path).exists():
            raise FileNotFoundError(
                "Policy weight checkpoint does not exist on the training worker: "
                f"{model_version.checkpoint_path}"
            )

        ray = _require_ray()
        actors = self._resolve_actor_handles(ray)
        if not actors:
            raise RuntimeError("No Ray model-server actors are available for policy weight sync.")

        preflight_results = []
        if self.preflight_weight_path:
            refs = [
                actor.validate_weight_path.remote(
                    model_version.checkpoint_path,
                    require_hf_checkpoint=self.require_hf_checkpoint,
                )
                for actor in actors
            ]
            try:
                preflight_results = list(ray.get(refs, timeout=self.timeout_s))
            except Exception as exc:
                raise RuntimeError(
                    "Policy weight checkpoint is not visible or not loadable from every Ray model-server actor. "
                    "trainer_args.rl_config.update_weight_disk_dir must be an absolute shared filesystem path visible "
                    f"from train and rollout nodes. checkpoint_path={model_version.checkpoint_path}"
                ) from exc

        payload = {
            "checkpoint_path": model_version.checkpoint_path,
            "version_id": model_version.version_id,
            "metadata": model_version.metadata,
            "require_hf_checkpoint": self.require_hf_checkpoint,
            **self.extra_kwargs,
        }
        refs = [actor.update_weights.remote(**payload) for actor in actors]
        results = ray.get(refs, timeout=self.timeout_s)
        return {
            "backend": "ray_actor_pool",
            "version_id": model_version.version_id,
            "checkpoint_path": model_version.checkpoint_path,
            "num_actors": len(actors),
            "preflight": preflight_results,
            "actors": list(results),
        }

    def _resolve_actor_handles(self, ray: Any) -> list[Any]:
        if self.actor_handles:
            return list(self.actor_handles)
        return [ray.get_actor(name, namespace=self.namespace) for name in self.actor_names]
