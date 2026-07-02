from __future__ import annotations

from typing import Any

import requests

from lmms_engine.rl.config import VLLMServerConfig
from lmms_engine.rl.core.interfaces import WeightSyncClient
from lmms_engine.rl.protocol import ModelVersion


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
