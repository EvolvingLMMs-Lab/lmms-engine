from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from lmms_engine.rl.model_server.ray import (
    RayModelServerPool,
    start_ray_model_server_pool,
)

POLICY_ROLE = "policy"
OPENAI_COMPATIBLE_BACKENDS = {"openai"}
SUPPORTED_MODEL_SERVER_BACKENDS = {"ray_actor_pool", *OPENAI_COMPATIBLE_BACKENDS}


@dataclass(slots=True)
class ManagedModelServer:
    """One inference role owned by ModelServerManager."""

    role: str
    spec: dict[str, Any]
    pool: RayModelServerPool | None = None

    @property
    def backend(self) -> str | None:
        return self.spec.get("name") or self.spec.get("backend")

    def client_spec(self, **overrides: Any) -> dict[str, Any]:
        if self.pool is not None:
            return self.pool.client_spec(**dict(self.spec.get("client", {}) or {}), **overrides)
        return _external_client_spec(self.spec, **overrides)


class ModelServerManager:
    """Owns inference servers by algorithm role.

    The manager is role-oriented rather than algorithm-oriented: GRPO needs
    policy + optional reference; PPO can add reward and value roles without
    changing rollout scheduling.
    """

    def __init__(self, role_specs: Mapping[str, Mapping[str, Any]]) -> None:
        normalized = normalize_model_server_configs({"model_servers": role_specs})
        validate_model_server_configs(normalized)
        self._servers = {role: ManagedModelServer(role=role, spec=dict(spec)) for role, spec in normalized.items()}

    @classmethod
    def from_rl_config(
        cls,
        rl_config: Mapping[str, Any],
        *,
        legacy_model_server: Any = None,
    ) -> "ModelServerManager":
        return cls(normalize_model_server_configs(rl_config, legacy_model_server=legacy_model_server))

    def start(self) -> None:
        for role, server in self._servers.items():
            if server.backend != "ray_actor_pool" or server.pool is not None:
                continue
            spec = dict(server.spec)
            spec.setdefault("actor_name_prefix", f"lmms-engine-{role}-{os.getpid()}-{uuid.uuid4().hex[:8]}")
            server.pool = start_ray_model_server_pool(spec)

    def client_spec(self, role: str = POLICY_ROLE, **overrides: Any) -> dict[str, Any]:
        return self._require(role).client_spec(**overrides)

    def pool(self, role: str = POLICY_ROLE) -> RayModelServerPool | None:
        return self._require(role).pool

    def roles(self) -> list[str]:
        return list(self._servers)

    def specs(self) -> dict[str, dict[str, Any]]:
        return {role: dict(server.spec) for role, server in self._servers.items()}

    def summary(self) -> str:
        parts = []
        for role, server in self._servers.items():
            if server.pool is None:
                parts.append(f"{role}:{server.backend}")
            else:
                parts.append(f"{role}:ray_actor_pool[{len(server.pool.actor_names)}]")
        return ", ".join(parts)

    def _require(self, role: str) -> ManagedModelServer:
        if role not in self._servers:
            available = ", ".join(sorted(self._servers))
            raise KeyError(f"Unknown model server role {role!r}. Available roles: {available}.")
        return self._servers[role]


def normalize_model_server_configs(
    rl_config: Mapping[str, Any],
    *,
    legacy_model_server: Any = None,
) -> dict[str, dict[str, Any]]:
    """Normalize legacy single-server config into a role map."""

    configured = rl_config.get("model_servers")
    role_specs: dict[str, dict[str, Any]] = {}
    if configured is not None:
        if not isinstance(configured, Mapping):
            raise TypeError("rl_config.model_servers must be a mapping from role name to model-server spec.")
        for role, spec in configured.items():
            if spec is None:
                continue
            role_specs[str(role)] = _coerce_model_server_spec(spec, role=str(role))

    legacy = rl_config.get("model_server", legacy_model_server)
    if POLICY_ROLE not in role_specs and legacy is not None:
        role_specs[POLICY_ROLE] = _coerce_model_server_spec(legacy, role=POLICY_ROLE)

    if not role_specs:
        raise ValueError(
            "RL requires at least a policy model server. Set rl_config.model_server or rl_config.model_servers.policy."
        )
    if POLICY_ROLE not in role_specs:
        raise ValueError("rl_config.model_servers must include a 'policy' role.")
    return role_specs


def validate_model_server_configs(
    role_specs: Mapping[str, Mapping[str, Any]],
    *,
    policy_weight_sync_backend: str | None = None,
) -> None:
    if POLICY_ROLE not in role_specs:
        raise ValueError("model server roles must include 'policy'.")
    for role, spec in role_specs.items():
        backend = spec.get("name") or spec.get("backend")
        if backend not in SUPPORTED_MODEL_SERVER_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_MODEL_SERVER_BACKENDS))
            raise ValueError(f"model_servers.{role}.name must be one of {{{supported}}}, got {backend!r}.")
        if (
            role == POLICY_ROLE
            and backend in OPENAI_COMPATIBLE_BACKENDS
            and policy_weight_sync_backend == "ray_actor_pool"
        ):
            raise ValueError(
                "model_servers.policy.name='openai' uses an external OpenAI-compatible endpoint. "
                "Set rl_config.vllm.backend='vllm_http' or a native vLLM HTTP weight-sync backend, not 'ray_actor_pool'."
            )
        if backend == "ray_actor_pool":
            _validate_ray_actor_pool_spec(role, spec)


def _coerce_model_server_spec(spec: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise TypeError(f"model_servers.{role} must be a dict model-server spec.")
    return dict(spec)


def _validate_ray_actor_pool_spec(role: str, spec: Mapping[str, Any]) -> None:
    server = spec.get("server") or spec.get("server_spec")
    if not isinstance(server, Mapping):
        raise ValueError(f"model_servers.{role}.server must be a dict vLLM server spec.")
    factory = server.get("factory")
    expected = "lmms_engine.rl.model_server.vllm:VLLMChatModelServer"
    if factory != expected:
        raise ValueError(f"model_servers.{role}.server.factory must be {expected!r}; got {factory!r}.")


def _external_client_spec(spec: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    backend = spec.get("name") or spec.get("backend")
    if backend not in OPENAI_COMPATIBLE_BACKENDS:
        raise ValueError(f"Cannot build external client spec for backend {backend!r}.")
    client = {key: value for key, value in dict(spec).items() if key not in {"client"}}
    client.update(dict(spec.get("client", {}) or {}))
    client.update(overrides)
    client["name"] = "openai"
    return client
