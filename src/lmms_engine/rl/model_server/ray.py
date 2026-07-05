from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

from lmms_engine.rl.lmms_eval.paths import ensure_lmms_eval_importable

ensure_lmms_eval_importable()

from lmms_eval.agentic.model_server.ray import (
    RayActorModelServer,
    RayModelServerActor,
    RayModelServerLoadBalancer,
)


def _require_ray():
    try:
        import ray
    except ImportError as exc:
        raise ImportError("Ray is required for ray_actor_pool model serving. Install `lmms_engine[rl]`.") from exc
    return ray


def _detached_actor_options(options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = dict(options or {})
    options.setdefault("scheduling_strategy", "DEFAULT")
    return options


@dataclass(slots=True)
class RayModelServerPool:
    actor_names: list[str]
    load_balancer_name: str
    namespace: str | None
    actor_handles: list[Any]
    load_balancer_handle: Any

    def client_spec(self, **overrides: Any) -> dict[str, Any]:
        return {
            "name": "ray",
            "load_balancer_name": self.load_balancer_name,
            "namespace": self.namespace,
            **overrides,
        }


def start_ray_model_server_pool(config: dict[str, Any]) -> RayModelServerPool:
    """Start a Ray actor pool and load balancer for lmms-eval ModelServer specs."""
    ray = _require_ray()
    server_spec = config.get("server") or config.get("server_spec")
    if server_spec is None:
        raise ValueError("ray_actor_pool model_server config requires a nested `server` spec.")

    namespace = config.get("namespace")
    num_replicas = int(config.get("num_replicas", 1))
    actor_options = _detached_actor_options(config.get("actor_options"))
    factory_components = dict(config.get("factory_components", {}) or {})
    prefix = config.get("actor_name_prefix") or f"lmms-engine-policy-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    lifetime = config.get("lifetime")
    get_if_exists = bool(config.get("get_if_exists", False))

    actor_cls = ray.remote(**actor_options)(RayModelServerActor)
    actor_names = []
    actor_handles = []
    for replica_idx in range(num_replicas):
        actor_name = f"{prefix}-{replica_idx}"
        options = {"name": actor_name}
        if namespace is not None:
            options["namespace"] = namespace
        if lifetime is not None:
            options["lifetime"] = lifetime
        if get_if_exists:
            options["get_if_exists"] = True
        actor_handle = actor_cls.options(**options).remote(server_spec, factory_components)
        actor_names.append(actor_name)
        actor_handles.append(actor_handle)

    lb_name = config.get("load_balancer_name") or f"{prefix}-lb"
    lb_actor_cls = ray.remote(**_detached_actor_options(config.get("load_balancer_actor_options")))(
        RayModelServerLoadBalancer
    )
    lb_options = {"name": lb_name}
    if namespace is not None:
        lb_options["namespace"] = namespace
    if lifetime is not None:
        lb_options["lifetime"] = lifetime
    if get_if_exists:
        lb_options["get_if_exists"] = True
    load_balancer_handle = lb_actor_cls.options(**lb_options).remote(
        actor_names=actor_names,
        namespace=namespace,
        sticky=bool(config.get("sticky", True)),
    )
    ray.get(load_balancer_handle.status.remote())
    return RayModelServerPool(
        actor_names=actor_names,
        load_balancer_name=lb_name,
        namespace=namespace,
        actor_handles=actor_handles,
        load_balancer_handle=load_balancer_handle,
    )
