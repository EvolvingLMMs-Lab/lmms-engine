from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import ray
from loguru import logger

TRAIN_NODE_RESOURCE = "train_node"
ROLLOUT_NODE_RESOURCE = "rollout_node"


@dataclass(frozen=True)
class RayClusterSpec:
    num_nodes: int
    gpus_per_node: int
    master_addr: str
    ray_port: str
    node_rank: str
    train_node_rank: str
    head_node_ip: str
    wait_timeout: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RayClusterSpec":
        num_nodes = int(config.get("nnodes") or 1)
        master_addr = config.get("master_addr")
        if num_nodes > 1 and not master_addr:
            raise ValueError("--master-addr is required when --nnodes > 1.")

        return cls(
            num_nodes=num_nodes,
            gpus_per_node=int(config.get("gpus_per_node") or default_gpus_per_node()),
            master_addr=str(master_addr or "127.0.0.1"),
            ray_port=str(config.get("ray_port") or "6379"),
            node_rank=str(config.get("node_rank") or "0"),
            train_node_rank=str(config.get("train_node_rank") or "0"),
            head_node_ip=str(config.get("head_node_ip") or master_addr or "127.0.0.1"),
            wait_timeout=int(config.get("wait_timeout") or 300),
        )

    @property
    def ray_address(self) -> str:
        return f"{self.master_addr}:{self.ray_port}"


@dataclass(frozen=True)
class WorkerDescriptor:
    id: str
    role: str
    ip: str
    worker_ports: tuple[str, ...]
    resources: dict[str, float]


@dataclass(frozen=True)
class RayResourcePlan:
    rollout_gpus: int
    train_gpus: int
    model_server_replicas: int
    model_server_gpus_per_replica: float
    model_server_role_resources: dict[str, tuple[int, float]]
    rollout_workers: int
    rollout_max_inflight_per_worker: int
    rollout_batch_size: int
    train_workers: int
    train_use_gpu: bool
    train_gpus_per_worker: float

    @classmethod
    def from_config(cls, spec: RayClusterSpec, config: dict[str, Any]) -> "RayResourcePlan":
        for removed_name in (
            "TRAIN_BATCH_SIZE",
            "TRAIN_BATCH_SIZE_PER_GPU",
            "MIN_TRAJECTORIES_PER_BATCH",
            "RL_GLOBAL_BATCH_SIZE",
        ):
            if removed_name in os.environ:
                raise ValueError(
                    f"{removed_name} has been removed from the Ray resource environment. "
                    "Set trainer_args.rl_config.data_buffer.train_batch_size_per_gpu in YAML instead."
                )

        trainer_args = dict(config.get("trainer_args", {}) or {})
        rl_config = dict(trainer_args.get("rl_config", {}) or {})
        model_server_config = dict(rl_config.get("model_server", {}) or {})
        rollout_config = dict(rl_config.get("rollout", {}) or {})
        ray_train_config = dict(config.get("ray_train", {}) or {})

        rollout_gpus = max(0, spec.num_nodes - 1) * spec.gpus_per_node
        train_gpus = spec.gpus_per_node
        model_server_role_resources = _model_server_role_resources(
            rl_config,
            rollout_gpus=rollout_gpus,
            legacy_model_server_config=model_server_config,
        )
        model_server_replicas = sum(replicas for replicas, _gpus in model_server_role_resources.values())
        model_server_gpus_per_replica = (
            next(iter(model_server_role_resources.values()))[1] if model_server_role_resources else 1.0
        )
        model_server_gpu_total = sum(
            replicas * gpus_per_replica
            for replicas, gpus_per_replica in model_server_role_resources.values()
        )
        rollout_workers = int(rollout_config.get("num_workers") or model_server_replicas or 1)
        train_workers = int(ray_train_config.get("num_workers") or train_gpus)
        resources_per_worker = dict(ray_train_config.get("resources_per_worker", {}) or {})
        train_gpus_per_worker = float(resources_per_worker.get("GPU") or 1.0)
        if train_workers < 1:
            raise ValueError(f"ray_train.num_workers must be >= 1, got {train_workers}.")
        if rollout_gpus < model_server_gpu_total:
            role_details = ", ".join(
                f"{role}={replicas}x{gpus_per_replica:g}"
                for role, (replicas, gpus_per_replica) in model_server_role_resources.items()
            )
            raise ValueError(
                "Not enough rollout GPUs for vLLM replicas: "
                f"rollout_gpus={rollout_gpus}, requested={model_server_gpu_total:g} "
                f"({role_details}). "
                "Check NUM_GPUS_PER_NODE and Ray node GPU registration."
            )
        if train_gpus < train_workers * train_gpus_per_worker:
            raise ValueError(
                "Not enough train GPUs for Ray Train workers: "
                f"train_gpus={train_gpus}, train_workers={train_workers}, "
                f"gpus_per_worker={train_gpus_per_worker}. "
                "Check NUM_GPUS_PER_NODE and Ray node GPU registration."
            )

        return cls(
            rollout_gpus=rollout_gpus,
            train_gpus=train_gpus,
            model_server_replicas=model_server_replicas,
            model_server_gpus_per_replica=model_server_gpus_per_replica,
            model_server_role_resources=model_server_role_resources,
            rollout_workers=rollout_workers,
            rollout_max_inflight_per_worker=int(rollout_config.get("max_inflight_per_worker") or 1),
            rollout_batch_size=int(rollout_config.get("batch_size") or 2),
            train_workers=train_workers,
            train_use_gpu=bool(ray_train_config.get("use_gpu", True)),
            train_gpus_per_worker=train_gpus_per_worker,
        )

    def apply_to(self, config: dict[str, Any]) -> None:
        rl_config = config.setdefault("trainer_args", {}).setdefault("rl_config", {})
        self._apply_model_server(rl_config)
        self._apply_rollout(rl_config)
        self._apply_data_buffer(rl_config)
        self._apply_training(rl_config)
        self._apply_ray_train(config)

    def _apply_model_server(self, rl_config: dict[str, Any]) -> None:
        model_servers = rl_config.get("model_servers")
        if isinstance(model_servers, dict) and model_servers:
            for server in model_servers.values():
                if not isinstance(server, dict):
                    continue
                if (server.get("name") or server.get("backend")) != "ray_actor_pool":
                    continue
                actor_options = server.setdefault("actor_options", {})
                actor_options.setdefault("scheduling_strategy", "DEFAULT")
                actor_options.setdefault("resources", {})[ROLLOUT_NODE_RESOURCE] = 0.001
                load_balancer_options = server.setdefault("load_balancer_actor_options", {})
                load_balancer_options.setdefault("scheduling_strategy", "DEFAULT")
                load_balancer_options.setdefault("resources", {})[ROLLOUT_NODE_RESOURCE] = 0.001
            return

        model_server = rl_config.setdefault("model_server", {})
        model_server["num_replicas"] = self.model_server_replicas
        actor_options = model_server.setdefault("actor_options", {})
        actor_options["num_gpus"] = self.model_server_gpus_per_replica
        actor_options.setdefault("scheduling_strategy", "DEFAULT")
        actor_options.setdefault("resources", {})[ROLLOUT_NODE_RESOURCE] = 0.001
        load_balancer_options = model_server.setdefault("load_balancer_actor_options", {})
        load_balancer_options.setdefault("scheduling_strategy", "DEFAULT")
        load_balancer_options.setdefault("resources", {})[ROLLOUT_NODE_RESOURCE] = 0.001

    def _apply_rollout(self, rl_config: dict[str, Any]) -> None:
        rollout = rl_config.setdefault("rollout", {})
        rollout["num_workers"] = self.rollout_workers
        rollout["max_inflight_per_worker"] = self.rollout_max_inflight_per_worker
        rollout["batch_size"] = self.rollout_batch_size
        actor_options = rollout.setdefault("actor_options", {})
        actor_options.setdefault("scheduling_strategy", "DEFAULT")
        actor_options.setdefault("resources", {})[ROLLOUT_NODE_RESOURCE] = 0.001

    def _apply_data_buffer(self, rl_config: dict[str, Any]) -> None:
        data_buffer = rl_config.setdefault("data_buffer", {})
        if "train_batch_size" in data_buffer:
            raise ValueError(
                "trainer_args.rl_config.data_buffer.train_batch_size has been removed. "
                "Use trainer_args.rl_config.data_buffer.train_batch_size_per_gpu instead."
            )
        train_batch_size_per_gpu = int(data_buffer.get("train_batch_size_per_gpu", 1))
        if train_batch_size_per_gpu < 1:
            raise ValueError(
                "trainer_args.rl_config.data_buffer.train_batch_size_per_gpu must be >= 1, "
                f"got {train_batch_size_per_gpu}."
            )
        global_train_batch_size = train_batch_size_per_gpu * self.train_workers
        configured_global = data_buffer.get("global_train_batch_size")
        if configured_global not in (None, global_train_batch_size):
            raise ValueError(
                "trainer_args.rl_config.data_buffer.global_train_batch_size is derived from "
                "train_batch_size_per_gpu * ray_train.num_workers; leave it null."
            )
        data_buffer["global_train_batch_size"] = global_train_batch_size
        if data_buffer.get("min_trajectories_per_batch") is None:
            data_buffer["min_trajectories_per_batch"] = global_train_batch_size

    def _apply_training(self, rl_config: dict[str, Any]) -> None:
        training = rl_config.setdefault("training", {})
        data_buffer = rl_config.setdefault("data_buffer", {})
        global_train_batch_size = data_buffer["global_train_batch_size"]
        configured_global = training.get("global_batch_size")
        if configured_global not in (None, global_train_batch_size):
            raise ValueError(
                "trainer_args.rl_config.training.global_batch_size is derived from "
                "data_buffer.train_batch_size_per_gpu * ray_train.num_workers; leave it null."
            )
        training["global_batch_size"] = global_train_batch_size

    def _apply_ray_train(self, config: dict[str, Any]) -> None:
        ray_train = dict(config.get("ray_train", {}) or {})
        ray_train["num_workers"] = self.train_workers
        ray_train["use_gpu"] = self.train_use_gpu
        ray_train["resources_per_worker"] = {
            **dict(ray_train.get("resources_per_worker", {}) or {}),
            "GPU": self.train_gpus_per_worker,
            TRAIN_NODE_RESOURCE: 0.001,
        }
        ray_train["placement_strategy"] = str(ray_train.get("placement_strategy", "PACK"))
        config["ray_train"] = ray_train


class RayNodeScheduler:
    def __init__(self, spec: RayClusterSpec):
        self.spec = spec

    def current_worker(self) -> WorkerDescriptor:
        role = "train" if self.is_train_node() else "rollout"
        resource_name = TRAIN_NODE_RESOURCE if role == "train" else ROLLOUT_NODE_RESOURCE
        ip = self.spec.head_node_ip if role == "train" else _local_node_ip()
        node_rank = self.spec.node_rank
        return WorkerDescriptor(
            id=f"{role}/{node_rank}",
            role=role,
            ip=ip,
            worker_ports=(str(self.spec.ray_port),) if role == "train" else (),
            resources={resource_name: 1000.0},
        )

    def is_train_node(self) -> bool:
        return self.spec.node_rank == self.spec.train_node_rank

    def start_head(self) -> None:
        self.stop()
        _ray_start(
            [
                "--head",
                "--node-ip-address",
                self.spec.head_node_ip,
                "--port",
                str(self.spec.ray_port),
                "--num-gpus",
                str(self.spec.gpus_per_node),
                "--resources",
                json.dumps({TRAIN_NODE_RESOURCE: 1000}),
            ],
            gpus_per_node=self.spec.gpus_per_node,
        )

    def join_rollout_node(self) -> None:
        self.stop()
        while True:
            try:
                logger.info(f"Starting rollout node and joining Ray at {self.spec.ray_address}")
                _ray_start(
                    [
                        "--block",
                        "--address",
                        self.spec.ray_address,
                        "--num-gpus",
                        str(self.spec.gpus_per_node),
                        "--resources",
                        json.dumps({ROLLOUT_NODE_RESOURCE: 1000}),
                    ],
                    gpus_per_node=self.spec.gpus_per_node,
                )
                return
            except subprocess.CalledProcessError:
                logger.info(f"Waiting for Ray head at {self.spec.ray_address}...")
                time.sleep(5)

    def wait_for_workers(self) -> None:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(address=self.spec.ray_address, ignore_reinit_error=True)
        deadline = time.time() + self.spec.wait_timeout
        try:
            while True:
                alive = [node for node in ray.nodes() if node.get("Alive")]
                if len(alive) >= self.spec.num_nodes:
                    logger.info(f"Ray cluster ready: {len(alive)}/{self.spec.num_nodes} nodes alive.")
                    _validate_cluster_resources(alive, self.spec)
                    return
                if time.time() > deadline:
                    raise TimeoutError(f"Timed out waiting for Ray nodes: {len(alive)}/{self.spec.num_nodes} alive.")
                time.sleep(5)
        finally:
            ray.shutdown()

    def stop(self) -> None:
        _ray_stop()


def _model_server_role_resources(
    rl_config: Mapping[str, Any],
    *,
    rollout_gpus: int,
    legacy_model_server_config: Mapping[str, Any],
) -> dict[str, tuple[int, float]]:
    configured_roles = rl_config.get("model_servers")
    if isinstance(configured_roles, Mapping) and configured_roles:
        role_specs = {
            str(role): spec
            for role, spec in configured_roles.items()
            if isinstance(spec, Mapping) and (spec.get("name") or spec.get("backend")) == "ray_actor_pool"
        }
    elif isinstance(legacy_model_server_config, Mapping):
        role_specs = {"policy": legacy_model_server_config}
    else:
        role_specs = {}

    resources: dict[str, tuple[int, float]] = {}
    for role, spec in role_specs.items():
        actor_options = dict(spec.get("actor_options", {}) or {})
        gpus_per_replica = float(actor_options.get("num_gpus") or 1.0)
        default_replicas = max(1, int(rollout_gpus / gpus_per_replica)) if len(role_specs) == 1 else 1
        replicas = int(spec.get("num_replicas") or default_replicas)
        resources[role] = (replicas, gpus_per_replica)
    return resources


class RayRLMultinodeRuntime:
    def __init__(self, spec: RayClusterSpec, scheduler: RayNodeScheduler):
        self.spec = spec
        self.scheduler = scheduler

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RayRLMultinodeRuntime":
        spec = RayClusterSpec.from_config(config)
        return cls(spec=spec, scheduler=RayNodeScheduler(spec))

    def run(self, config: dict[str, Any], train: Callable[[dict[str, Any]], None]) -> None:
        worker = self.scheduler.current_worker()
        logger.info(
            f"Multinode RL worker: id={worker.id}, role={worker.role}, " f"ip={worker.ip}, resources={worker.resources}"
        )

        if worker.role == "rollout":
            self.scheduler.join_rollout_node()
            return

        self.scheduler.start_head()
        try:
            self.scheduler.wait_for_workers()
            RayResourcePlan.from_config(self.spec, config).apply_to(config)
            ray_train = config.setdefault("ray_train", {})
            ray_init_kwargs = ray_train.setdefault("ray_init_kwargs", {})
            ray_init_kwargs["address"] = self.spec.ray_address
            clear_external_distributed_env()
            train(config)
        finally:
            self.scheduler.stop()


def use_multinode_config(config: dict[str, Any]) -> bool:
    return bool(config.get("enabled", True)) and int(config.get("nnodes") or 1) > 1


def clear_external_distributed_env() -> None:
    for name in (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        os.environ.pop(name, None)


def default_gpus_per_node() -> int:
    nvidia_smi_count = _nvidia_smi_gpu_count()
    if nvidia_smi_count:
        return nvidia_smi_count

    return 1


def _nvidia_smi_gpu_count() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    count = sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU "))
    return count or None


def _local_node_ip() -> str:
    for address in sorted(_local_addresses()):
        if address not in {"127.0.0.1", "::1", "localhost"}:
            return address
    return socket.gethostname()


def _local_addresses() -> set[str]:
    addresses = {"127.0.0.1", "::1", "localhost"}
    names = {socket.gethostname(), socket.getfqdn()}
    for name in names:
        addresses.add(name)
        addresses.update(_resolve_addresses(name))
    return addresses


def _resolve_addresses(host: str) -> set[str]:
    addresses = {host}
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
            if family in {socket.AF_INET, socket.AF_INET6}:
                addresses.add(sockaddr[0])
    except socket.gaierror:
        pass
    return addresses


def _validate_cluster_resources(alive_nodes: list[dict[str, Any]], spec: RayClusterSpec) -> None:
    summaries = []
    train_nodes = 0
    rollout_nodes = 0
    total_gpus = 0.0

    for node in alive_nodes:
        resources = dict(node.get("Resources") or {})
        node_id = node.get("NodeID") or "unknown"
        node_ip = node.get("NodeManagerAddress") or node.get("NodeManagerHostname") or "unknown"
        gpus = float(resources.get("GPU", 0.0))
        total_gpus += gpus
        if resources.get(TRAIN_NODE_RESOURCE, 0.0) > 0:
            train_nodes += 1
            role = "train"
        elif resources.get(ROLLOUT_NODE_RESOURCE, 0.0) > 0:
            rollout_nodes += 1
            role = "rollout"
        else:
            role = "unknown"
        summaries.append(f"{role}@{node_ip} gpu={gpus:g} node_id={node_id[:8]}")
        if gpus < spec.gpus_per_node:
            raise ValueError(
                "Ray node registered fewer GPUs than expected: "
                f"{role}@{node_ip} has GPU={gpus:g}, expected={spec.gpus_per_node}. "
                "Set NUM_GPUS_PER_NODE=8 and make sure Ray is started with all node GPUs visible."
            )

    expected_rollout_nodes = max(0, spec.num_nodes - 1)
    expected_total_gpus = spec.num_nodes * spec.gpus_per_node
    logger.info("Ray cluster resource summary: " + "; ".join(summaries))
    if train_nodes != 1:
        raise ValueError(f"Expected exactly one Ray train node, got {train_nodes}.")
    if rollout_nodes != expected_rollout_nodes:
        raise ValueError(f"Expected {expected_rollout_nodes} Ray rollout nodes, got {rollout_nodes}.")
    if total_gpus < expected_total_gpus:
        raise ValueError(
            "Ray cluster registered too few total GPUs: "
            f"got {total_gpus:g}, expected {expected_total_gpus}. "
            "Check NUM_NODES/NUM_GPUS_PER_NODE and launcher GPU visibility."
        )


def _ray_start(args: list[str], *, gpus_per_node: int | None = None) -> None:
    env = os.environ.copy()
    if gpus_per_node is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(gpus_per_node))
    subprocess.run([_ray_bin(), "start", *args], check=True, env=env)


def _ray_stop() -> None:
    subprocess.run(
        [_ray_bin(), "stop", "--force"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _ray_bin() -> str:
    explicit = os.environ.get("RAY")
    if explicit:
        return explicit
    sibling = Path(sys.executable).with_name("ray")
    if sibling.exists():
        return str(sibling)
    resolved = shutil.which("ray")
    if resolved:
        return resolved
    raise FileNotFoundError("Could not find the `ray` executable. Set RAY=/path/to/ray.")
