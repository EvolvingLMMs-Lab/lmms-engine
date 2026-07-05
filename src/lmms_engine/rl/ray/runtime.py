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
from typing import Any, Callable

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
    train_node_rank: str
    head_node_ip: str
    wait_timeout: int

    @classmethod
    def from_env(cls, *, default_gpus_per_node: int) -> "RayClusterSpec":
        master_addr = os.environ.get("MASTER_ADDR")
        if not master_addr:
            raise ValueError("WORLD_SIZE>1 requires MASTER_ADDR for the Ray head/training node.")

        return cls(
            num_nodes=env_int("WORLD_SIZE", 1),
            gpus_per_node=env_int("NUM_GPUS_PER_NODE", default_gpus_per_node),
            master_addr=master_addr,
            ray_port=os.environ.get("RAY_PORT") or os.environ.get("MASTER_PORT", "6379"),
            train_node_rank=os.environ.get("TRAIN_NODE_RANK", "0"),
            head_node_ip=os.environ.get("HEAD_NODE_IP", master_addr),
            wait_timeout=env_int("RAY_WAIT_TIMEOUT", 300),
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
    rollout_workers: int
    rollout_max_inflight_per_worker: int
    data_buffer_max_trajectories: int
    data_buffer_high_watermark: int
    data_buffer_low_watermark: int
    train_batch_size: int
    min_trajectories_per_batch: int
    global_batch_size: int
    train_workers: int
    train_use_gpu: bool
    train_gpus_per_worker: float

    @classmethod
    def from_env(cls, spec: RayClusterSpec) -> "RayResourcePlan":
        rollout_gpus = env_int("ROLLOUT_GPUS", max(0, spec.num_nodes - 1) * spec.gpus_per_node)
        train_gpus = env_int("TRAIN_GPUS", spec.gpus_per_node)
        model_server_replicas = env_int("MODEL_SERVER_REPLICAS", rollout_gpus)
        rollout_workers = env_int("ROLLOUT_WORKERS", model_server_replicas)

        return cls(
            rollout_gpus=rollout_gpus,
            train_gpus=train_gpus,
            model_server_replicas=model_server_replicas,
            model_server_gpus_per_replica=env_float("MODEL_SERVER_GPUS_PER_REPLICA", 1.0),
            rollout_workers=rollout_workers,
            rollout_max_inflight_per_worker=env_int("ROLLOUT_MAX_INFLIGHT_PER_WORKER", 1),
            data_buffer_max_trajectories=env_int("DATA_BUFFER_MAX_TRAJECTORIES", max(1, rollout_workers * 4)),
            data_buffer_high_watermark=env_int("DATA_BUFFER_HIGH_WATERMARK", max(1, rollout_workers * 3)),
            data_buffer_low_watermark=env_int("DATA_BUFFER_LOW_WATERMARK", max(1, rollout_workers)),
            train_batch_size=env_int("TRAIN_BATCH_SIZE", train_gpus),
            min_trajectories_per_batch=env_int("MIN_TRAJECTORIES_PER_BATCH", train_gpus),
            global_batch_size=env_int("RL_GLOBAL_BATCH_SIZE", train_gpus),
            train_workers=env_int("RAY_TRAIN_NUM_WORKERS", train_gpus),
            train_use_gpu=env_bool("RAY_TRAIN_USE_GPU", True),
            train_gpus_per_worker=env_float("RAY_TRAIN_NUM_GPUS_PER_WORKER", 1.0),
        )

    def apply_to(self, config: dict[str, Any]) -> None:
        rl_config = config.setdefault("trainer_args", {}).setdefault("rl_config", {})
        self._apply_model_server(rl_config)
        self._apply_rollout(rl_config)
        self._apply_data_buffer(rl_config)
        self._apply_training(rl_config)
        self._apply_ray_train(config)

    def _apply_model_server(self, rl_config: dict[str, Any]) -> None:
        model_server = rl_config.setdefault("model_server", {})
        model_server["num_replicas"] = self.model_server_replicas
        actor_options = model_server.setdefault("actor_options", {})
        actor_options["num_gpus"] = self.model_server_gpus_per_replica
        actor_options.setdefault("scheduling_strategy", "DEFAULT")
        actor_options.setdefault("resources", {})[ROLLOUT_NODE_RESOURCE] = 0.001

    def _apply_rollout(self, rl_config: dict[str, Any]) -> None:
        rollout = rl_config.setdefault("rollout", {})
        rollout["num_workers"] = self.rollout_workers
        rollout["max_inflight_per_worker"] = self.rollout_max_inflight_per_worker
        actor_options = rollout.setdefault("actor_options", {})
        actor_options.setdefault("scheduling_strategy", "DEFAULT")
        actor_options.setdefault("resources", {})[ROLLOUT_NODE_RESOURCE] = 0.001

    def _apply_data_buffer(self, rl_config: dict[str, Any]) -> None:
        data_buffer = rl_config.setdefault("data_buffer", {})
        data_buffer["max_trajectories"] = self.data_buffer_max_trajectories
        data_buffer["high_watermark"] = self.data_buffer_high_watermark
        data_buffer["low_watermark"] = self.data_buffer_low_watermark
        data_buffer["train_batch_size"] = self.train_batch_size
        data_buffer["min_trajectories_per_batch"] = self.min_trajectories_per_batch

    def _apply_training(self, rl_config: dict[str, Any]) -> None:
        training = rl_config.setdefault("training", {})
        training["global_batch_size"] = self.global_batch_size

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
        node_rank = self._node_rank() or "local"
        return WorkerDescriptor(
            id=f"{role}/{node_rank}",
            role=role,
            ip=ip,
            worker_ports=(str(self.spec.ray_port),) if role == "train" else (),
            resources={resource_name: 1000.0},
        )

    def is_train_node(self) -> bool:
        role = os.environ.get("ROLE", "").lower()
        if role in {"train", "trainer", "head"}:
            return True
        if role in {"rollout", "worker"}:
            return False

        node_rank = self._node_rank()
        if node_rank is not None:
            return node_rank == self.spec.train_node_rank

        return bool(_resolve_addresses(self.spec.master_addr) & _local_addresses())

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
            ]
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
                    ]
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
                    return
                if time.time() > deadline:
                    raise TimeoutError(f"Timed out waiting for Ray nodes: {len(alive)}/{self.spec.num_nodes} alive.")
                time.sleep(5)
        finally:
            ray.shutdown()

    def stop(self) -> None:
        _ray_stop()

    @staticmethod
    def _node_rank() -> str | None:
        for env_name in ("NODE_RANK", "SLURM_NODEID", "RANK"):
            value = os.environ.get(env_name)
            if value is not None:
                return value
        return None


class RayRLMultinodeRuntime:
    def __init__(self, spec: RayClusterSpec, scheduler: RayNodeScheduler):
        self.spec = spec
        self.scheduler = scheduler

    @classmethod
    def from_env(cls, *, default_gpus_per_node: int) -> "RayRLMultinodeRuntime":
        spec = RayClusterSpec.from_env(default_gpus_per_node=default_gpus_per_node)
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
            os.environ["RAY_ADDRESS"] = self.spec.ray_address
            self.scheduler.wait_for_workers()
            RayResourcePlan.from_env(self.spec).apply_to(config)
            clear_external_distributed_env()
            train(config)
        finally:
            self.scheduler.stop()


def use_multinode_default() -> bool:
    return env_bool("LMMS_ENGINE_RL_MULTINODE", True) and env_int("WORLD_SIZE", 1) > 1


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


def default_num_workers() -> int:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [device for device in visible_devices.split(",") if device.strip()]
        return max(1, len(devices))
    return 1


def env_bool(name: str, default: Any) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: Any) -> int:
    value = os.environ.get(name)
    return int(value if value is not None else default)


def env_float(name: str, default: Any) -> float:
    value = os.environ.get(name)
    return float(value if value is not None else default)


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


def _ray_start(args: list[str]) -> None:
    subprocess.run([_ray_bin(), "start", *args], check=True)


def _ray_stop() -> None:
    subprocess.run([_ray_bin(), "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


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
