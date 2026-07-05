from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import ray
import torch.distributed as dist
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from lmms_engine.train import RLTrainRunner
from lmms_engine.utils.logging_utils import setup_distributed_logging

from .cli import create_train_task, save_config


def _train_loop_per_worker(train_loop_config: dict[str, Any]) -> None:
    setup_distributed_logging()
    config = deepcopy(train_loop_config["config"])
    original_config = deepcopy(config)

    task = create_train_task(config)
    if not isinstance(task, RLTrainRunner):
        raise ValueError(
            "lmms_engine.launch.rl requires an RL trainer config, " "for example trainer_type=fsdp2_grpo_rl_trainer."
        )

    save_config(original_config)
    task.build()
    task.run()
    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="config", config_name="default_config")
def main(config: DictConfig):
    setup_distributed_logging()
    config = OmegaConf.to_yaml(config)
    config = yaml.safe_load(config)

    config_yaml = config.pop("config_yaml")
    if config_yaml:
        logger.info(f"Detected config yaml, merging with the default config: {config_yaml}")
        with open(config_yaml, "r") as f:
            config_yaml = yaml.safe_load(f)
        config.update(config_yaml)

    if _use_multinode_default():
        _run_multinode_default(config)
        return

    _run_ray_train(config)


def _run_ray_train(config: dict[str, Any]) -> None:
    ray_train_config = _ray_train_config(config)
    ray_init_kwargs = dict(ray_train_config.get("ray_init_kwargs", {}) or {})
    ray_address = os.environ.get("RAY_ADDRESS") or ray_init_kwargs.pop("address", None)
    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            ignore_reinit_error=True,
            include_dashboard=_env_bool("RAY_DASHBOARD", ray_train_config.get("include_dashboard", False)),
            runtime_env=_runtime_env(ray_train_config),
            **ray_init_kwargs,
        )

    trainer = TorchTrainer(
        train_loop_per_worker=_train_loop_per_worker,
        train_loop_config={"config": deepcopy(config)},
        scaling_config=_scaling_config(ray_train_config),
        run_config=_run_config(config, ray_train_config),
    )
    result = trainer.fit()
    if result.error is not None:
        raise result.error


def _use_multinode_default() -> bool:
    return _env_bool("LMMS_ENGINE_RL_MULTINODE", True) and _env_int("WORLD_SIZE", 1) > 1


def _run_multinode_default(config: dict[str, Any]) -> None:
    num_nodes = _env_int("WORLD_SIZE", 1)
    gpus_per_node = _env_int("NUM_GPUS_PER_NODE", _default_num_workers())
    master_addr = os.environ.get("MASTER_ADDR")
    if not master_addr:
        raise ValueError("WORLD_SIZE>1 requires MASTER_ADDR for the Ray head/training node.")
    ray_port = os.environ.get("RAY_PORT") or os.environ.get("MASTER_PORT", "6379")
    ray_address = f"{master_addr}:{ray_port}"
    train_node_rank = os.environ.get("TRAIN_NODE_RANK", "0")

    if not _is_train_node(master_addr, train_node_rank):
        _ray_stop()
        _start_rollout_node(ray_address, gpus_per_node)
        return

    _ray_stop()
    _ray_start(
        [
            "--head",
            "--node-ip-address",
            os.environ.get("HEAD_NODE_IP", master_addr),
            "--port",
            str(ray_port),
            "--num-gpus",
            str(gpus_per_node),
            "--resources",
            json.dumps({"train_node": 1000}),
        ]
    )

    try:
        os.environ["RAY_ADDRESS"] = ray_address
        _wait_for_ray_nodes(ray_address, num_nodes)
        _apply_multinode_defaults(config, num_nodes=num_nodes, gpus_per_node=gpus_per_node)
        _clear_external_distributed_env()
        _run_ray_train(config)
    finally:
        _ray_stop()


def _start_rollout_node(ray_address: str, gpus_per_node: int) -> None:
    while True:
        try:
            logger.info(f"Starting rollout node and joining Ray at {ray_address}")
            _ray_start(
                [
                    "--block",
                    "--address",
                    ray_address,
                    "--num-gpus",
                    str(gpus_per_node),
                    "--resources",
                    json.dumps({"rollout_node": 1000}),
                ]
            )
            return
        except subprocess.CalledProcessError:
            logger.info(f"Waiting for Ray head at {ray_address}...")
            time.sleep(5)


def _apply_multinode_defaults(config: dict[str, Any], *, num_nodes: int, gpus_per_node: int) -> None:
    rollout_gpus = _env_int("ROLLOUT_GPUS", max(0, num_nodes - 1) * gpus_per_node)
    train_gpus = _env_int("TRAIN_GPUS", gpus_per_node)
    model_server_replicas = _env_int("MODEL_SERVER_REPLICAS", rollout_gpus)
    rollout_workers = _env_int("ROLLOUT_WORKERS", model_server_replicas)

    rl_config = config.setdefault("trainer_args", {}).setdefault("rl_config", {})
    model_server = rl_config.setdefault("model_server", {})
    model_server["num_replicas"] = model_server_replicas
    model_actor_options = model_server.setdefault("actor_options", {})
    model_actor_options["num_gpus"] = _env_float("MODEL_SERVER_GPUS_PER_REPLICA", 1.0)
    model_actor_options.setdefault("resources", {})["rollout_node"] = 0.001

    rollout = rl_config.setdefault("rollout", {})
    rollout["num_workers"] = rollout_workers
    rollout["max_inflight_per_worker"] = _env_int("ROLLOUT_MAX_INFLIGHT_PER_WORKER", 1)
    rollout.setdefault("actor_options", {}).setdefault("resources", {})["rollout_node"] = 0.001

    data_buffer = rl_config.setdefault("data_buffer", {})
    data_buffer["max_trajectories"] = _env_int("DATA_BUFFER_MAX_TRAJECTORIES", max(1, rollout_workers * 4))
    data_buffer["high_watermark"] = _env_int("DATA_BUFFER_HIGH_WATERMARK", max(1, rollout_workers * 3))
    data_buffer["low_watermark"] = _env_int("DATA_BUFFER_LOW_WATERMARK", max(1, rollout_workers))
    data_buffer["train_batch_size"] = _env_int("TRAIN_BATCH_SIZE", train_gpus)
    data_buffer["min_trajectories_per_batch"] = _env_int("MIN_TRAJECTORIES_PER_BATCH", train_gpus)

    training = rl_config.setdefault("training", {})
    training["global_batch_size"] = _env_int("RL_GLOBAL_BATCH_SIZE", train_gpus)

    ray_train = dict(config.get("ray_train", {}) or {})
    ray_train["num_workers"] = _env_int("RAY_TRAIN_NUM_WORKERS", train_gpus)
    ray_train["use_gpu"] = _env_bool("RAY_TRAIN_USE_GPU", True)
    ray_train["resources_per_worker"] = {
        **dict(ray_train.get("resources_per_worker", {}) or {}),
        "GPU": _env_float("RAY_TRAIN_NUM_GPUS_PER_WORKER", 1.0),
        "train_node": 0.001,
    }
    ray_train["placement_strategy"] = str(ray_train.get("placement_strategy", "PACK"))
    config["ray_train"] = ray_train


def _wait_for_ray_nodes(ray_address: str, expected_nodes: int) -> None:
    if ray.is_initialized():
        ray.shutdown()
    ray.init(address=ray_address, ignore_reinit_error=True)
    deadline = time.time() + _env_int("RAY_WAIT_TIMEOUT", 300)
    try:
        while True:
            alive = [node for node in ray.nodes() if node.get("Alive")]
            if len(alive) >= expected_nodes:
                logger.info(f"Ray cluster ready: {len(alive)}/{expected_nodes} nodes alive.")
                return
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for Ray nodes: {len(alive)}/{expected_nodes} alive.")
            time.sleep(5)
    finally:
        ray.shutdown()


def _is_train_node(master_addr: str, train_node_rank: str) -> bool:
    role = os.environ.get("ROLE", "").lower()
    if role in {"train", "trainer", "head"}:
        return True
    if role in {"rollout", "worker"}:
        return False
    for env_name in ("NODE_RANK", "SLURM_NODEID", "RANK"):
        value = os.environ.get(env_name)
        if value is not None:
            return value == train_node_rank
    return bool(_resolve_addresses(master_addr) & _local_addresses())


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


def _clear_external_distributed_env() -> None:
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


def _ray_train_config(config: dict[str, Any]) -> dict[str, Any]:
    trainer_args = dict(config.get("trainer_args", {}) or {})
    rl_config = dict(trainer_args.get("rl_config", {}) or {})
    return dict(config.get("ray_train", rl_config.get("ray_train", {})) or {})


def _scaling_config(ray_train_config: dict[str, Any]) -> ScalingConfig:
    resources_per_worker = ray_train_config.get("resources_per_worker")
    resources_per_worker = dict(resources_per_worker) if resources_per_worker is not None else None

    gpus_per_worker = os.environ.get("RAY_TRAIN_NUM_GPUS_PER_WORKER")
    if gpus_per_worker is not None:
        resources_per_worker = dict(resources_per_worker or {})
        resources_per_worker["GPU"] = float(gpus_per_worker)

    cpus_per_worker = os.environ.get("RAY_TRAIN_NUM_CPUS_PER_WORKER")
    if cpus_per_worker is not None:
        resources_per_worker = dict(resources_per_worker or {})
        resources_per_worker["CPU"] = float(cpus_per_worker)

    return ScalingConfig(
        num_workers=_env_int("RAY_TRAIN_NUM_WORKERS", ray_train_config.get("num_workers", _default_num_workers())),
        use_gpu=_env_bool("RAY_TRAIN_USE_GPU", ray_train_config.get("use_gpu", True)),
        resources_per_worker=resources_per_worker,
        placement_strategy=str(ray_train_config.get("placement_strategy", "PACK")),
    )


def _run_config(config: dict[str, Any], ray_train_config: dict[str, Any]) -> RunConfig:
    run_config = dict(ray_train_config.get("run_config", {}) or {})
    trainer_args = dict(config.get("trainer_args", {}) or {})
    return RunConfig(
        name=run_config.get("name", trainer_args.get("run_name") or "lmms-engine-rl"),
        storage_path=run_config.get("storage_path"),
        worker_runtime_env=_runtime_env(ray_train_config),
    )


def _runtime_env(ray_train_config: dict[str, Any]) -> dict[str, Any]:
    runtime_env = dict(ray_train_config.get("runtime_env", {}) or {})
    env_vars = dict(runtime_env.get("env_vars", {}) or {})
    env_vars["PYTHONPATH"] = _pythonpath(env_vars.get("PYTHONPATH"))
    runtime_env["env_vars"] = env_vars
    return runtime_env


def _pythonpath(existing: str | None = None) -> str:
    engine_src = Path(__file__).resolve().parents[2]
    lmms_eval_root = engine_src / "lmms-eval"
    current = existing if existing is not None else os.environ.get("PYTHONPATH", "")
    entries = [str(engine_src), str(lmms_eval_root)]
    entries.extend(item for item in current.split(os.pathsep) if item)
    return os.pathsep.join(dict.fromkeys(entries))


def _default_num_workers() -> int:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [device for device in visible_devices.split(",") if device.strip()]
        return max(1, len(devices))
    return 1


def _env_bool(name: str, default: Any) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: Any) -> int:
    value = os.environ.get(name)
    return int(value if value is not None else default)


def _env_float(name: str, default: Any) -> float:
    value = os.environ.get(name)
    return float(value if value is not None else default)


if __name__ == "__main__":
    main()
