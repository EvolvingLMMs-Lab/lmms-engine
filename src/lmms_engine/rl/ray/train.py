from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

# Ray Train v2 currently spawns a detached PlacementGroupCleaner that polls
# Ray State API and can spam warnings under load. Keep the launcher on the
# stable execution path unless the user explicitly opts into v2.
os.environ.setdefault("RAY_TRAIN_V2_ENABLED", "0")

import ray
import torch.distributed as dist
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from lmms_engine.launch.cli import create_train_task, save_config
from lmms_engine.rl.ray.runtime import default_num_workers, env_bool, env_int
from lmms_engine.train import RLTrainRunner
from lmms_engine.utils.logging_utils import setup_distributed_logging


def run_ray_train(config: dict[str, Any]) -> None:
    ray_train_config = _ray_train_config(config)
    runtime_env = _runtime_env(ray_train_config)
    _apply_driver_env(runtime_env)

    ray_init_kwargs = dict(ray_train_config.get("ray_init_kwargs", {}) or {})
    configured_ray_address = ray_init_kwargs.pop("address", None)
    ray_address = os.environ.get("RAY_ADDRESS") or configured_ray_address
    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            ignore_reinit_error=True,
            include_dashboard=env_bool("RAY_DASHBOARD", ray_train_config.get("include_dashboard", False)),
            runtime_env=runtime_env,
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
        num_workers=env_int(
            "RAY_TRAIN_NUM_WORKERS",
            ray_train_config.get("num_workers", default_num_workers()),
        ),
        use_gpu=env_bool("RAY_TRAIN_USE_GPU", ray_train_config.get("use_gpu", True)),
        resources_per_worker=resources_per_worker,
        placement_strategy=str(ray_train_config.get("placement_strategy", "PACK")),
    )


def _run_config(config: dict[str, Any], ray_train_config: dict[str, Any]) -> RunConfig:
    run_config = dict(ray_train_config.get("run_config", {}) or {})
    trainer_args = dict(config.get("trainer_args", {}) or {})
    return RunConfig(
        name=run_config.get("name", trainer_args.get("run_name") or "lmms-engine-rl"),
        storage_path=run_config.get("storage_path"),
    )


def _runtime_env(ray_train_config: dict[str, Any]) -> dict[str, Any]:
    runtime_env = dict(ray_train_config.get("runtime_env", {}) or {})
    env_vars = dict(runtime_env.get("env_vars", {}) or {})
    env_vars["PYTHONPATH"] = _pythonpath(env_vars.get("PYTHONPATH"))
    runtime_env["env_vars"] = env_vars
    return runtime_env


def _apply_driver_env(runtime_env: dict[str, Any]) -> None:
    for key, value in dict(runtime_env.get("env_vars", {}) or {}).items():
        os.environ[str(key)] = str(value)


def _pythonpath(existing: str | None = None) -> str:
    engine_src = Path(__file__).resolve().parents[3]
    lmms_eval_root = engine_src / "lmms-eval"
    current = existing if existing is not None else os.environ.get("PYTHONPATH", "")
    entries = [str(engine_src), str(lmms_eval_root)]
    entries.extend(item for item in current.split(os.pathsep) if item)
    return os.pathsep.join(dict.fromkeys(entries))
