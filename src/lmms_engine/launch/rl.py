from __future__ import annotations

import sys
from pathlib import Path

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from lmms_engine.rl.ray import RayRLMultinodeRuntime, use_multinode_config
from lmms_engine.rl.ray.train import run_ray_train
from lmms_engine.utils.logging_utils import setup_distributed_logging


@hydra.main(version_base=None, config_path="config", config_name="default_config")
def main(config: DictConfig):
    setup_distributed_logging()
    config = _compose_config(config)
    runtime_config = dict(config.get("rl_runtime", {}) or {})

    if use_multinode_config(runtime_config):
        RayRLMultinodeRuntime.from_config(runtime_config).run(config, run_ray_train)
        return

    run_ray_train(config)


def _compose_config(config: DictConfig) -> dict:
    """Compose RL config as default < config_yaml < CLI overrides."""

    raw_config = yaml.safe_load(OmegaConf.to_yaml(config))
    config_yaml = raw_config.get("config_yaml")
    if not config_yaml:
        raw_config.pop("config_yaml", None)
        return raw_config

    config_path = _resolve_config_yaml(config_yaml)
    logger.info(f"Detected config yaml, merging with the default config: {config_path}")

    default_config = OmegaConf.load(Path(__file__).resolve().parent / "config" / "default_config.yaml")
    with open(config_path, "r") as f:
        file_config = OmegaConf.create(yaml.safe_load(f) or {})

    merged = OmegaConf.merge(default_config, file_config)
    overrides = _cli_overrides_without_config_yaml()
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    output = OmegaConf.to_container(merged, resolve=True)
    output.pop("config_yaml", None)
    return output


def _resolve_config_yaml(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    original_cwd = Path(get_original_cwd())
    resolved = original_cwd / candidate
    if resolved.exists():
        return resolved
    return Path.cwd() / candidate


def _cli_overrides_without_config_yaml() -> list[str]:
    try:
        overrides = list(HydraConfig.get().overrides.task)
    except ValueError:
        return []
    return [override for override in overrides if not override.lstrip("+").startswith("config_yaml=")]


def _normalize_runtime_cli_flags(argv: list[str]) -> None:
    flag_to_key = {
        "--nnodes": "rl_runtime.nnodes",
        "--gpus-per-node": "rl_runtime.gpus_per_node",
        "--master-addr": "rl_runtime.master_addr",
        "--ray-port": "rl_runtime.ray_port",
        "--node-rank": "rl_runtime.node_rank",
        "--train-node-rank": "rl_runtime.train_node_rank",
        "--head-node-ip": "rl_runtime.head_node_ip",
        "--ray-wait-timeout": "rl_runtime.wait_timeout",
    }
    normalized = [argv[0]]
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--no-multinode":
            normalized.append("rl_runtime.enabled=false")
            idx += 1
            continue
        if arg == "--multinode":
            normalized.append("rl_runtime.enabled=true")
            idx += 1
            continue

        key = None
        value = None
        if "=" in arg:
            flag, maybe_value = arg.split("=", 1)
            key = flag_to_key.get(flag)
            if key is not None:
                value = maybe_value
        else:
            key = flag_to_key.get(arg)
            if key is not None:
                if idx + 1 >= len(argv):
                    raise SystemExit(f"{arg} requires a value")
                value = argv[idx + 1]
                idx += 1

        normalized.append(f"{key}={value}" if key is not None else arg)
        idx += 1
    argv[:] = normalized


if __name__ == "__main__":
    _normalize_runtime_cli_flags(sys.argv)
    main()
