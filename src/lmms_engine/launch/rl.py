from __future__ import annotations

import hydra
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from lmms_engine.rl.ray import (
    RayRLMultinodeRuntime,
    default_num_workers,
    use_multinode_default,
)
from lmms_engine.rl.ray.train import run_ray_train
from lmms_engine.utils.logging_utils import setup_distributed_logging


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

    if use_multinode_default():
        RayRLMultinodeRuntime.from_env(default_gpus_per_node=default_num_workers()).run(config, run_ray_train)
        return

    run_ray_train(config)


if __name__ == "__main__":
    main()
