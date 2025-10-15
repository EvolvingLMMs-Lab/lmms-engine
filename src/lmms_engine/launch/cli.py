import argparse
import datetime
import os

import hydra
import torch.distributed as dist
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from lmms_engine.parallel.process_group_manager import setup_process_group_manager
from lmms_engine.utils.logging_utils import setup_distributed_logging

from ..datasets import DatasetConfig
from ..models import ModelConfig
from ..train import TrainerConfig, TrainingArguments, TrainRunner


def create_train_task(config):
    dataset_config = config.pop("dataset_config")
    dataset_config = DatasetConfig(**dataset_config)

    model_config = config.pop("model_config")
    model_config = ModelConfig(**model_config)

    trainer_type = config.pop("trainer_type")
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    sp_degree = config.get("sp_ulysses_degree", 1)
    dp_size = world_size // sp_degree

    # For now, we haven't implement the tp and pp
    use_cpu = config.get("use_cpu", False)
    backend = "gloo" if use_cpu else "nccl"
    # If the process group is already initialized, don't initialize it again
    ddp_timeout = config.get("ddp_timeout", 30 * 60)
    if not dist.is_initialized():
        dist.init_process_group(
            rank=global_rank,
            world_size=world_size,
            backend=backend,
            init_method=f"env://",
            timeout=datetime.timedelta(seconds=ddp_timeout),
        )
    setup_process_group_manager(
        tp_size=1, cp_size=sp_degree, pp_size=1, dp_size=dp_size
    )

    trainer_args = config.pop("trainer_args")
    trainer_args = TrainingArguments(**trainer_args)

    train_config = TrainerConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        trainer_type=trainer_type,
        trainer_args=trainer_args,
    )
    return TrainRunner(config=train_config)


@hydra.main(version_base=None, config_path="config", config_name="default_config")
def main(config: DictConfig):
    setup_distributed_logging()
    config = OmegaConf.to_yaml(config)
    config = yaml.safe_load(config)

    # If you have a predefined config yaml
    config_yaml = config.pop("config_yaml")
    if config_yaml:
        logger.info(
            f"Detected config yaml, merging with the default config. Will use the args in {config_yaml} to override current config."
        )
        config_yaml = yaml.safe_load(config_yaml)
        config.update(config_yaml)
    task = create_train_task(config)
    task.build()
    task.run()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
