import argparse
import datetime
import os
import shutil
from copy import deepcopy

import hydra
import torch.distributed as dist
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from lmms_engine.parallel.process_group_manager import setup_process_group_manager
from lmms_engine.utils.logging_utils import setup_distributed_logging

from ..datasets import DatasetConfig
from ..eval import EvalConfig
from ..models import ModelConfig
from ..train import TrainerConfig, TrainingArguments, TrainRunner


def create_train_task(config):
    dataset_config = config.pop("dataset_config")
    dataset_config = DatasetConfig(**dataset_config)

    model_config = config.pop("model_config")
    model_config = ModelConfig(**model_config)

    trainer_type = config.pop("trainer_type")
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    trainer_args = config.get("trainer_args")
    sp_degree = trainer_args.get("sp_ulysses_degree", 1)
    ep_degree = trainer_args.get("ep_degree", 1)
    # DP size actually will not be affected by ep_degree, but kept for initialization here
    dp_size = world_size // sp_degree

    # For now, we haven't implement the tp and pp
    use_cpu = trainer_args.get("use_cpu", False)
    backend = "gloo" if use_cpu else "nccl"
    # If the process group is already initialized, don't initialize it again
    ddp_timeout = trainer_args.get("ddp_timeout", 30 * 60)
    if not dist.is_initialized():
        # For single GPU without distributed launcher, set required env vars
        if world_size == 1 and "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_RANK"] = "0"

        dist.init_process_group(
            rank=global_rank,
            world_size=world_size,
            backend=backend,
            init_method=f"env://",
            timeout=datetime.timedelta(seconds=ddp_timeout),
        )
    setup_process_group_manager(tp_size=1, cp_size=sp_degree, pp_size=1, dp_size=dp_size, ep_size=ep_degree)

    trainer_args = config.pop("trainer_args")

    eval_config_dict = trainer_args.pop("eval_config", None)
    if eval_config_dict is not None:
        eval_config = EvalConfig(**eval_config_dict)
        trainer_args["eval_config"] = eval_config.to_dict()

    trainer_args = TrainingArguments(**trainer_args)

    train_config = TrainerConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        trainer_type=trainer_type,
        trainer_args=trainer_args,
    )
    return TrainRunner(config=train_config)


def save_config(config):
    if dist.is_initialized():
        rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
    if rank == 0:
        data_config = config.get("dataset_config")
        trainer_args = config.get("trainer_args")
        output_dir = trainer_args.get("output_dir")
        data_type = data_config.get("dataset_type")
        os.makedirs(output_dir, exist_ok=True)
        if data_type == "yaml":
            dataset_path = data_config.get("dataset_path")
            shutil.copy(dataset_path, os.path.join(output_dir, "dataset.yaml"))

        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    if dist.is_initialized():
        dist.barrier(device_ids=[rank])


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
        with open(config_yaml, "r") as f:
            config_yaml = yaml.safe_load(f)
        config.update(config_yaml)
    original_config = deepcopy(config)
    task = create_train_task(config)
    save_config(original_config)
    task.build()
    task.run()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
