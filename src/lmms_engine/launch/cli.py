import argparse
import datetime
import os

import torch.distributed as dist

from lmms_engine.parallel.process_group_manager import setup_process_group_manager

from ..datasets import DatasetConfig
from ..models import ModelConfig
from ..train import TrainerConfig, TrainingArguments, TrainRunner
from ..utils.config_loader import load_config


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to your launch config")
    return parser.parse_args()


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
    if not dist.is_initialized():
        dist.init_process_group(
            rank=global_rank,
            world_size=world_size,
            backend=backend,
            init_method=f"env://",
            timeout=datetime.timedelta(minutes=30),
        )
    setup_process_group_manager(
        tp_size=1, cp_size=sp_degree, pp_size=1, dp_size=dp_size
    )

    # Extract trainer args from config, handling nested structure
    # trainer_args_dict = config.pop("trainer_args", {})
    # # If trainer_args is empty, use remaining config as trainer args
    # if not trainer_args_dict:
    #     trainer_args_dict = config.copy()
    #     # Remove non-trainer argument keys
    #     for key in ["sp_ulysses_degree", "use_cpu"]:
    #         trainer_args_dict.pop(key, None)

    trainer_args = TrainingArguments(**config)

    train_config = TrainerConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        trainer_type=trainer_type,
        trainer_args=trainer_args,
    )
    return TrainRunner(config=train_config)


def main():
    args = parse_argument()
    configs = load_config(args.config)

    for config in configs:
        task_type = config.pop("task_type", "trainer")
        task_config = config.pop("config", {})
        if task_type == "trainer":
            task = create_train_task(task_config)
            task.build()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        task.run()


if __name__ == "__main__":
    main()
