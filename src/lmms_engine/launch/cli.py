import argparse

from ..datasets import DatasetConfig
from ..models import ModelConfig
from ..train import Hf_Trainer, TrainerConfig, TrainingArguments
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

    trainer_args_type = config.pop("trainer_args_type", "sft")
    trainer_args = TrainingArguments(**config)

    train_config = TrainerConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        trainer_type=trainer_type,
        trainer_args=trainer_args,
        trainer_args_type=trainer_args_type,
    )
    return Hf_Trainer(config=train_config)


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
