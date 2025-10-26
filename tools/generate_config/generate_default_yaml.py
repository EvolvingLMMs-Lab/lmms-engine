import argparse
import os

import yaml

from lmms_engine.datasets import DatasetConfig
from lmms_engine.datasets.processor import ProcessorConfig
from lmms_engine.models import ModelConfig
from lmms_engine.train import TrainerConfig, TrainingArguments


def generate_default_yaml(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    processor_config = ProcessorConfig(processor_name=model_name, processor_type="qwen2_5_vl")
    model_config = ModelConfig(load_from_pretrained_path=model_name)
    dataset_config = DatasetConfig(
        dataset_type="vision_iterable",
        dataset_format="yaml",
        processor_config=processor_config,
    )
    trainer_args = TrainingArguments()
    trainer_config = TrainerConfig(
        trainer_type="fsdp2_trainer",
        dataset_config=dataset_config,
        trainer_args=trainer_args,
        model_config=model_config,
        extra_kwargs={},
    )

    yaml_config = trainer_config.to_dict()
    return yaml_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate default YAML config")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name to use in config (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    args = parser.parse_args()

    # Generate config
    config = generate_default_yaml(model_name=args.model_name)

    # Define output path relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "../../src/lmms_engine/launch/config/default_config.yaml")
    output_path = os.path.abspath(output_path)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    config["config_yaml"] = None
    config["trainer_args"]["fsdp_config"].update(
        {
            "transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer"],
            "reshard_after_forward": False,
        }
    )
    # Deprecated, pop to remove warning
    config["trainer_args"].pop("push_to_hub_token")

    # Save config to yaml
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config saved to: {output_path}")
