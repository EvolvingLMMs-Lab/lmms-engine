import json
import os
from typing import Literal

from pathlib import Path
import json5
from transformers import (  # AutoModelForVision2Seq,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMaskedLM,
    PretrainedConfig,
)
from transformers.modeling_utils import PreTrainedModel

from lmms_engine.utils import Logging

DATASET_MAPPING = {}
DATAPROCESSOR_MAPPING = {}


from lmms_engine.utils import Logging

try:
    import fla
except ImportError as e:
    Logging.warning(
        f"Failed to import the lib 'fla'. If you do not need it, you can ignore this warning."
    )


# A decorator class to register processors
def register_processor(processor_type: str):
    def decorator(cls):
        if processor_type in DATAPROCESSOR_MAPPING:
            raise ValueError(f"Processor type {processor_type} is already registered.")
        DATAPROCESSOR_MAPPING[processor_type] = cls
        return cls

    return decorator


# A decorator class to register dataset
def register_dataset(dataset_type: str):
    def decorator(cls):
        if dataset_type in DATASET_MAPPING:
            raise ValueError(f"Dataset type {dataset_type} is already registered.")
        DATASET_MAPPING[dataset_type] = cls
        return cls

    return decorator


AUTO_REGISTER_MODEL_MAPPING = {
    "causal_lm": AutoModelForCausalLM,
    "masked_lm": AutoModelForMaskedLM,
    "image_text_to_text": AutoModelForImageTextToText,
    "general": AutoModel,
}


def register_model(
    model_type: str,
    model_config: PretrainedConfig,
    model_class: PreTrainedModel,
    model_general_type: Literal[
        "causal_lm", "masked_lm", "image_text_to_text", "general"
    ] = "causal_lm",
):
    AutoConfig.register(model_type, model_config)
    AUTO_REGISTER_MODEL_MAPPING[model_general_type].register(model_config, model_class)


def create_model_from_pretrained(load_from_pretrained_path):
    # Handle both config object and model name/path
    try:
        config = AutoConfig.from_pretrained(load_from_pretrained_path)
    except Exception:
        # really weird, but some models' config.json (Bagel) is not valid json
        Logging.warning(f"Model {load_from_pretrained_path} config.json is not valid json, trying to fix it.")
        for file in Path(load_from_pretrained_path).iterdir():
            if file.is_file() and file.suffix == ".json":
                config_path = file
            with open(config_path, "r") as f:
                json5_dict = json5.load(f)
            if file.name == "config.json" and "BAGEL-7B-MoT" in json5_dict.get("name", []) and "model_type" not in json5_dict:
                json5_dict["model_type"] = "bagel"
            with open(config_path, "w") as f:
                json.dump(json5_dict, f, indent=2)
            config = AutoConfig.from_pretrained(load_from_pretrained_path)
    if type(config) in AutoModelForCausalLM._model_mapping.keys():
        model_class = AutoModelForCausalLM
    elif type(config) in AutoModelForImageTextToText._model_mapping.keys():
        model_class = AutoModelForImageTextToText
    elif type(config) in AutoModelForMaskedLM._model_mapping.keys():
        model_class = AutoModelForMaskedLM
    elif type(config) in AutoModel._model_mapping.keys():
        model_class = AutoModel
    else:
        raise ValueError(f"Model {load_from_pretrained_path} is not supported.")
    return model_class


def create_model_from_config(model_type, config):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if model_type in CONFIG_MAPPING:
        config_class = CONFIG_MAPPING[model_type]
        m_config = config_class(**config)
        if type(m_config) in AutoModelForCausalLM._model_mapping.keys():
            model_class = AutoModelForCausalLM
        elif type(m_config) in AutoModelForImageTextToText._model_mapping.keys():
            model_class = AutoModelForImageTextToText
        elif type(m_config) in AutoModelForMaskedLM._model_mapping.keys():
            model_class = AutoModelForMaskedLM
        elif type(m_config) in AutoModel._model_mapping.keys():
            model_class = AutoModel
    else:
        raise ValueError(f"Model type '{model_type}' is not found in CONFIG_MAPPING.")
    return model_class, m_config
