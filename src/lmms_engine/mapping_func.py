from transformers import (  # AutoModelForVision2Seq,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMaskedLM,
    PretrainedConfig,
)
from transformers.modeling_utils import PreTrainedModel

DATASET_MAPPING = {}
DATAPROCESSOR_MAPPING = {}
from lmms_engine.utils import Logging

try:
    import fla
except ImportError as e:
    Logging.warning(f"Failed to import fla.")


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


def register_model(
    model_type: str,
    model_config: PretrainedConfig,
    model_class: PreTrainedModel,
    is_masked_lm: bool = False,
):
    AutoConfig.register(model_type, model_config)
    if is_masked_lm:
        AutoModelForMaskedLM.register(model_config, model_class)
    else:
        AutoModelForCausalLM.register(model_config, model_class)


def create_model_from_pretrained(load_from_pretrained_path):
    # Handle both config object and model name/path
    config = AutoConfig.from_pretrained(load_from_pretrained_path)
    if type(config) in AutoModelForCausalLM._model_mapping.keys():
        model_class = AutoModelForCausalLM
    elif type(config) in AutoModelForImageTextToText._model_mapping.keys():
        model_class = AutoModelForImageTextToText
    elif type(config) in AutoModelForMaskedLM._model_mapping.keys():
        model_class = AutoModelForMaskedLM
    else:
        raise ValueError(f"Model {load_from_pretrained_path} is not supported.")
    return model_class


def create_model_from_config(model_type, config):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config_class = CONFIG_MAPPING[model_type]
    m_config = config_class(**config)
    if type(m_config) in AutoModelForCausalLM._model_mapping.keys():
        model_class = AutoModelForCausalLM
    elif type(m_config) in AutoModelForImageTextToText._model_mapping.keys():
        model_class = AutoModelForImageTextToText
    elif type(m_config) in AutoModelForMaskedLM._model_mapping.keys():
        model_class = AutoModelForMaskedLM
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")
    return model_class, m_config
