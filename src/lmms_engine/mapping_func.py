from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    # AutoModelForVision2Seq,
    AutoModelForImageTextToText,
    PretrainedConfig,
)
from transformers.modeling_utils import PreTrainedModel

DATASET_MAPPING = {}
DATAPROCESSOR_MAPPING = {}
from lmms_engine import TRANSFORMERS_MODEL_REGISTERED, FLA_MODEL_REGISTERED


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
    model_type: str, model_config: PretrainedConfig, model_class: PreTrainedModel
):
    AutoConfig.register(model_type, model_config)
    AutoModelForCausalLM.register(model_config, model_class)


def create_model_from_pretrained(load_from_pretrained_path):
    # Handle both config object and model name/path
    config = AutoConfig.from_pretrained(load_from_pretrained_path)
    if type(config) in AutoModelForCausalLM._model_mapping.keys():
        model_class = AutoModelForCausalLM
    elif type(config) in AutoModelForImageTextToText._model_mapping.keys():
        model_class = AutoModelForImageTextToText
    else:
        raise ValueError(f"Model: '{load_from_pretrained_path}' is not supported.")
    return model_class

def create_model_from_config(model_type, config):
    if model_type.lower() in FLA_MODEL_REGISTERED:
        try: 
            import fla
        except ImportError:
            raise ImportError("`import fla` failed. Please install fla first to use this model.")
        config_class = getattr(fla.models, FLA_MODEL_REGISTERED[model_type.lower()], None)
        assert config_class is not None, f"We did not find the model type: {model_type} in fla.models."
    elif model_type.lower() in TRANSFORMERS_MODEL_REGISTERED:
        try:
            import transformers
        except ImportError:
            raise ImportError("`import transformers` failed. Please install transformers first to use this model.")
        config_class = getattr(transformers, TRANSFORMERS_MODEL_REGISTERED[model_type.lower()], None)
        assert config_class is not None, f"We did not find the model type: {model_type} in transformers."
    else:
        raise ValueError(f"Currently, we only support these models: {FLA_MODEL_REGISTERED.keys()} and {TRANSFORMERS_MODEL_REGISTERED.keys()}")

    try: 
        m_config = config_class(**config)
    except Exception as e:
        raise ValueError(f"Error creating model from config: {e}. Please check the config is correct.")
    
    if type(m_config) in AutoModelForCausalLM._model_mapping.keys():
        model_class = AutoModelForCausalLM
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")
    return model_class, m_config
    
