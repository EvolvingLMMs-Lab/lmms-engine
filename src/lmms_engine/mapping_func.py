from typing import Literal

from transformers import (  # AutoModelForVision2Seq,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMaskedLM,
    PretrainedConfig,
)
from transformers.modeling_utils import PreTrainedModel

DATASET_MAPPING = {}
DATAPROCESSOR_MAPPING = {}

from loguru import logger

try:
    import fla
except ImportError as e:
    logger.warning(f"Failed to import the lib 'fla'. If you do not need it, you can ignore this warning.")


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
    model_general_type: Literal["causal_lm", "masked_lm", "image_text_to_text", "general"] = "causal_lm",
):
    AutoConfig.register(model_type, model_config, exist_ok=True)
    AUTO_REGISTER_MODEL_MAPPING[model_general_type].register(model_config, model_class)


def create_model_from_pretrained(
    load_from_pretrained_path,
    model_general_type: str | None = None,
    trust_remote_code: bool = False,
):
    """Pick an HF Auto* class for ``load_from_pretrained_path``.

    Args:
        load_from_pretrained_path: HF hub id or local path.
        model_general_type: Optional override; one of the keys in
            ``AUTO_REGISTER_MODEL_MAPPING`` (``"causal_lm"``,
            ``"image_text_to_text"``, ``"masked_lm"``, ``"general"``). Use it
            to disambiguate when the same config is registered under multiple
            AutoModel mappings (e.g. Qwen3.5 registers under both
            ``causal_lm`` and ``image_text_to_text``).
        trust_remote_code: forwarded to ``AutoConfig.from_pretrained``; needed
            for checkpoints that ship custom modeling code via ``auto_map``.
    """
    # Handle both config object and model name/path
    config = AutoConfig.from_pretrained(load_from_pretrained_path, trust_remote_code=trust_remote_code)

    if model_general_type is not None:
        if model_general_type not in AUTO_REGISTER_MODEL_MAPPING:
            raise ValueError(
                f"Unknown model_general_type={model_general_type!r}; "
                f"choose one of {list(AUTO_REGISTER_MODEL_MAPPING)}"
            )
        return AUTO_REGISTER_MODEL_MAPPING[model_general_type]

    if type(config) in AutoModelForCausalLM._model_mapping.keys():
        model_class = AutoModelForCausalLM
    elif type(config) in AutoModelForImageTextToText._model_mapping.keys():
        model_class = AutoModelForImageTextToText
    elif type(config) in AutoModelForMaskedLM._model_mapping.keys():
        model_class = AutoModelForMaskedLM
    elif type(config) in AutoModel._model_mapping.keys():
        model_class = AutoModel
    else:
        # Fallback for trust_remote_code checkpoints: the config class is loaded
        # dynamically via auto_map and won't be in any HF model mapping. Pick
        # the AutoModelFor* class declared in the config's auto_map; that class
        # will resolve the remote modeling code itself when from_pretrained
        # is called with trust_remote_code=True.
        auto_map = getattr(config, "auto_map", None) or {}
        if "AutoModelForImageTextToText" in auto_map:
            model_class = AutoModelForImageTextToText
        elif "AutoModelForCausalLM" in auto_map:
            model_class = AutoModelForCausalLM
        elif "AutoModelForMaskedLM" in auto_map:
            model_class = AutoModelForMaskedLM
        elif "AutoModel" in auto_map:
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
