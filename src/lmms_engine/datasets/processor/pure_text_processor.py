from typing import List

import torch
from PIL import Image
from transformers import AutoTokenizer

from lmms_engine.mapping_func import register_processor

from .config import ProcessorConfig

# from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessorKwargs


@register_processor("pure_text")
class PureTextDataProcessor:
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config
        self.tokenizer = None

    def build(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.processor_name)

    def __call__(
        self,
        texts,
        truncation=True,
        padding=False,
        max_length=2048,
        return_attention_mask=True,
        return_special_tokens_mask=False,
    ):
        return self.tokenizer(
            texts,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            return_special_tokens_mask=return_special_tokens_mask,
        )
