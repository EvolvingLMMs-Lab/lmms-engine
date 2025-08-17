import os
from copy import deepcopy
from typing import Dict

import datasets
import torch
from accelerate.state import PartialState
from datasets.distributed import split_dataset_by_node
from PIL import Image

# from datasets import Dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from lmms_engine.mapping_func import register_dataset

from ..utils import Logging
from ..utils.train_utils import TrainUtilities
from .collator.text_dllm_collator import TextDllmCollator
from .config import DatasetConfig


@register_dataset("fineweb_edu_dllm")
class FinewebEduDllmDataset(Dataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer_id = config.tokenizer
        self.processor = None
        self.p_min = 0.01
        self.p_max = 0.99

    def get_collator(self):
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        """
        Strictly speaking, the shape of the embedding needs to be resized. 
        However, in most models, a portion of the embedding dim is reserved for newly added tokens, 
        so resize is omitted here
        """
        collator = TextDllmCollator(
            p_min=self.p_min,
            p_max=self.p_max,
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return collator

    def build(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
            Logging.info(f"Tokenizer {self.tokenizer_id} is loaded.")
        except Exception as e:
            raise ValueError(f"Tokenizer {self.tokenizer_id} not found")

        state = PartialState()
        with state.main_process_first():
            raw_train_dataset = datasets.load_dataset(
                self.config.dataset_path,
                "default",
                split="train",
                streaming=True,
            )

        raw_train_dataset = split_dataset_by_node(
            raw_train_dataset,
            rank=state.process_index,
            world_size=state.num_processes,
        )

        self.dataset = raw_train_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=raw_train_dataset.column_names,
        )

    def _tokenize_function(self, examples):
        texts = examples["text"]
        return self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.config.packing_length,
            return_attention_mask=True,
            return_special_tokens_mask=False,
        )
