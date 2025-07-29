import os
from copy import deepcopy
from typing import Dict

import datasets
# from datasets import Dataset
from torch.utils.data import Dataset
import torch
from PIL import Image

from lmms_engine.mapping_func import register_dataset
from .config import DatasetConfig

from ..utils.train_utils import TrainUtilities
from .collator import VisionCollator
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from accelerate.state import PartialState
from datasets.distributed import split_dataset_by_node
from ..utils import Logging

@register_dataset("fineweb_edu")
class FinewebEduPretrainDataset(Dataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer_id = config.tokenizer

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
                raw_train_dataset, rank=state.process_index, world_size=state.num_processes,
            )

        self.dataset = raw_train_dataset.map(
            self.tokenize_function, 
            batched=True,     
            remove_columns=raw_train_dataset.column_names
        )
        self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )

    def tokenize_function(self,examples):
        texts = examples["text"]
        return self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.config.packing_length,
            return_attention_mask=True,
            return_special_tokens_mask=False,
        )
    # def __getitem__(self, index):
    #     if self.config.dataset_format == "hf_dataset":
    #         data_dict = self.load_from_hf(self.data_list[index])
    #     else:
    #         raise NotImplementedError
    #     return data_dict
    # Apply tokenization
