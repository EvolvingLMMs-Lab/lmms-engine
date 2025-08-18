import itertools
import os
from copy import deepcopy
from typing import Dict

import datasets
import torch
from accelerate.state import PartialState
from datasets.distributed import split_dataset_by_node
from PIL import Image

# from datasets import Dataset
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from lmms_engine.mapping_func import DATAPROCESSOR_MAPPING, register_dataset

from ..utils import Logging
from ..utils.train_utils import TrainUtilities
from .base_dataset import BaseDataset
from .collator.text_dllm_collator import TextDllmCollator
from .config import DatasetConfig
from .processor import ProcessorConfig


@register_dataset("fineweb_edu")
class FinewebEduDataset(IterableDataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.processor_config = config.processor_config
        if isinstance(self.processor_config, dict):
            self.processor_config = ProcessorConfig(**self.processor_config)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)
        return iter_data

    def get_collator(self):
        if self.processor.tokenizer.mask_token is None:
            self.processor.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        """
        Strictly speaking, the shape of the embedding needs to be resized. 
        However, in most models, a portion of the embedding dim is reserved for newly added tokens, 
        so resize is omitted here
        """
        collator = TextDllmCollator(
            p_min=self.p_min,
            p_max=self.p_max,
            tokenizer=self.processor.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return collator

    def _build_processor(self):
        processor_cls = DATAPROCESSOR_MAPPING[self.processor_config.processor_type]
        processor = processor_cls(self.processor_config)
        return processor

    def build(self):
        self._build_from_config()
        self.processor = self._build_processor()
        if self.processor is not None:
            self.processor.build()

    def _build_from_config(self):
        raw_train_dataset = datasets.load_dataset(
            self.config.dataset_path,
            "default",
            split="train",
            streaming=True,
        )

        def _process(examples):
            return self.processor(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.packing_length,
                return_attention_mask=True,
                return_special_tokens_mask=False,
            )

        self.data = raw_train_dataset.map(
            _process,
            batched=True,
            remove_columns=raw_train_dataset.column_names,
        )
        self.p_min = 0.01
        self.p_max = 0.99
