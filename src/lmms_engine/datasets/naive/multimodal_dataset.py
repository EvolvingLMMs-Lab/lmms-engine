import os
import random
from copy import deepcopy
from typing import List

import torch.distributed as dist
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from loguru import logger
from tqdm import tqdm

from lmms_engine.datasets.multimodal_mixin import MultiModalDataLoadingMixin
from lmms_engine.utils import DataUtilities

from .base_dataset import BaseDataset

try:
    from google.cloud.storage import Client
except ImportError:
    logger.info("Google Cloud SDK not installed. Skipping import.")

try:
    from azure.storage.blob import BlobServiceClient, LinearRetry

    RETRY_POLICY = LinearRetry(backoff=10, retry_total=5, random_jitter_range=0)
    SAS_URL = os.environ.get("AZURE_STORAGE_SAS_URL", "YOUR_SAS_URL")
except ImportError:
    logger.info("Azure SDK not installed. Skipping import.")


class MultiModalDataset(BaseDataset, MultiModalDataLoadingMixin):
    """
    MultiModalDataset provides concrete implementation for handling multimodal data
    including images, audio, and videos with support for various data formats and
    object storage backends.

    This class inherits from BaseDataset and implements all the abstract methods
    with full functionality for data loading, processing, and packing.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        # Initialize object storage clients if needed
        if self.config.object_storage == "gcs":
            self.storage_client = Client()
            self.bucket_name = self.config.bucket_name
        elif self.config.object_storage == "azure":
            self.storage_client = BlobServiceClient(account_url=SAS_URL, retry_policy=RETRY_POLICY)
            self.bucket_name = self.config.bucket_name

    def filter_overlong(self):
        """Filter out data samples that are too long for packing."""
        if self.config.filter_overlong:
            if not self.config.packing and self.config.max_length is None:
                # If not packing and max length is not specified, we don't need to filter overlong
                return
            max_length = self.config.max_length if self.config.max_length is not None else self.config.packing_length
            logger.info(f"Filter overlong data, max length: {max_length}")
            original_length = len(self.data_list)
            seq_len = max_length
            overlong_indices = [i for i, length in enumerate(self.data_lengths) if length > seq_len]
            overlong_indices = set(overlong_indices)
            total_indices = set(range(len(self.data_list)))
            select_indices = total_indices - overlong_indices
            if isinstance(self.data_list, HFDataset):
                self.data_list = self.data_list.select(select_indices)
            else:
                self.data_list = [self.data_list[i] for i in range(len(self.data_list)) if i not in overlong_indices]
            if getattr(self, "data_folder", None) is not None:
                self.data_folder = [self.data_folder[i] for i in select_indices]
            self.data_lengths = [self.data_lengths[i] for i in select_indices]
            logger.info(
                f"Filter overlong data done, original length: {original_length}, new length: {len(self.data_list)}"
            )

    def _build_from_config(self):
        """Load and prepare data from the configuration."""
        if self.config.dataset_format == "json":
            self.data_list = DataUtilities.load_json(self.config.dataset_path)
        elif self.config.dataset_format == "jsonl":
            self.data_list = DataUtilities.load_jsonlines(self.config.dataset_path)
        elif self.config.dataset_format == "arrow":
            self.data_list = load_from_disk(self.config.dataset_path)
        elif self.config.dataset_format == "parquet":
            self.data_list = HFDataset.from_parquet(self.config.dataset_path)
        elif self.config.dataset_format == "hf_dataset":
            self.data_list = load_dataset(self.config.dataset_path, split="train")
            self.data_list_no_image = deepcopy(self.data_list)
            self.data_list_no_image = self.data_list_no_image.remove_columns("image")
        elif self.config.dataset_format == "yaml":
            # Handle both external YAML files and inline datasets
            if self.config.datasets is not None:
                # Use inline datasets defined in the config
                self.data_list, self.data_folder = DataUtilities.load_inline_datasets(self.config.datasets)
            elif self.config.dataset_path is not None:
                # Load from external YAML file
                self.data_list, self.data_folder = DataUtilities.load_yaml(self.config.dataset_path)
            else:
                raise ValueError("For yaml format, either 'datasets' or 'dataset_path' must be provided")
        else:
            raise NotImplementedError

        if self.config.shuffle:
            logger.info("Shuffle Dataset ...")
            data_index = [i for i in range(len(self.data_list))]
            random.shuffle(data_index)
            if isinstance(self.data_list, HFDataset):
                self.data_list = self.data_list.select(data_index)
            else:
                self.data_list = [self.data_list[i] for i in data_index]
            if getattr(self, "data_folder", None) is not None:
                self.data_folder = [self.data_folder[i] for i in data_index]

        if isinstance(self.data_list, HFDataset):
            self.data_lengths = self.data_list.map(
                lambda x: {"length": self.estimate_data_tokens_per_row(x)},
                num_proc=self.config.filter_overlong_workers,
            ).select_columns("length")
            self.data_lengths = self.data_lengths.to_list()
            self.data_lengths = [da["length"] for da in self.data_lengths]
        else:
            self.data_lengths = (
                self._estimate_data_tokens(self.data_list)
                if self.config.dataset_format != "hf_dataset"
                else self.data_list_no_image
            )
        self.filter_overlong()

        if self.config.packing:
            if self.config.packing_strategy is None:
                raise ValueError("Packing strategy is not specified.")
            packing_length = self.config.packing_length
            if self.config.packing_strategy == "first_fit":
                self.packing_index = self._pack_by_first_fit(self.data_lengths, packing_length)
            elif "window" in self.config.packing_strategy:
                window_size = int(self.config.packing_strategy.split("_")[1])
                self.packing_index = self._pack_by_window(self.data_lengths, packing_length, window_size)
            else:
                raise NotImplementedError
            logger.info(f"Before packing : {len(self.data_list)}, After packing : {len(self.packing_index)}")

    def estimate_data_tokens_per_row(self, row):
        """
        Estimate the number of tokens in a data row.

        Args:
            row: Data row containing messages

        Returns:
            Estimated token count
        """
        messages = row["messages"]
        cur_len = 0
        for message in messages:
            content = message["content"]
            for cont in content:
                precomputed_tokens = getattr(cont, "precomputed_tokens", None)
                # In case arrow where every place has a field
                if cont["type"] == "image_url":
                    if precomputed_tokens is not None:
                        cur_len += precomputed_tokens
                    else:
                        cur_len += 2000
                elif cont["type"] == "audio_url":
                    if precomputed_tokens is not None:
                        cur_len += precomputed_tokens
                    else:
                        cur_len += 750
                elif cont["type"] == "video_url":
                    if precomputed_tokens is not None:
                        cur_len += precomputed_tokens
                    else:
                        cur_len += 5000
                elif cont["type"] == "text":
                    cur_len += len(cont["text"].split()) * 1.5
                    if "audio_text" in cont:
                        cur_len = max(cur_len, len(cont["text"]))
                else:
                    raise TypeError(f"Encountered invalid content type {cont['type']}")
        return cur_len

    def _estimate_data_tokens(self, data_list):
        """
        Estimate token counts for a list of data samples.

        Args:
            data_list: List of data samples

        Returns:
            List of estimated token counts
        """
        lengths = []
        pbar = tqdm(
            total=len(data_list),
            desc="Estimating data tokens...",
            disable=dist.get_rank() != 0,
        )
        for data in data_list:
            cur_len = self.estimate_data_tokens_per_row(data)
            lengths.append(cur_len)
            pbar.update(1)
        pbar.close()
        return lengths

    def _pack_by_first_fit(self, lengths: List[int], packing_length: int):
        """
        Pack data using first-fit strategy.

        Args:
            lengths: List of sample lengths
            packing_length: Maximum packing length

        Returns:
            List of packed index groups
        """
        max_length = packing_length
        logger.info(f"Packing inputs...pack max length: {max_length}")

        result = []
        current_concatenated_length = 0
        current_list = []
        for i, cur_length in enumerate(lengths):
            if cur_length + current_concatenated_length <= max_length:
                current_concatenated_length += cur_length
                current_list.append(i)
            else:  # current_list is done, create a new one
                if len(current_list) > 0:
                    result.append(current_list)
                current_list = [i]
                current_concatenated_length = cur_length

        if len(current_list) > 0:
            result.append(current_list)

        # assert to make sure no indices were missing
        assert sum([len(indices) for indices in result]) == len(lengths)
        return result

    def _pack_by_window(
        self,
        lengths: List[int],
        packing_length: int,
        window_size: int = 100,
        control_threshold: float = 1,
        max_size: int = -1,
    ):
        """
        Pack data using window strategy.

        Args:
            lengths: List of sample lengths
            packing_length: Maximum packing length
            window_size: Size of the sliding window
            control_threshold: Threshold for packing control
            max_size: Maximum size of each pack

        Returns:
            List of packed index groups
        """
        max_length = packing_length
        logger.info(f"Packing inputs...pack length:{max_length}")
        result = []
        current_concatenated_length = 0
        current_list = []
        i = 0
        cur_window = {}
        next_window = {}
        for k in range(window_size):
            next_window[f"{k}"] = lengths[k]
        while i < len(lengths):
            cur_window = next_window
            next_window = {}
            for j in cur_window.keys():
                cur_length = cur_window[j]
                if (cur_length + current_concatenated_length) * control_threshold <= max_length and (
                    max_size == -1 or len(current_list) < max_size
                ):
                    current_concatenated_length += cur_length
                    current_list.append(int(j))
                else:
                    next_window[j] = cur_window[j]

            if current_list == []:
                if i != len(lengths) - 1:
                    current_list.append(int(next(iter(next_window))))
                    next_window.pop(next(iter(next_window)))
                    cur_window.pop(next(iter(next_window)))
                else:
                    i += 1
                    continue

            for k in range(min(len(current_list), len(lengths) - i - 1)):
                if k + i + window_size < len(lengths):
                    index = k + i + window_size
                    next_window[f"{index}"] = lengths[index]
            i += min(len(current_list), len(lengths) - i)

            result.append(current_list)

            current_concatenated_length = 0
            current_list = []

        # assert to make sure no indices were missing
        assert sum([len(indices) for indices in result]) == len(lengths)
        return result

    @property
    def modality_length(self):
        """
        Get the length of each modality in the dataset.

        Returns:
            List of modality lengths
        """
        # If it is packing, we add by packing index
        if self.config.packing:
            lengths = []
            for index_group in self.packing_index:
                cur_length = 0
                for index in index_group:
                    cur_length += self.data_lengths[index]
                lengths.append(cur_length)
            return lengths
        # Otherwise, the original data lengths is sufficient
        return self.data_lengths

    def __len__(self):
        """Return the length of the dataset."""
        if self.config.packing:
            return len(self.packing_index)
        return len(self.data_list)

    def __getitem__(self, index):
        """Get a sample from the dataset by index."""
        if self.config.packing:
            index_group = self.packing_index[index]
            data_dict_list = self.load_from_packing(index_group)
            return data_dict_list

        if (
            self.config.dataset_format == "json"
            or self.config.dataset_format == "jsonl"
            or self.config.dataset_format == "arrow"
        ):
            data_dict = self.load_from_json(self.data_list[index])
        elif self.config.dataset_format == "yaml":
            data_dict = self.load_from_json(self.data_list[index], self.data_folder[index])
        elif self.config.dataset_format == "hf_dataset":
            data_dict = self.load_from_hf(self.data_list[index])
        else:
            raise NotImplementedError
        return data_dict

    def load_from_packing(self, index_group):
        """
        Load data from a packing index group.

        Args:
            index_group: List of indices to load

        Returns:
            List of loaded data dictionaries
        """
        if self.config.dataset_format == "json" or self.config.dataset_format == "jsonl":
            data_dict_list = [self.load_from_json(self.data_list[index]) for index in index_group]
        elif self.config.dataset_format == "yaml":
            data_dict_list = [
                self.load_from_json(self.data_list[index], self.data_folder[index]) for index in index_group
            ]
        elif self.config.dataset_format == "hf_dataset":
            data_dict_list = [self.load_from_hf(self.data_list[index]) for index in index_group]
        else:
            raise NotImplementedError
        return data_dict_list

    def load_from_json(self, data, data_folder=None):
        """
        Default implementation for loading from JSON format.
        Subclasses should override this method to provide specific implementations.

        Args:
            data: The JSON data to process
            data_folder: Optional folder path for data files

        Returns:
            Processed data dictionary
        """
        raise NotImplementedError("Subclasses must implement load_from_json")

    def load_from_hf(self, data):
        """
        Default implementation for loading from HuggingFace dataset format.
        Subclasses should override this method to provide specific implementations.

        Args:
            data: The HuggingFace dataset data to process

        Returns:
            Processed data dictionary
        """
        raise NotImplementedError("Subclasses must implement load_from_hf")

    def get_collator(self):
        """
        Get the appropriate collator for this dataset.
        Subclasses should override this method to provide specific implementations.

        Returns:
            A collator instance suitable for this dataset type
        """
        raise NotImplementedError("Subclasses must implement get_collator")
