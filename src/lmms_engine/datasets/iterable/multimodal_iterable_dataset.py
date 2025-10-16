import math
import os
import random
import traceback
from copy import deepcopy

import torch.distributed as dist
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from loguru import logger
from torch.utils.data import get_worker_info

from lmms_engine.datasets.multimodal_mixin import MultiModalDataLoadingMixin
from lmms_engine.utils import DataUtilities

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

from lmms_engine.datasets.iterable.base_iterable_dataset import BaseIterableDataset


class MultiModalIterableDataset(BaseIterableDataset, MultiModalDataLoadingMixin):
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
            self.storage_client = BlobServiceClient(
                account_url=SAS_URL, retry_policy=RETRY_POLICY
            )
            self.bucket_name = self.config.bucket_name
        self.cur_idx = 0
        if not dist.is_initialized():
            logger.info(
                "Distributed environment not initialized, setting rank and world size to 0 and 1"
            )
            self.rank = 0
            self.world_size = 1
        else:
            logger.info(
                "Distributed environment initialized, setting rank and world size to dist.get_rank() and dist.get_world_size()"
            )
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

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
                self.data_list, self.data_folder = DataUtilities.load_inline_datasets(
                    self.config.datasets
                )
            elif self.config.dataset_path is not None:
                # Load from external YAML file
                self.data_list, self.data_folder = DataUtilities.load_yaml(
                    self.config.dataset_path
                )
            else:
                raise ValueError(
                    "For yaml format, either 'datasets' or 'dataset_path' must be provided"
                )
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

    def get_one_sample(self, index, data_folder=None, data_list=None):
        """Get a sample from the dataset by index."""
        if data_folder is None:
            data_folder = self.data_folder[index]
        if data_list is None:
            data_list = self.data_list
        if (
            self.config.dataset_format == "json"
            or self.config.dataset_format == "jsonl"
            or self.config.dataset_format == "arrow"
        ):
            data_dict = self.load_from_json(data_list[index])
        elif self.config.dataset_format == "yaml":
            data_dict = self.load_from_json(data_list[index], data_folder)
        elif self.config.dataset_format == "hf_dataset":
            data_dict = self.load_from_hf(data_list[index])
        else:
            raise NotImplementedError
        return data_dict

    def __iter__(self):
        worker_info = get_worker_info()
        rank = self.rank
        world_size = self.world_size

        assert isinstance(
            self.data_list, HFDataset
        ), "Data list must be a HuggingFace dataset for IterableDataset"

        # HF shard logic, if len(dataset) % n == l
        # The first l ranks will have dataset length (len(dataset) // n) + 1
        # The rest ranks will have dataset length (len(dataset) // n)
        rank_mod_size = len(self.data_list) % world_size
        per_rank_size = [
            (len(self.data_list) // world_size) + 1
            if i < rank_mod_size
            else (len(self.data_list) // world_size)
            for i in range(world_size)
        ]
        start_index = sum(per_rank_size[:rank])
        end_index = start_index + per_rank_size[rank]

        # Shard the data according to distributed environment
        curr_data_folder = self.data_folder[start_index:end_index]
        # self.data_folder = self.data_folder[start_index:end_index]
        curr_data_list = self.data_list.shard(
            num_shards=world_size, index=rank, contiguous=True
        )

        if worker_info is None:
            iter_start = 0
            iter_end = len(curr_data_list)
        else:
            # split workload
            per_worker = int(
                math.ceil(len(curr_data_list) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(curr_data_list))

        # Distrbute the data to each worker
        curr_data_list = curr_data_list.select(range(iter_start, iter_end))
        if getattr(self, "data_folder", None) is not None:
            curr_data_folder = curr_data_folder[iter_start:iter_end]

        if self.config.packing:
            # Reset index at the start of each iteration pass
            self.cur_idx = 0
            buffer = []
            buffer_length = 0
            packing_length = self.config.packing_length

            # Iterate through the dataset once per epoch
            while self.cur_idx < len(curr_data_list):
                try:
                    data_dict = self.get_one_sample(
                        self.cur_idx, curr_data_folder[self.cur_idx], curr_data_list
                    )
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error getting one sample: {e}, skip this sample")
                    self.cur_idx += 1
                    continue
                input_ids = data_dict["input_ids"]
                data_length = input_ids.shape[0]
                self.cur_idx += 1

                # Drop overlong sample if filtering is enabled
                if data_length > packing_length and self.config.filter_overlong:
                    continue

                # If current sample cannot fit into current buffer, yield the buffer first
                if buffer_length > 0 and buffer_length + data_length > packing_length:
                    yield buffer
                    buffer = []
                    buffer_length = 0

                # If the sample is still longer than packing_length (and not filtered),
                # yield it as its own batch to avoid stalling
                if data_length > packing_length:
                    yield [data_dict]
                    continue

                # Append to buffer
                buffer.append(data_dict)
                buffer_length += data_length

            # Flush remaining buffer
            if len(buffer) > 0:
                yield buffer
        else:
            self.cur_idx = 0
            while self.cur_idx < len(curr_data_list):
                try:
                    yield self.get_one_sample(
                        self.cur_idx, curr_data_folder[self.cur_idx], curr_data_list
                    )
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error getting one sample: {e}, skip this sample")
                    self.cur_idx += 1
                    continue
                self.cur_idx += 1

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
