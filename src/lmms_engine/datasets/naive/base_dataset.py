from abc import abstractmethod
from typing import Dict

from torch.utils.data import Dataset

from lmms_engine.datasets.config import DatasetConfig
from lmms_engine.datasets.processor import ProcessorConfig
from lmms_engine.mapping_func import DATAPROCESSOR_MAPPING


class BaseDataset(Dataset):
    """
    Abstract base class for all datasets in the LMMS Engine.

    This class defines the interface that all dataset implementations must follow.
    It provides abstract methods that must be implemented by subclasses to handle
    different data formats and modalities.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """
        Initialize the base dataset with configuration.

        Args:
            config: Dataset configuration object containing all necessary parameters
        """
        super().__init__()
        self.config = config
        self.processor_config = config.processor_config
        if isinstance(self.processor_config, dict):
            self.processor_config = ProcessorConfig(**self.processor_config)

    def _build_processor(self):
        """
        Build the data processor based on the processor configuration.

        Returns:
            A configured processor instance
        """
        processor_cls = DATAPROCESSOR_MAPPING[self.processor_config.processor_type]
        processor = processor_cls(self.processor_config)
        return processor

    def build(self):
        """
        Build the dataset by loading data and building the processor.
        This method should be called after initialization to prepare the dataset.
        """
        self._build_from_config()
        self.processor = self._build_processor()
        self.processor.build()

    @abstractmethod
    def _build_from_config(self):
        """
        Load and prepare data from the configuration.

        This method should implement the logic to load data from various sources
        (JSON, JSONL, Arrow, Parquet, HF Dataset, YAML) based on the dataset format
        specified in the configuration.
        """
        pass

    @abstractmethod
    def load_from_json(self, data, data_folder=None):
        """
        Load and process data from JSON format.

        Args:
            data: The JSON data to process
            data_folder: Optional folder path for data files

        Returns:
            Processed data dictionary
        """
        pass

    @abstractmethod
    def load_from_hf(self, data):
        """
        Load and process data from HuggingFace dataset format.

        Args:
            data: The HuggingFace dataset data to process

        Returns:
            Processed data dictionary
        """
        pass

    @abstractmethod
    def get_collator(self):
        """
        Get the appropriate collator for this dataset.

        Returns:
            A collator instance suitable for this dataset type
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            Number of samples in the dataset
        """
        pass

    @abstractmethod
    def __getitem__(self, index):
        """
        Get a sample from the dataset by index.

        Args:
            index: Index of the sample to retrieve

        Returns:
            The sample data at the specified index
        """
        pass
