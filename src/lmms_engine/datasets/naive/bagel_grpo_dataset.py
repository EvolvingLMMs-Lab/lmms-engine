"""
RL Prompt Datasets for reinforcement learning training.

These datasets support loading prompts from various formats:
1. Text files (one prompt per line) - rl_prompt_text
2. JSONL files (with prompt and metadata) - rl_prompt_jsonl
3. JSONL files with images (for image editing tasks) - rl_prompt_image_jsonl

All datasets are designed to be general-purpose and can be used for any RL training task.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from loguru import logger
from PIL import Image

from lmms_engine.datasets.config import DatasetConfig
from lmms_engine.datasets.naive.base_dataset import BaseDataset
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils import DataUtilities


@register_dataset("rl_prompt_text")
class RLPromptTextDataset(BaseDataset):
    """
    Dataset for loading text prompts from a text file (one prompt per line).

    This is a general-purpose dataset for RL training where each line in the file is a prompt.
    The dataset returns {"prompt": str, "metadata": {}} for each sample.

    Configuration:
        dataset_type: rl_prompt_text
        dataset_format: jsonl  # Will auto-detect .txt files
        dataset_path: /path/to/prompts.txt
    """

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.data_list: List[Dict[str, str]] = []
        self.modality_length: Optional[List[int]] = None

    def _build_from_config(self):
        """Load prompts from text file."""
        file_path = self.config.dataset_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Check file extension to determine format
        # If it's a .txt file, treat as text format regardless of dataset_format
        # Otherwise, use dataset_format
        file_ext = os.path.splitext(file_path)[1].lower()
        is_text_file = file_ext in (".txt", ".text") or self.config.dataset_format in ("text", "txt")

        if is_text_file:
            # Load from text file (one prompt per line)
            with open(file_path, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f.readlines() if line.strip()]

            # Limit test set size if specified (for faster evaluation)
            if hasattr(self.config, "test_max_samples") and self.config.test_max_samples:
                if "test" in file_path.lower():
                    prompts = prompts[: self.config.test_max_samples]
                    logger.info(f"Limited test set to {len(prompts)} samples")

            # Convert to list of dicts
            self.data_list = [{"prompt": prompt, "metadata": {}} for prompt in prompts]

            logger.info(f"Loaded {len(self.data_list)} prompts from {file_path}")
        else:
            raise ValueError(
                f"Unsupported dataset_format: {self.config.dataset_format} for text file. "
                "Use a .txt file or set dataset_format to 'text' or 'txt'."
            )

        # Estimate modality length (for compatibility with length-based sampling)
        # For text prompts, we estimate based on prompt length
        self.modality_length = [len(prompt.split()) for prompt in [item["prompt"] for item in self.data_list]]

        # Shuffle if requested
        if self.config.shuffle:
            import random

            random.seed(self.config.data_seed)
            combined = list(zip(self.data_list, self.modality_length))
            random.shuffle(combined)
            self.data_list, self.modality_length = zip(*combined)
            self.data_list = list(self.data_list)
            self.modality_length = list(self.modality_length)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Return a single sample."""
        return self.data_list[idx]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_list)

    def load_from_json(self, data, data_folder=None):
        """Not used for text format, but required by base class."""
        raise NotImplementedError("Use dataset_format='text' for text files")

    def load_from_hf(self, data):
        """Not used for text format, but required by base class."""
        raise NotImplementedError("Use dataset_format='text' for text files")

    def get_collator(self):
        """Return a simple collator that returns (prompts, metadatas) tuple."""
        return RLPromptCollator()


@register_dataset("rl_prompt_jsonl")
class RLPromptJSONLDataset(BaseDataset):
    """
    Dataset for loading prompts from JSONL files with metadata.

    Each line in the JSONL file should be a JSON object with at least a "prompt" field.
    Additional fields will be stored in the "metadata" dict.

    This is a general-purpose dataset for RL training that supports rich metadata.

    Example JSONL format:
        {"prompt": "A beautiful sunset", "task": "image_generation", "difficulty": "easy"}
        {"prompt": "A cat playing piano", "task": "image_generation", "difficulty": "hard"}

    Configuration:
        dataset_type: rl_prompt_jsonl
        dataset_format: jsonl
        dataset_path: /path/to/prompts.jsonl
    """

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.data_list: List[Dict[str, any]] = []
        self.modality_length: Optional[List[int]] = None

    def _build_from_config(self):
        """Load prompts from JSONL file."""
        if self.config.dataset_format == "jsonl":
            # Load from JSONL file
            file_path = self.config.dataset_path
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")

            data_list = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Extract prompt and metadata
                        prompt = data.get("prompt", "")
                        if not prompt:
                            logger.warning(f"Skipping line without 'prompt' field: {line[:100]}")
                            continue

                        # Store everything except 'prompt' as metadata
                        metadata = {k: v for k, v in data.items() if k != "prompt"}
                        data_list.append({"prompt": prompt, "metadata": metadata})
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")

            # Limit test set size if specified
            if hasattr(self.config, "test_max_samples") and self.config.test_max_samples:
                if "test" in file_path.lower():
                    data_list = data_list[: self.config.test_max_samples]
                    logger.info(f"Limited test set to {len(data_list)} samples")

            self.data_list = data_list
            logger.info(f"Loaded {len(self.data_list)} prompts from {file_path}")
        else:
            raise ValueError(
                f"Unsupported dataset_format: {self.config.dataset_format}. " "Use 'jsonl' for JSONL files."
            )

        # Estimate modality length
        self.modality_length = [len(item["prompt"].split()) for item in self.data_list]

        # Shuffle if requested
        if self.config.shuffle:
            import random

            random.seed(self.config.data_seed)
            combined = list(zip(self.data_list, self.modality_length))
            random.shuffle(combined)
            self.data_list, self.modality_length = zip(*combined)
            self.data_list = list(self.data_list)
            self.modality_length = list(self.modality_length)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """Return a single sample."""
        return self.data_list[idx]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_list)

    def load_from_json(self, data, data_folder=None):
        """Not used for JSONL format, but required by base class."""
        raise NotImplementedError("Use dataset_format='jsonl' for JSONL files")

    def load_from_hf(self, data):
        """Not used for JSONL format, but required by base class."""
        raise NotImplementedError("Use dataset_format='jsonl' for JSONL files")

    def get_collator(self):
        """Return a simple collator that returns (prompts, metadatas) tuple."""
        return RLPromptCollator()


@register_dataset("rl_prompt_image_jsonl")
class RLPromptImageJSONLDataset(BaseDataset):
    """
    Dataset for loading prompts with images from JSONL files.

    This dataset is designed for image editing tasks and other RL training scenarios
    that require both prompts and reference images.

    Each line in the JSONL file should be a JSON object with:
    - "prompt": The text prompt (required)
    - "image": Path to the image file relative to dataset_path (required)
    - Other fields: Stored in metadata

    Example JSONL format:
        {"prompt": "Make the sky more blue", "image": "images/sunset.jpg", "task": "image_edit"}
        {"prompt": "Add a rainbow", "image": "images/landscape.jpg", "task": "image_edit"}

    Configuration:
        dataset_type: rl_prompt_image_jsonl
        dataset_format: jsonl
        dataset_path: /path/to/dataset_directory  # Directory containing images and metadata.jsonl
        metadata_file: metadata.jsonl  # Optional, defaults to train_metadata.jsonl or test_metadata.jsonl
    """

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.data_list: List[Dict[str, any]] = []
        self.modality_length: Optional[List[int]] = None
        self.dataset_dir: Optional[str] = None

    def _build_from_config(self):
        """Load prompts and images from JSONL file."""
        if self.config.dataset_format != "jsonl":
            raise ValueError(
                f"Unsupported dataset_format: {self.config.dataset_format}. " "Use 'jsonl' for JSONL files with images."
            )

        # Determine dataset directory and metadata file
        dataset_path = self.config.dataset_path
        if os.path.isfile(dataset_path):
            # If path is a file, use its directory
            self.dataset_dir = os.path.dirname(dataset_path)
            metadata_file = os.path.basename(dataset_path)
        else:
            # If path is a directory, look for metadata file
            self.dataset_dir = dataset_path
            # Try to determine split from path or use default
            if "test" in dataset_path.lower():
                metadata_file = getattr(self.config, "metadata_file", "test_metadata.jsonl")
            else:
                metadata_file = getattr(self.config, "metadata_file", "train_metadata.jsonl")

        metadata_path = os.path.join(self.dataset_dir, metadata_file)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load metadata
        data_list = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Extract prompt and image path
                    prompt = data.get("prompt", "")
                    image_path = data.get("image", "")
                    if not prompt:
                        logger.warning(f"Skipping line without 'prompt' field: {line[:100]}")
                        continue
                    if not image_path:
                        logger.warning(f"Skipping line without 'image' field: {line[:100]}")
                        continue

                    # Store everything except 'prompt' and 'image' as metadata
                    metadata = {k: v for k, v in data.items() if k not in ("prompt", "image")}
                    data_list.append(
                        {
                            "prompt": prompt,
                            "image_path": image_path,
                            "metadata": metadata,
                        }
                    )
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")

        # Limit test set size if specified
        if hasattr(self.config, "test_max_samples") and self.config.test_max_samples:
            if "test" in dataset_path.lower():
                data_list = data_list[: self.config.test_max_samples]
                logger.info(f"Limited test set to {len(data_list)} samples")

        self.data_list = data_list
        logger.info(f"Loaded {len(self.data_list)} prompts with images from {metadata_path}")

        # Estimate modality length
        self.modality_length = [len(item["prompt"].split()) for item in self.data_list]

        # Shuffle if requested
        if self.config.shuffle:
            import random

            random.seed(self.config.data_seed)
            combined = list(zip(self.data_list, self.modality_length))
            random.shuffle(combined)
            self.data_list, self.modality_length = zip(*combined)
            self.data_list = list(self.data_list)
            self.modality_length = list(self.modality_length)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """Return a single sample with loaded image."""
        item = self.data_list[idx].copy()

        # Load image
        image_path = item["image_path"]
        full_image_path = os.path.join(self.dataset_dir, image_path)
        if not os.path.exists(full_image_path):
            raise FileNotFoundError(f"Image not found: {full_image_path}")

        image = Image.open(full_image_path).convert("RGB")
        item["image"] = image

        # Create a unique identifier combining prompt and image path
        item["prompt_with_image_path"] = f"{item['prompt']}_{image_path}"

        return item

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_list)

    def load_from_json(self, data, data_folder=None):
        """Not used for JSONL format, but required by base class."""
        raise NotImplementedError("Use dataset_format='jsonl' for JSONL files")

    def load_from_hf(self, data):
        """Not used for JSONL format, but required by base class."""
        raise NotImplementedError("Use dataset_format='jsonl' for JSONL files")

    def get_collator(self):
        """Return a collator that returns (prompts, metadatas, images, prompt_with_image_paths) tuple."""
        return RLPromptImageCollator()


class RLPromptCollator:
    """
    Simple collator for RL training with text prompts.

    Returns a tuple of (prompts, metadatas) which matches the format expected
    by RL trainers' sampling phase.
    """

    def __call__(self, examples: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        Collate a batch of examples.

        Args:
            examples: List of dicts with 'prompt' and 'metadata' keys

        Returns:
            Tuple of (prompts, metadatas) lists
        """
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class RLPromptImageCollator:
    """
    Collator for RL training with prompts and images.

    Returns a tuple of (prompts, metadatas, images, prompt_with_image_paths)
    which matches the format expected by image editing RL trainers.
    """

    def __call__(self, examples: List[Dict]) -> Tuple[List[str], List[Dict], List[Image.Image], List[str]]:
        """
        Collate a batch of examples with images.

        Args:
            examples: List of dicts with 'prompt', 'metadata', 'image', and 'prompt_with_image_path' keys

        Returns:
            Tuple of (prompts, metadatas, images, prompt_with_image_paths)
        """
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        prompt_with_image_paths = [example["prompt_with_image_path"] for example in examples]
        return prompts, metadatas, images, prompt_with_image_paths
