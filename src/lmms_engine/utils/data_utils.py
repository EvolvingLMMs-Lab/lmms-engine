import json
import math
from io import BytesIO
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Literal, Tuple, Union

import jsonlines
import numpy as np
import yaml
from datasets import Dataset, concatenate_datasets, load_from_disk
from librosa import resample
from tqdm import tqdm

from .logging_utils import Logging
from .train_utils import TrainUtilities

FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


class DataUtilities:
    @staticmethod
    def load_json(path: str) -> List[Dict[str, List]]:
        with open(path, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def load_jsonlines(path: str) -> List[Dict[str, List]]:
        data_list = []
        with jsonlines.open(path, "r") as f:
            for data in f:
                data_list.append(data)

        return data_list

    @staticmethod
    def maybe_load_json_or_jsonlines(
        path: str, data_type: Literal["json", "jsonl"]
    ) -> List[Dict[str, List]]:
        if data_type == "json":
            return DataUtilities.load_json(path)
        elif data_type == "jsonl":
            return DataUtilities.load_jsonlines(path)
        else:
            raise NotImplementedError

    @staticmethod
    def maybe_load_by_type(
        path: str, data_type: Literal["json", "jsonl", "arrow"]
    ) -> Union[List[Dict[str, List]], Dataset]:
        if data_type == "arrow":
            dataset = load_from_disk(path)
            return dataset
        else:
            return DataUtilities.maybe_load_json_or_jsonlines(path, data_type)

    @staticmethod
    def wrap_func(args):
        path, data_type = args
        return DataUtilities.maybe_load_by_type(path, data_type)

    @staticmethod
    def load_yaml(path: str) -> Tuple[List[Dict[str, List]], List[str]]:
        data_list = []
        data_folder_list = []
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)
            datasets = yaml_data.get("datasets")
            data_paths = [dataset.get("path") for dataset in datasets]
            data_folders = [dataset.get("data_folder") for dataset in datasets]
            data_types = [dataset.get("data_type") for dataset in datasets]
            force_arrow = any([d_type == "arrow" for d_type in data_types])
            with Pool(cpu_count()) as p:
                Logging.info("Loading data with multiprocess...")
                nested_data_list = list(
                    p.imap(DataUtilities.wrap_func, zip(data_paths, data_types))
                )

            if force_arrow:
                Logging.info(
                    "Detecting arrow dataset, force everything to be loaded in arrow..."
                )
                for data, data_folder, data_path in zip(
                    nested_data_list, data_folders, data_paths
                ):
                    Logging.info(f"Data : {data_path}")
                    if isinstance(data, Dataset):
                        data_list.append(data)
                    else:
                        Logging.info(f"Convert to arrow dataset")
                        data = Dataset.from_list(data)
                        data_list.append(data)
                    Logging.info(f"Dataset size: {len(data)}")
                    data_folder_list.extend([data_folder] * len(data))
                data_list = concatenate_datasets(data_list)
                return data_list, data_folder_list

            for data, data_folder, data_path in zip(
                nested_data_list, data_folders, data_paths
            ):
                Logging.info(f"Data : {data_path}")
                data_list.extend(data)
                data_folder_list.extend([data_folder] * len(data))
                Logging.info(f"Dataset size: {len(data)}")
        return data_list, data_folder_list

    @staticmethod
    def smart_nframes(
        total_frames: int,
        video_fps: int | float,
        fps: int,
    ) -> int:
        """calculate the number of frames for video used for model inputs.

        Args:
            ele (dict): a dict contains the configuration of video.
                support either `fps` or `nframes`:
                    - nframes: the number of frames to extract for model inputs.
                    - fps: the fps to extract frames for model inputs.
                        - min_frames: the minimum number of frames of the video, only used when fps is provided.
                        - max_frames: the maximum number of frames of the video, only used when fps is provided.
            total_frames (int): the original total number of frames of the video.
            video_fps (int | float): the original fps of the video.

        Raises:
            ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

        Returns:
            int: the number of frames for video used for model inputs.
        """
        min_frames = DataUtilities.ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
        max_frames = DataUtilities.floor_by_factor(
            min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = DataUtilities.floor_by_factor(nframes, FRAME_FACTOR)
        if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
            raise ValueError(
                f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
            )
        return nframes

    @staticmethod
    def round_by_factor(number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    @staticmethod
    def ceil_by_factor(number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    @staticmethod
    def floor_by_factor(number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    @staticmethod
    def download_blob_to_stream(
        storage_client,
        bucket_name: str,
        source_blob_name: str,
        file_obj: BytesIO,
        storage_type: Literal["gcs", "azure"] = "azure",
        max_retries: int = 5,
    ) -> BytesIO:
        for i in range(max_retries):
            try:
                if storage_type == "gcs":
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(source_blob_name)
                    blob.download_to_file(file_obj)
                elif storage_type == "azure":
                    blob_client = storage_client.get_blob_client(
                        container=bucket_name, blob=source_blob_name
                    )
                    blob_client.download_blob().readinto(file_obj)
                break
            except Exception as e:
                Logging.error(f"Attempt {i} Error downloading blob: {source_blob_name}")
                Logging.error(f"Error: {e}")
                Logging.error(f"Retrying ...")

        return file_obj

    @staticmethod
    def resample_audio(
        audio_array: np.ndarray, original_sr: int, target_sr: int
    ) -> np.ndarray:
        audio_resample_array = resample(
            audio_array, orig_sr=original_sr, target_sr=target_sr
        )
        return audio_resample_array

    @staticmethod
    def load_inline_datasets(
        datasets: List[Dict],
    ) -> Tuple[List[Dict[str, List]], List[str]]:
        """Load datasets from inline configuration (similar to load_yaml but without file loading).

        Args:
            datasets: List of dataset configurations with path, data_folder, and data_type

        Returns:
            Tuple of (data_list, data_folder_list)
        """
        data_list = []
        data_folder_list = []

        if not datasets:
            return data_list, data_folder_list

        data_paths = [dataset.get("path") for dataset in datasets]
        data_folders = [dataset.get("data_folder", "") for dataset in datasets]
        data_types = [dataset.get("data_type", "json") for dataset in datasets]
        force_arrow = any([d_type == "arrow" for d_type in data_types])

        with Pool(cpu_count()) as p:
            Logging.info("Loading data with multiprocess...")
            nested_data_list = list(
                p.imap(DataUtilities.wrap_func, zip(data_paths, data_types))
            )

        if force_arrow:
            Logging.info(
                "Detecting arrow dataset, force everything to be loaded in arrow..."
            )
            for data, data_folder, data_path in zip(
                nested_data_list, data_folders, data_paths
            ):
                Logging.info(f"Data : {data_path}")
                if isinstance(data, Dataset):
                    data_list.append(data)
                else:
                    Logging.info(f"Convert to arrow dataset")
                    data = Dataset.from_list(data)
                    data_list.append(data)
                Logging.info(f"Dataset size: {len(data)}")
                data_folder_list.extend([data_folder] * len(data))
            data_list = concatenate_datasets(data_list)
            return data_list, data_folder_list

        for data, data_folder, data_path in zip(
            nested_data_list, data_folders, data_paths
        ):
            Logging.info(f"Data : {data_path}")
            data_list.extend(data)
            data_folder_list.extend([data_folder] * len(data))
            Logging.info(f"Dataset size: {len(data)}")

        return data_list, data_folder_list
