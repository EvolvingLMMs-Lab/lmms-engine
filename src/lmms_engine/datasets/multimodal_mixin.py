import os
import random
from copy import deepcopy
from io import BytesIO
from multiprocessing import cpu_count
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torchvision.io
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from decord import VideoReader, cpu
from loguru import logger
from PIL import Image
from tqdm import tqdm

from lmms_engine.utils import DataUtilities

try:
    from qwen_vl_utils import fetch_video
except ImportError:
    logger.info("qwen_vl_utils not installed. Skipping import.")


class MultiModalDataLoadingMixin:
    """
    Mixin for loading multimodal data.
    """

    def load_image(self, image_path: str, data_folder=None) -> Image.Image:
        """
        Load an image from file path or object storage.

        Args:
            image_path: Path to the image file
            data_folder: Optional folder path to prepend

        Returns:
            PIL Image object
        """
        if data_folder is not None:
            image_path = os.path.join(data_folder, image_path)

        if self.config.object_storage != "none":
            file_obj = BytesIO()
            file_obj = DataUtilities.download_blob_to_stream(
                self.storage_client,
                self.bucket_name,
                image_path,
                file_obj,
                self.config.object_storage,
            )
            file_obj.seek(0)
            image = Image.open(file_obj)
        else:
            image = Image.open(image_path)
        return image

    def load_audio(self, audio_path: str, sr: int, data_folder=None) -> np.ndarray:
        """
        Load audio from file path or object storage.

        Args:
            audio_path: Path to the audio file
            sr: Target sampling rate
            data_folder: Optional folder path to prepend

        Returns:
            Audio data as numpy array
        """
        if data_folder is not None:
            audio_path = os.path.join(data_folder, audio_path)
        if self.config.object_storage != "none":
            file_obj = BytesIO()
            file_obj = DataUtilities.download_blob_to_stream(
                self.storage_client,
                self.bucket_name,
                audio_path,
                file_obj,
                self.config.object_storage,
            )
            file_obj.seek(0)
            audio, orig_sr = sf.read(file_obj)
            # This is an 2d array, so we need to convert it to 1d
            # Convert the left and right channel to 1
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = DataUtilities.resample_audio(audio, orig_sr, sr)
        else:
            audio = librosa.load(audio_path, sr=sr)[0]
        return audio

    def load_videos(
        self, video_path: str, data_folder=None, fps: int = 1
    ) -> Tuple[np.ndarray, float]:
        """
        Load video from file path or object storage.

        Args:
            video_path: Path to the video file
            data_folder: Optional folder path to prepend
            fps: Target frames per second

        Returns:
            Tuple of (video frames, sample fps)
        """
        if data_folder is not None:
            video_path = os.path.join(data_folder, video_path)

        if self.config.object_storage != "none":
            file_obj = BytesIO()
            file_obj = DataUtilities.download_blob_to_stream(
                self.storage_client,
                self.bucket_name,
                video_path,
                file_obj,
                self.config.object_storage,
            )
            file_obj.seek(0)
            # Forcing to use decord at this time, torchvision actually also can, but I don't want to deal with it now
            return self.load_video_decord(file_obj, fps)

        if self.config.video_backend == "decord":
            return self.load_video_decord(video_path, fps)
        elif self.config.video_backend == "qwen_vl_utils":
            return self.load_video_qwen_vl_utils(video_path, fps)
        else:
            raise ValueError(f"Video backend {self.config.video_backend} not supported")

    def load_video_decord(
        self,
        video_path: Union[str, List[str], BytesIO],
        fps: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Load video using Decord backend.

        Args:
            video_path: Path to video file or BytesIO object
            fps: Target frames per second

        Returns:
            Tuple of (video frames, sample fps)
        """
        if isinstance(video_path, str) or isinstance(video_path, BytesIO):
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        elif isinstance(video_path, list):
            vr = VideoReader(video_path[0], ctx=cpu(0), num_threads=1)
        else:
            raise ValueError(f"Unsupported video path type: {type(video_path)}")

        total_frames, video_fps = len(vr), vr.get_avg_fps()
        if self.config.video_sampling_strategy == "fps":
            nframes = DataUtilities.smart_nframes(
                total_frames, video_fps=video_fps, fps=fps
            )
        elif self.config.video_sampling_strategy == "frame_num":
            nframes = self.config.frame_num
        else:
            raise ValueError(
                f"Invalid video sampling strategy: {self.config.video_sampling_strategy}"
            )
        uniform_sampled_frames = np.linspace(0, total_frames - 1, nframes, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        spare_frames = torch.tensor(spare_frames).permute(
            0, 3, 1, 2
        )  # Convert to TCHW format
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        return spare_frames, sample_fps  # (frames, height, width, channels)

    def load_video_qwen_vl_utils(
        self,
        video_path: str,
        fps: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Load video using Qwen VL utils.

        Args:
            video_path: Path to video file
            fps: Target frames per second

        Returns:
            Tuple of (video frames, sample fps)
        """
        video_dict = {
            "type": "video",
            "video": f"file://{video_path}",
            "min_frames": 1,
            "max_pixels": self.config.video_max_pixels,
            "max_frames": self.config.video_max_frames,
            "min_pixels": self.config.video_min_pixels,
        }

        if self.config.video_sampling_strategy == "frame_num":
            is_even = self.config.frame_num % 2 == 0
            n_frames = self.config.frame_num if is_even else self.config.frame_num + 1
            video_dict["nframes"] = n_frames
            frames, sample_fps = fetch_video(video_dict, return_video_sample_fps=True)
            frames = frames.numpy()
            if is_even:
                return frames, sample_fps
            else:
                return frames[:-1], sample_fps
        elif self.config.video_sampling_strategy == "fps":
            video_dict["fps"] = fps
            frames, sample_fps = fetch_video(video_dict, return_video_sample_fps=True)
            frames = frames.numpy()
            return frames, sample_fps
        else:
            raise ValueError(
                f"Invalid video sampling strategy: {self.config.video_sampling_strategy}"
            )
