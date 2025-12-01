import os
from copy import deepcopy
from io import BytesIO
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from lmms_engine.datasets.collator import LLaVACollator, VisionCollator
from lmms_engine.datasets.naive.multimodal_dataset import MultiModalDataset
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("llava_video")
class LLaVAVideoDataset(MultiModalDataset):
    """
    LLaVA-Video dataset with time instruction support.

    This dataset implements the LLaVA-Video data processing pipeline:
    1. Load videos with time information (video_time, frame_time, num_frames)
    2. Optionally inject time instructions into prompts
    3. Process mixed image and video batches
    4. Use LLaVA-Video specific processor
    """

    def load_from_csv(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        """Load from CSV data directly without intermediate transformation."""
        images_list = []
        videos = []
        video_metadata = None
        kwargs = {}

        # Build messages directly from CSV data
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data["video"]}},
                    {"type": "text", "text": data["prompt"]},
                ],
            }
        ]

        # Process video content directly
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "video_url":
                    frames, metadata = self._load_video_or_frames(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    # Skip if video loading failed
                    if frames is None:
                        return None
                    videos.append(frames)
                    video_metadata = metadata
                    kwargs["fps"] = metadata["sample_fps"]

        # Inject time instruction if configured
        extra_kwargs = getattr(self.config, "extra_kwargs", {}) or {}
        add_time_instruction = extra_kwargs.get("add_time_instruction", False)
        if add_time_instruction and video_metadata is not None:
            messages = self.processor.inject_time_instruction(
                messages,
                video_time=video_metadata["video_time"],
                num_frames=video_metadata["num_frames"],
                frame_time=video_metadata["frame_time"],
            )

        # Pass slow-fast parameters to processor for token calculation
        for key in ["faster_token_stride", "mm_spatial_pool_stride", "mm_spatial_pool_mode"]:
            if key in extra_kwargs:
                kwargs[key] = extra_kwargs[key]

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if data_folder is not None:
            images = [Image.open(os.path.join(data_folder, image)) for image in images_list]
        else:
            images = [Image.open(image) for image in images_list]
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None

        inputs = self.processor.process(
            images=images,
            hf_messages=hf_messages,
            videos=videos,
            video_metadata=video_metadata,
            **kwargs,
        )
        return inputs

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        """Load from JSON data with video support and time instruction injection."""
        images_list = []
        videos = []
        video_metadata = None
        kwargs = {}
        messages = data["messages"]

        # Extract images and videos
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "video_url":
                    # Load video with time information
                    frames, metadata = self._load_video_or_frames(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    # Skip if video loading failed
                    if frames is None:
                        return None
                    videos.append(frames)
                    video_metadata = metadata
                    kwargs["fps"] = metadata["sample_fps"]

        # Inject time instruction if configured
        extra_kwargs = getattr(self.config, "extra_kwargs", {}) or {}
        add_time_instruction = extra_kwargs.get("add_time_instruction", False)
        if add_time_instruction and video_metadata is not None:
            messages_copy = deepcopy(messages)
            messages_copy = self.processor.inject_time_instruction(
                messages_copy,
                video_time=video_metadata["video_time"],
                num_frames=video_metadata["num_frames"],
                frame_time=video_metadata["frame_time"],
            )
        else:
            messages_copy = messages

        # Pass slow-fast parameters to processor for token calculation
        for key in ["faster_token_stride", "mm_spatial_pool_stride", "mm_spatial_pool_mode"]:
            if key in extra_kwargs:
                kwargs[key] = extra_kwargs[key]

        hf_messages = TrainUtilities.convert_open_to_hf(messages_copy)

        if data_folder is not None:
            images = [Image.open(os.path.join(data_folder, image)) for image in images_list]
        else:
            images = [Image.open(image) for image in images_list]

        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None

        inputs = self.processor.process(
            images=images,
            hf_messages=hf_messages,
            videos=videos,
            video_metadata=video_metadata,
            **kwargs,
        )
        return inputs

    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        """Load from HuggingFace dataset format."""
        images_list = []
        videos = []
        video_metadata = None
        kwargs = {}
        messages = data["messages"]

        # Get data_folder from multiple sources (in order of priority)
        # 1. From self.data_folder (set by yaml format)
        # 2. From config.extra_kwargs
        # 3. From config.data_folder (if it exists)
        data_folder = None
        if hasattr(self, "data_folder") and self.data_folder is not None:
            # For yaml format, data_folder is a list
            data_folder = self.data_folder if not isinstance(self.data_folder, list) else None
        if data_folder is None:
            extra_kwargs = getattr(self.config, "extra_kwargs", {}) or {}
            data_folder = extra_kwargs.get("data_folder", None)
        if data_folder is None:
            data_folder = getattr(self.config, "data_folder", None)

        # Extract images and videos from messages
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "video_url":
                    # Load video with time information
                    frames, metadata = self._load_video_or_frames(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    # Skip if video loading failed
                    if frames is None:
                        return None
                    videos.append(frames)
                    video_metadata = metadata
                    kwargs["fps"] = metadata["sample_fps"]

        # Inject time instruction if configured
        extra_kwargs = getattr(self.config, "extra_kwargs", {}) or {}
        add_time_instruction = extra_kwargs.get("add_time_instruction", False)
        if add_time_instruction and video_metadata is not None:
            messages_copy = deepcopy(messages)
            messages_copy = self.processor.inject_time_instruction(
                messages_copy,
                video_time=video_metadata["video_time"],
                num_frames=video_metadata["num_frames"],
                frame_time=video_metadata["frame_time"],
            )
        else:
            messages_copy = messages

        # Pass slow-fast parameters to processor for token calculation
        for key in ["faster_token_stride", "mm_spatial_pool_stride", "mm_spatial_pool_mode"]:
            if key in extra_kwargs:
                kwargs[key] = extra_kwargs[key]

        hf_messages = TrainUtilities.convert_open_to_hf(messages_copy)

        # Fallback: use data["image"] field if no images/videos in messages
        if len(images_list) == 0 and "image" in data and data["image"] is not None:
            if isinstance(data["image"], list):
                images = data["image"]
            else:
                images = [data["image"]]
        else:
            # Load images with data_folder support
            if len(images_list) > 0:
                if data_folder is not None:
                    images = [Image.open(os.path.join(data_folder, img)) for img in images_list]
                else:
                    images = [Image.open(img) for img in images_list]
            else:
                images = None

        if len(videos) == 0:
            videos = None

        inputs = self.processor.process(
            images=images,
            hf_messages=hf_messages,
            videos=videos,
            video_metadata=video_metadata,
            **kwargs,
        )
        return inputs

    def get_collator(self):
        """Return appropriate collator based on processor type."""
        if self.processor_config.processor_type == "llava_video":
            return LLaVACollator(self.processor)
        elif self.processor_config.processor_type == "llava":
            return LLaVACollator(self.processor)
        else:
            return VisionCollator(self.processor)

    def __getitem__(self, index):
        """
        Get a sample from the dataset by index.
        Override parent method to handle corrupted videos gracefully.
        """
        # Try to load valid sample, skip corrupted ones
        while True:
            data_dict = super().__getitem__(index)

            # If data loading succeeded (not None), return it
            if data_dict is not None:
                return data_dict

            # If loading failed (corrupted video), try next sample
            from loguru import logger

            logger.warning(f"Sample {index} failed to load (corrupted video), trying next sample...")
            index = (index + 1) % len(self.data_list)

    def load_video_with_time(
        self,
        video_path: Union[str, List[str], BytesIO],
        fps: int,
        data_folder=None,
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Load video with time information for LLaVA-Video.

        This method implements the LLaVA-Video video loading logic with:
        - frames_upbound control
        - force_sample behavior
        - Time information (video_time, frame_time, num_frames)

        Args:
            video_path: Path to video file or BytesIO object
            fps: Target frames per second (for FPS-based sampling)
            data_folder: Optional folder path to prepend

        Returns:
            Tuple of (video_frames, metadata) where metadata contains:
                - video_time: Total video duration in seconds
                - frame_time: String of frame timestamps like "0.00s,0.50s,1.00s"
                - num_frames: Number of sampled frames
                - sample_fps: Effective sampling FPS
        """
        from decord import VideoReader, cpu
        from loguru import logger

        if data_folder is not None:
            video_path = os.path.join(data_folder, video_path)

        try:
            if isinstance(video_path, str) or isinstance(video_path, BytesIO):
                vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            elif isinstance(video_path, list):
                vr = VideoReader(video_path[0], ctx=cpu(0), num_threads=1)
            else:
                raise ValueError(f"Unsupported video path type: {type(video_path)}")
        except Exception as e:
            # Skip corrupted videos by returning None
            logger.warning(f"Failed to load video {video_path}: {e}. Skipping this sample...")
            return None, None

        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        video_time = total_frames / video_fps

        # Initial sampling based on FPS
        avg_fps = round(video_fps / fps) if fps > 0 else 1
        frame_idx = list(range(0, total_frames, avg_fps))
        frame_time = [i / video_fps for i in frame_idx]

        # Apply frames_upbound and force_sample logic
        extra_kwargs = getattr(self.config, "extra_kwargs", {}) or {}
        frames_upbound = extra_kwargs.get("frames_upbound", 0)
        force_sample = extra_kwargs.get("force_sample", False)

        if frames_upbound > 0:
            if len(frame_idx) > frames_upbound or force_sample:
                # Force uniform sampling to frames_upbound
                uniform_sampled_frames = np.linspace(0, total_frames - 1, frames_upbound, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frame_time = [i / video_fps for i in frame_idx]

        # Load frames
        try:
            video_frames = vr.get_batch(frame_idx).asnumpy()
            video_frames = torch.tensor(video_frames).permute(0, 3, 1, 2)  # Convert to TCHW format
        except Exception as e:
            # Skip corrupted videos (decoding errors like NAL unit errors)
            logger.warning(f"Failed to decode video frames from {video_path}: {e}. Skipping this sample...")
            return None, None

        # Format frame time string
        frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])
        num_frames = len(frame_idx)
        sample_fps = num_frames / max(total_frames, 1e-6) * video_fps

        metadata = {
            "video_time": video_time,
            "frame_time": frame_time_str,
            "num_frames": num_frames,
            "sample_fps": sample_fps,
        }

        try:
            vr.seek(0)  # Reset video reader
        except Exception:
            pass  # Ignore seek errors
        return video_frames, metadata

    def _load_video_or_frames(
        self,
        video_path: str,
        data_folder: str = None,
        fps: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Load video from either video file or pre-extracted frames directory.

        This method automatically detects the format:
        - If video_path points to a directory: load from all_frames format
        - If video_path points to a file: load from video file

        Args:
            video_path: Path to video file or directory containing frames
            data_folder: Optional folder path to prepend
            fps: Target frames per second

        Returns:
            Tuple of (video_frames, metadata)
        """
        if data_folder is not None:
            full_path = os.path.join(data_folder, video_path)
        else:
            full_path = video_path

        # Check if this is all_frames format (directory of image files)
        if os.path.isdir(full_path):
            return self._load_video_from_frames(full_path, fps)
        else:
            # Use parent class method for standard video files
            return self.load_video_with_time(full_path, fps, data_folder=None)

    def _load_video_from_frames(
        self,
        frame_dir: str,
        fps: int,
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Load video from directory of pre-extracted frames (all_frames format).

        This format is used by ShareGPTVideo and LLaVA-Hound datasets where
        videos are pre-extracted into individual frame images.

        Args:
            frame_dir: Directory containing frame images
            fps: Target frames per second (used for time calculation)

        Returns:
            Tuple of (video_frames, metadata) or (None, None) if loading fails
        """
        from loguru import logger

        # Get all image files in directory
        try:
            frame_files = [
                os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if os.path.isfile(os.path.join(frame_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            frame_files.sort()  # Ensure frames are in sequence
        except Exception as e:
            logger.warning(f"Failed to list frames in directory {frame_dir}: {e}. Skipping this sample...")
            return None, None

        if len(frame_files) == 0:
            logger.warning(f"No image frames found in directory: {frame_dir}. Skipping this sample...")
            return None, None

        total_frames = len(frame_files)

        # Get configuration from extra_kwargs
        extra_kwargs = getattr(self.config, "extra_kwargs", {}) or {}
        frames_upbound = extra_kwargs.get("frames_upbound", 0)
        force_sample = extra_kwargs.get("force_sample", False)

        # Determine number of frames to sample
        if force_sample and frames_upbound > 0:
            num_frames_to_sample = frames_upbound
        else:
            num_frames_to_sample = min(10, total_frames)  # Default to 10 frames

        # Uniform sampling
        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        # Hardcoded average FPS for pre-extracted frames (same as LLaVA-NeXT)
        avg_fps = 2

        # Calculate time information
        frame_time = [idx / avg_fps for idx in sampled_indices]
        frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])
        video_time = total_frames / avg_fps

        # Load sampled frames
        video_frames = []
        for idx in sampled_indices:
            frame_path = frame_files[idx]
            try:
                with Image.open(frame_path) as img:
                    frame = img.convert("RGB")
                    # Convert to numpy array and add to list
                    frame_array = np.array(frame)
                    video_frames.append(frame_array)
            except Exception as e:
                logger.warning(f"Failed to read frame at path {frame_path}: {e}. Skipping this sample...")
                return None, None

        # Convert to tensor in TCHW format
        video_frames = np.stack(video_frames, axis=0)  # (T, H, W, C)
        video_frames = torch.tensor(video_frames).permute(0, 3, 1, 2)  # (T, C, H, W)

        metadata = {
            "video_time": video_time,
            "frame_time": frame_time_str,
            "num_frames": num_frames_to_sample,
            "sample_fps": num_frames_to_sample / max(total_frames, 1e-6) * avg_fps,
        }

        return video_frames, metadata
