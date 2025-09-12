from typing import List, Literal, Optional, Union

from pydantic import field_validator

from lmms_engine.protocol import Args

from .processor import ProcessorConfig


class DatasetConfig(Args):
    dataset_type: str
    dataset_format: Literal["json", "jsonl", "csv", "yaml", "hf_dataset", "arrow"]
    processor_config: Union[dict, ProcessorConfig]

    # Dataset configuration
    dataset_path: Optional[str] = None  # Optional - used for external files
    datasets: Optional[List[dict]] = None  # Optional - used for inline YAML definitions
    shuffle: bool = True
    eval_dataset_path: Optional[str] = None

    # Object storage configuration
    object_storage: Optional[Literal["azure", "gcs", "none"]] = "none"
    bucket_name: Optional[str] = None

    # Packing configuration
    packing: Optional[bool] = False
    packing_strategy: Optional[str] = None
    packing_length: Optional[int] = 32000
    filter_overlong: Optional[bool] = True
    filter_overlong_workers: Optional[int] = 8
    max_length: Optional[int] = None

    # Video configuration
    video_sampling_strategy: Optional[Literal["fps", "frame_num"]] = "fps"
    video_max_pixels: Optional[int] = 768 * 28 * 28
    video_max_frames: Optional[int] = 768
    frame_num: Optional[int] = 64
    fps: Optional[int] = 1
    video_backend: Optional[Literal["decord", "qwen_vl_utils"]] = "qwen_vl_utils"

    @field_validator(
        "video_max_pixels",
        "video_max_frames",
        "frame_num",
        "fps",
        "packing_length",
        "max_length",
        "filter_overlong_workers",
    )
    @classmethod
    def validate_positive_values(cls, v, info):
        """Validate that numeric video and packing parameters are positive."""
        if v is not None and v <= 0:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v

    @field_validator("video_backend")
    @classmethod
    def validate_video_backend_migration(cls, v):
        """Provide migration warning for deprecated torchvision backend."""
        if v == "torchvision":
            raise ValueError(
                "The 'torchvision' video backend has been removed. "
                "Please use 'decord' or 'qwen_vl_utils' instead. "
                "Migration guide: If you were using torchvision, 'decord' provides "
                "similar functionality with better performance."
            )
        return v
