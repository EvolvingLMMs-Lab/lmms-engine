from dataclasses import dataclass
from typing import Literal, Optional, Union

from .processor import ProcessorConfig


@dataclass
class DatasetConfig:
    dataset_type: Literal["vision", "vision_audio"]
    dataset_format: Literal["json", "jsonl", "yaml", "hf_dataset", "arrow"]
    dataset_path: str
    shuffle: bool = True
    processor_config: Optional[Union[dict, ProcessorConfig]] = None
    eval_dataset_path: Optional[str] = None
    object_storage: Optional[Literal["azure", "gcs", "none"]] = "none"
    bucket_name: Optional[str] = None
    packing: Optional[bool] = False
    packing_strategy: Optional[str] = None
    packing_length: Optional[int] = 32000
    video_sampling_strategy: Optional[Literal["fps", "frame_num"]] = "fps"
    frame_num: Optional[int] = 64
    fps: Optional[int] = 1
    video_backend: Optional[Literal["decord", "torchvision"]] = "decord"
    tokenizer: Optional[str] = None
