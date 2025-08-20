from dataclasses import dataclass
from typing import List, Literal, Optional, Union

from .processor import ProcessorConfig


@dataclass
class DatasetConfig:
    dataset_type: Literal["vision", "vision_audio"]
    dataset_format: Literal["json", "jsonl", "csv", "yaml", "hf_dataset", "arrow"]
    processor_config: Union[dict, ProcessorConfig]
    dataset_path: Optional[str] = None  # Optional - used for external files
    datasets: Optional[List[dict]] = None  # Optional - used for inline YAML definitions
    shuffle: bool = True
    eval_dataset_path: Optional[str] = None
    object_storage: Optional[Literal["azure", "gcs", "none"]] = "none"
    bucket_name: Optional[str] = None
    packing: Optional[bool] = False
    packing_strategy: Optional[str] = None
    packing_length: Optional[int] = 32000
    video_sampling_strategy: Optional[Literal["fps", "frame_num"]] = "fps"
    frame_num: Optional[int] = 64
    fps: Optional[int] = 1
    video_backend: Optional[
        Literal["decord", "torchvision", "qwen_vl_utils"]
    ] = "qwen_vl_utils"
