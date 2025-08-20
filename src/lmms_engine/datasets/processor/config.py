from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ProcessorConfig:
    processor_name: str
    processor_type: str
    max_pixels: Optional[int] = None
    min_pixels: Optional[int] = None
    kwargs: Optional[dict] = None
