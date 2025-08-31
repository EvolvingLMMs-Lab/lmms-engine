from typing import Optional

from lmms_engine.protocol import Args


class ProcessorConfig(Args):
    processor_name: str
    processor_type: str
    max_pixels: Optional[int] = None
    min_pixels: Optional[int] = None
