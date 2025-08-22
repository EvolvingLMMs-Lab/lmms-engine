from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from PIL import Image


class Processor(ABC):
    @abstractmethod
    def process(
        self,
        images: List[Image.Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        add_system_prompt=True,
        **kwargs,
    ) -> dict:
        raise NotImplementedError("Processor.process() is not implemented")
