import numpy as np
from PIL import Image
from torchvision import transforms

from lmms_engine.mapping_func import register_processor

from .config import ProcessorConfig


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


@register_processor("sit")
class SitDataProcessor:
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config

    def build(self):
        image_size = getattr(self.config.extra_kwargs, "image_size", 256)
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def process(
        self,
        x: Image.Image,
        y: int,
    ):
        x = self.transform(x)
        return x, y

    # Pass since no processor to save
    def save_pretrained(self, output_dir: str):
        pass
