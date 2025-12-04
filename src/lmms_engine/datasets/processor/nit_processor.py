import functools
from typing import List, Optional

import diffusers
import numpy as np
import torch
from diffusers import AutoencoderDC, AutoencoderKL
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import hflip

from lmms_engine.mapping_func import register_processor

from .config import ProcessorConfig


def native_resolution_resize(pil_image, min_image_size, max_image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    w, h = pil_image.size
    if w * h < max_image_size**2:
        new_w = max(1, int(w / min_image_size)) * min_image_size
        new_h = max(1, int(h / min_image_size)) * min_image_size
    else:
        new_w = np.sqrt(w / h) * max_image_size
        new_h = new_w * h / w
        new_w = int(new_w / min_image_size) * min_image_size
        new_h = int(new_h / min_image_size) * min_image_size
    pil_image = pil_image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    return pil_image


# from https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/Large-DiT-ImageNet/train.py#L60
def get_train_sampler(global_batch_size, max_steps, resume_step):
    sample_indices = torch.arange(
        0,
        max_steps * global_batch_size,
    ).to(torch.long)
    return sample_indices[resume_step * global_batch_size :].tolist()


class NitProcessorConfig:
    def __init__(self, config: ProcessorConfig) -> None:
        extra_kwargs = getattr(config, "extra_kwargs", {})
        self.min_image_size = extra_kwargs.get("min_image_size", 256)  # TODO: check default
        self.max_image_size = extra_kwargs.get("max_image_size", 1024)  # TODO: check default
        self.vae_downsample_factor = extra_kwargs.get("vae_downsample_factor", 32)
        self.vae_num_channels = extra_kwargs.get("vae_num_channels", 32)


@register_processor("nit")
class NitDataProcessor:
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = NitProcessorConfig(config)
        self.tokenizer = config.tokenizer
        self.processor = config.processor
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.audio_token_id = config.audio_token_id

    def build(self):
        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: native_resolution_resize(
                        pil_image, self.config.min_image_size, self.config.max_image_size
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def process(
        self,
        row: dict,
        **kwargs,
    ):
        image = row["image"]
        label = row["label"]
        processed_image = self.transform(image.convert("RGB"))  # (C, H, W)

        # Calculate tokens
        c, h, w = processed_image.shape
        # VAE downsample factor is 32 for DC-AE
        h_tokens = h // self.config.vae_downsample_factor
        w_tokens = w // self.config.vae_downsample_factor
        num_tokens = h_tokens * w_tokens * self.config.vae_num_channels

        inputs = {"processed_image": processed_image, "num_tokens": num_tokens, "label": label}

        return inputs
