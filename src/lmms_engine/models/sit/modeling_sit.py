import math

import torch
from torchvision.utils import make_grid
from transformers import PreTrainedModel

from .configuration_sit import SiTConfig
from .models import SiT
from .transport import Sampler, create_transport

try:
    from diffusers import AutoencoderKL
except ImportError:
    AutoencoderKL = None


class SiTModel(PreTrainedModel):
    config_class = SiTConfig

    def __init__(self, config: SiTConfig):
        super().__init__(config)
        self.vae_path = config.vae_path
        self.model = SiT(**config.model_params)
        self.vae = AutoencoderKL.from_pretrained(self.vae_path)
        self.transport = create_transport(
            path_type=config.path_type,
            prediction=config.prediction,
            loss_weight=config.loss_weight,
            train_eps=config.train_eps,
            sample_eps=config.sample_eps,
        )
        self.sampler = Sampler(self.transport)
        self.use_cfg = config.cfg_scale > 1.0
        self.post_init()

    def vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x).latent_dist.sample().mul_(0.18215)

    def forward(self, x, y):
        with torch.no_grad():
            x = self.vae_encode(x)
        model_kwargs = dict(y=y)
        loss_dict = self.transport.training_losses(self.model, x, model_kwargs)
        return loss_dict

    def generate_samples(
        self,
        zs,
        model_fn,
        **kwargs,
    ):
        sample_fn = self.sampler.sample_ode()  # default to ode sampling
        samples = sample_fn(zs, model_fn, **kwargs)[-1]
        if self.use_cfg:  # remove null samples
            samples, _ = samples.chunk(2, dim=0)
        samples = self.vae.decode(samples / 0.18215).sample
        # out_samples = torch.zeros((zs.size(0), 3, self.config.image_size, self.config.image_size), device=zs.device)
        return self.array2grid(samples)

    def array2grid(self, x):
        nrow = round(math.sqrt(x.size(0)))
        x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1, 1))
        x = (
            x.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        return x
