from typing import List, Optional

from transformers import PretrainedConfig, SiglipVisionConfig, ViTMAEConfig


class RaeSiglipConfig(PretrainedConfig):
    model_type = "rae_siglip"
    sub_configs = {
        "decoder_config": ViTMAEConfig,
        "encoder_config": SiglipVisionConfig,
    }

    def __init__(
        self,
        decoder_config=None,
        encoder_config=None,
        encoder_processor_path="google/siglip2-base-patch16-256",
        noise_tau: float = 0.8,
        encoder_input_size: Optional[int] = None,
        reshape_to_2d: bool = True,
        latent_mean: Optional[List[float]] = None,
        latent_var: Optional[List[float]] = None,
        eps: float = 1e-5,
        **kwargs,
    ):
        if decoder_config is None:
            decoder_config = ViTMAEConfig()
        elif isinstance(decoder_config, dict):
            decoder_config = ViTMAEConfig(**decoder_config)

        if encoder_config is None:
            encoder_config = SiglipVisionConfig()
        elif isinstance(encoder_config, dict):
            encoder_config = SiglipVisionConfig(**encoder_config)

        self.decoder_config = decoder_config
        self.encoder_config = encoder_config
        self.encoder_processor_path = encoder_processor_path
        self.noise_tau = noise_tau
        self.encoder_input_size = (
            int(encoder_input_size) if encoder_input_size is not None else int(self.decoder_config.image_size)
        )
        self.reshape_to_2d = bool(reshape_to_2d)
        self.latent_mean = latent_mean
        self.latent_var = latent_var
        self.eps = float(eps)
        self.initializer_factor = 1.0
        self.hidden_size = decoder_config.hidden_size

        super().__init__(**kwargs)
