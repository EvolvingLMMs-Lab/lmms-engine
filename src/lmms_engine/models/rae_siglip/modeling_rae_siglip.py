import math
from dataclasses import dataclass
from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoImageProcessor, AutoProcessor
from transformers.models.siglip.modeling_siglip import (
    SiglipPreTrainedModel,
    SiglipVisionModel,
)
from transformers.utils import ModelOutput

from .configuration_rae_siglip import RaeSiglipConfig
from .general_decoder import GeneralDecoder


class RaeSiglipPreTrainedModel(SiglipPreTrainedModel):
    config_class = RaeSiglipConfig
    base_model_prefix = "rae_siglip"


@dataclass
class RaeSiglipOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None
    out_pixels: torch.FloatTensor = None
    loss: torch.FloatTensor = None


class RaeSiglipModel(RaeSiglipPreTrainedModel):
    config: RaeSiglipConfig

    def __init__(self, config: RaeSiglipConfig):
        super().__init__(config)

        self.encoder_config = config.encoder_config
        self.decoder_config = config.decoder_config
        self.proc = AutoImageProcessor.from_pretrained(config.encoder_processor_path)
        self.encoder_mean = torch.tensor(self.proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(self.proc.image_std).view(1, 3, 1, 1)
        # see if the encoder has patch size attribute
        self.encoder_input_size = self.encoder_config.image_size
        self.encoder_patch_size = self.encoder_config.patch_size
        self.latent_dim = self.encoder_config.hidden_size
        assert (
            self.encoder_input_size % self.encoder_patch_size == 0
        ), f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (
            self.encoder_input_size // self.encoder_patch_size
        ) ** 2  # number of patches of the latent

        # decoder
        decoder_config = self.decoder_config
        decoder_config.hidden_size = (
            self.latent_dim
        )  # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = self.encoder_patch_size
        decoder_config.image_size = int(
            self.encoder_patch_size * sqrt(self.base_patches)
        )
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        self.noise_tau = config.noise_tau
        logger.info(f"[RAE SigLIP] Noise tau: {self.noise_tau}")
        self.reshape_to_2d = config.reshape_to_2d
        if config.latent_mean is not None and config.latent_var is not None:
            # Convert lists back to tensors if needed
            if isinstance(config.latent_mean, list):
                latent_mean = torch.tensor(config.latent_mean, dtype=torch.float32)
            else:
                latent_mean = config.latent_mean
            if isinstance(config.latent_var, list):
                latent_var = torch.tensor(config.latent_var, dtype=torch.float32)
            else:
                latent_var = config.latent_var
            # Register as buffers for proper device handling
            self.register_buffer("latent_mean", latent_mean, persistent=False)
            self.register_buffer("latent_var", latent_var, persistent=False)
            self.do_normalization = True
            self.eps = config.eps
        else:
            self.register_buffer("latent_mean", None, persistent=False)
            self.register_buffer("latent_var", None, persistent=False)
            self.do_normalization = False

        encoder = SiglipVisionModel._from_config(self.encoder_config).vision_model
        encoder.post_layernorm.elementwise_affine = False
        encoder.post_layernorm.weight = None
        encoder.post_layernorm.bias = None
        encoder.requires_grad_(False)
        self.encoder = encoder

        self.decoder = GeneralDecoder(
            self.decoder_config, num_patches=self.base_patches
        )

        self.post_init()

    @torch.no_grad()
    def encode(self, pixel_values, interpolate_pos_encoding: bool = False, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "attention_mask"}
        if pixel_values.dim() != 4:
            raise ValueError(
                "pixel_values must be a 4D tensor [batch, channels, height, width]."
            )
        pixel_values = pixel_values.to(dtype=self.encoder_mean.dtype)
        _, _, height, width = pixel_values.shape
        if height != self.encoder_input_size or width != self.encoder_input_size:
            pixel_values = F.interpolate(
                pixel_values,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )
        mean = self.encoder_mean.to(pixel_values.device, pixel_values.dtype)
        std = self.encoder_std.to(pixel_values.device, pixel_values.dtype)
        normed = (pixel_values - mean) / std
        outputs = self.encoder(
            pixel_values=normed,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **filtered_kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if self.training and self.noise_tau > 0:
            shape = (hidden_states.size(0),) + (1,) * (hidden_states.dim() - 1)
            noise_sigma = self.noise_tau * torch.rand(
                shape, device=hidden_states.device, dtype=hidden_states.dtype
            )
            hidden_states = hidden_states + noise_sigma * torch.randn_like(
                hidden_states
            )

        if self.reshape_to_2d:
            batch_size, num_tokens, hidden = hidden_states.shape
            side = int(math.sqrt(num_tokens))
            if side * side != num_tokens:
                raise ValueError(
                    f"Cannot reshape latent with {num_tokens} tokens into square grid."
                )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, hidden, side, side
            )
        if self.do_normalization:
            # Buffers are automatically moved to the correct device
            hidden_states = (hidden_states - self.latent_mean) / torch.sqrt(
                self.latent_var + self.eps
            )

        outputs.last_hidden_state = hidden_states
        return outputs

    def decode(
        self,
        hidden_states: torch.FloatTensor,
        interpolate_pos_encoding: bool = False,
        **kwargs,
    ):
        if hidden_states.dim() == 4:
            latent = hidden_states
        elif hidden_states.dim() == 3 and self.reshape_to_2d:
            batch, seq_len, dim = hidden_states.shape
            side = int(math.sqrt(seq_len))
            if side * side != seq_len:
                raise ValueError(
                    f"Cannot reshape latent with {seq_len} tokens into square grid."
                )
            latent = hidden_states.transpose(1, 2).reshape(batch, dim, side, side)
        elif hidden_states.dim() == 3:
            latent = hidden_states
        else:
            raise ValueError(
                "decoder expects latent states as (batch, seq_len, dim) or (batch, dim, h, w)."
            )

        if self.do_normalization:
            # Buffers are automatically moved to the correct device
            latent = latent * torch.sqrt(self.latent_var + self.eps) + self.latent_mean

        if self.reshape_to_2d:
            bsz, channel, height, width = latent.shape
            hidden_states = latent.view(bsz, channel, height * width).transpose(1, 2)
        else:
            hidden_states = latent

        # Ensure hidden_states dtype matches decoder parameters
        decoder_dtype = next(self.decoder.parameters()).dtype
        hidden_states = hidden_states.to(dtype=decoder_dtype)

        decoder_outputs = self.decoder(
            hidden_states=hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        logits = decoder_outputs.logits
        out_pixels = self.decoder.unpatchify(logits)
        mean = self.encoder_mean.to(out_pixels.device, out_pixels.dtype)
        std = self.encoder_std.to(out_pixels.device, out_pixels.dtype)
        out_pixels = out_pixels * std + mean

        return RaeSiglipOutput(
            hidden_states=decoder_outputs.last_hidden_state,
            out_pixels=out_pixels,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_loss: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs,
    ):
        encoder_outputs = self.encode(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )
        decoder_outputs = self.decode(
            hidden_states=encoder_outputs.last_hidden_state,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        return decoder_outputs
