from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .c2i.nit_model import NiT
from .configuration_nit import NitConfig


class NitPreTrainedModel(PreTrainedModel):
    config_class = NitConfig
    base_model_prefix = "nit"
    main_input_name = "x"


@dataclass
class NitOutput(ModelOutput):
    x: torch.FloatTensor = None
    zs: Optional[torch.FloatTensor] = None


class NitModel(NitPreTrainedModel):
    def __init__(self, config: NitConfig):
        super().__init__(config)
        self.nit = NiT(
            input_size=config.input_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            class_dropout_prob=config.class_dropout_prob,
            num_classes=config.num_classes,
            encoder_depth=config.encoder_depth,
            projector_dim=config.projector_dim,
            z_dim=config.z_dim,
            use_checkpoint=config.use_checkpoint,
            custom_freqs=config.custom_freqs,
            theta=config.theta,
            max_pe_len_h=config.max_pe_len_h,
            max_pe_len_w=config.max_pe_len_w,
            decouple=config.decouple,
            ori_max_pe_len=config.ori_max_pe_len,
        )
        self.vae = self.load_vae(config.vae_name_or_path)
        if config.compile:
            self.nit = torch.compile(self.nit)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        hw_list: torch.Tensor,
        return_zs: bool = False,
        return_logvar: bool = False,
        **kwargs,
    ):
        outputs = self.nit(x, t, y, hw_list, return_zs=return_zs, return_logvar=return_logvar)

        if return_zs:
            x_out, zs = outputs
            return NitOutput(x=x_out, zs=zs)
        else:
            return NitOutput(x=outputs)
