import copy
import os
from collections import OrderedDict

import torch
import wandb
from torch.distributed.tensor import DTensor

from lmms_engine.train.fsdp2.fsdp2_trainer import FSDP2SFTTrainer
from lmms_engine.train.registry import TRAINER_REGISTER


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


@TRAINER_REGISTER.register("sit_trainer")
class SitTrainer(FSDP2SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_model(self):
        super().prepare_model()
        self.ema_model = copy.deepcopy(self.model.model)
        self.ema_model.eval()
        # Labels to condition the model with (feel free to change):
        self.ys = torch.randint(
            1000,
            size=(self.args.per_device_train_batch_size,),
            device=self.model.device,
        )
        self.use_cfg = self.model.config.cfg_scale > 1.0
        # Create sampling noise:
        n = self.ys.size(0)
        self.zs = torch.randn(
            n,
            4,
            self.model.config.input_size,
            self.model.config.input_size,
            device=self.model.device,
        )

        # Setup classifier-free guidance:
        if self.use_cfg:
            self.zs = torch.cat([self.zs, self.zs], 0)
            y_null = torch.tensor([1000] * n, device=self.model.device)
            self.ys = torch.cat([self.ys, y_null], 0)
            sample_model_kwargs = dict(y=self.ys, cfg_scale=self.model.config.cfg_scale)
            model_fn = self.ema_model.forward_with_cfg
        else:
            sample_model_kwargs = dict(y=self.ys)
            model_fn = self.ema_model.forward
        self.sample_model_kwargs = sample_model_kwargs
        self.model_fn = model_fn

    def training_step(self, batch):
        train_metrics = super().training_step(batch)
        # SiT update
        update_ema(self.ema_model, self.fsdp2_model.model)
        # TODO: generate samples, need to convert zs to DTensor
        # if self.global_step % 10000 == 0:
        #     samples = self.model.generate_samples(self.zs, self.model_fn, **self.sample_model_kwargs)
        #     train_metrics["samples"] = wandb.Image(samples)
        return train_metrics

    def save_checkpoints(self, output_path: str, step: int, total_limit: int = None):
        super().save_checkpoints(output_path, step, total_limit)
        ema_state_dict = self.ema_model.state_dict()
        torch.save(ema_state_dict, os.path.join(output_path, "ema.pt"))

    def load_checkpoints(self, output_path: str, step: int):
        super().load_checkpoints(output_path, step)
        ema_state_dict = torch.load(os.path.join(output_path, "ema.pt"))
        self.ema_model.load_state_dict(ema_state_dict)

    # Dummy function to avoid error
    def calculate_training_metrics(self, *args, **kwargs):
        return {}, 0
