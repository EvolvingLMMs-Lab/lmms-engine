import math
import types
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from transformers import Trainer as HFTrainer
from transformers import TrainerCallback

from lmms_engine.models.wanvideo.wan_video_scheduler import FlowMatchScheduler
from lmms_engine.train.registry import TRAINER_REGISTER


class WanVideoCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, logs=None, **kwargs):
        model.freeze_except()
        logger.info(
            f"Trainable_modules: {model.trainable_modules}. Freezing other modules."
        )


@TRAINER_REGISTER.register("wan_trainer")
class WanVideoTrainer(HFTrainer):
    def __init__(self, *args, **kwargs):
        kwargs["callbacks"] = [WanVideoCallback()]
        super().__init__(*args, **kwargs)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        logger.info(
            f"Setting timesteps for diffusion training: {len(self.scheduler.timesteps)} steps"
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        pixel_values = inputs.get("video")
        num_frames, height, width = pixel_values.shape[:3]
        inputs_dict = {
            "video": pixel_values,
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "input_image": pixel_values[0],
            "cfg_scale": inputs.get("cfg_scale", 1),
            "cfg_merge": inputs.get("cfg_merge", False),
            "vace_scale": inputs.get("vace_scale", 1),
            "seed": inputs.get("seed", None),
            "vace_reference_image": inputs.get("vace_reference_image", None),
            "reference_image": inputs.get("reference_image", None),
            "tiled": inputs.get("tiled", False),
            "tile_size": inputs.get("tile_size", None),
            "tile_stride": inputs.get("tile_stride", None),
            "end_image": inputs.get("end_image", None),
            "camera_control_direction": inputs.get("camera_control_direction", None),
            "camera_control_speed": inputs.get("camera_control_speed", None),
            "camera_control_origin": inputs.get("camera_control_origin", None),
            "control_video": inputs.get("control_video", None),
            "motion_bucket_id": inputs.get("motion_bucket_id", None),
            "vace_video": inputs.get("vace_video", None),
            "vace_video_mask": inputs.get("vace_video_mask", None),
        }
        max_timestep_boundary = int(1 * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(0 * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id]

        pre_precessed_inputs = model.forward_preprocess(self.scheduler, inputs_dict)
        training_target = self.scheduler.training_target(
            pre_precessed_inputs["input_latents"],
            pre_precessed_inputs["noise"],
            timestep,
        )
        pre_precessed_inputs["latents"] = self.scheduler.add_noise(
            pre_precessed_inputs["input_latents"],
            pre_precessed_inputs["noise"],
            timestep,
        )
        output = model(
            latents=pre_precessed_inputs.get("latents", None),
            context=pre_precessed_inputs.get("context", None),
            timestep=timestep,
            y=pre_precessed_inputs.get("y", None),
            reference_latents=pre_precessed_inputs.get("reference_latents", None),
            clip_feature=pre_precessed_inputs.get("clip_feature", None),
            vace_context=pre_precessed_inputs.get("vace_context", None),
            vace_scale=pre_precessed_inputs.get("vace_scale", 1.0),
            motion_bucket_id=pre_precessed_inputs.get("motion_bucket_id", None),
            control_camera_latents_input=pre_precessed_inputs.get(
                "control_camera_latents_input", None
            ),
        )
        noise_pred = output.noise_pred
        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), training_target.float(), reduction="mean"
        )
        loss = loss * self.scheduler.training_weight(timestep)

        return loss
