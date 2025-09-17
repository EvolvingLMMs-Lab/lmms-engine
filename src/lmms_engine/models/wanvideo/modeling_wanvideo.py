import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from loguru import logger
from safetensors.torch import load_file
from transformers import PreTrainedModel

# from transformers.configuration_utils import PretrainedConfig
# from transformers.modeling_utils import (
#     SpecificPreTrainedModelType,
#     restore_default_torch_dtype,
# )
from transformers.utils import ModelOutput

from .configuration_wanvideo import WanVideoConfig
from .wan_video_dit import WanDitModel, sinusoidal_embedding_1d
from .wan_video_text_encoder import WanTextEncoder
from .wan_video_vae import WanVideoVAE38

PATTERN = "B C H W"


@dataclass
class WanVideoOutput(ModelOutput):
    noise_pred: Optional[torch.FloatTensor] = None
    text_embeddings: Optional[torch.FloatTensor] = None


class WanVideoPreTrainedModel(PreTrainedModel):
    config: WanVideoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["DiTBlock"]
    # _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    # _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True


class WanVideoForConditionalGeneration(WanVideoPreTrainedModel):
    def __init__(self, config: WanVideoConfig):
        super().__init__(config)
        self.config = config
        # Main DiT model
        self.dit = WanDitModel(config)
        self.vae = WanVideoVAE38()
        self.text_encoder = WanTextEncoder()

        self.image_encoder = None
        self.motion_controller = None
        self.vace = None

        self.seperated_timestep = config.seperated_timestep
        self.require_vae_embedding = config.require_vae_embedding
        self.require_clip_embedding = config.require_clip_embedding
        self.fuse_vae_embedding_in_latents = config.fuse_vae_embedding_in_latents

        # The following parameters are used for shape check.
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.time_division_factor = 4
        self.time_division_remainder = 1
        self.trainable_modules = config.trainable_modules

    def freeze_except(self):
        trainable_modules = (
            [] if not self.trainable_modules else self.trainable_modules.split(",")
        )
        for name, model in self.named_children():
            if name in trainable_modules:
                model.train()
                model.requires_grad_(True)
            else:
                model.eval()
                model.requires_grad_(False)

    def encode_prompt(self, input_ids, attetnion_mask, device="cuda"):
        seq_lens = attetnion_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(input_ids, attetnion_mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def preprocess_video(
        self,
        video,
        dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):
        # Transform a list of PIL.Image to torch.Tensor
        video = [
            repeat(image, f"H W C -> {PATTERN}", **({"B": 1} if "B" in PATTERN else {}))
            for image in video
        ]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    def generate_noise(
        self,
        shape,
        seed=None,
        rand_device="cpu",
        rand_dtype=torch.float32,
        device=None,
        dtype=None,
    ):
        # Initialize Gaussian noise
        generator = (
            None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        )
        noise = torch.randn(
            shape, generator=generator, device=rand_device, dtype=rand_dtype
        )
        noise = noise.to(dtype=dtype or self.dtype, device=device or self.device)
        return noise

    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (
                (height + self.height_division_factor - 1)
                // self.height_division_factor
                * self.height_division_factor
            )
            logger.info(
                f"height % {self.height_division_factor} != 0. We round it up to {height}."
            )

        if width % self.width_division_factor != 0:
            width = (
                (width + self.width_division_factor - 1)
                // self.width_division_factor
                * self.width_division_factor
            )
            logger.info(
                f"width % {self.width_division_factor} != 0. We round it up to {width}."
            )

        if num_frames is not None:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (
                    (num_frames + self.time_division_factor - 1)
                    // self.time_division_factor
                    * self.time_division_factor
                    + self.time_division_remainder
                )
                logger.info(
                    f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}."
                )

        return height, width, num_frames

    def noise_initialize(
        self, height, width, num_frames, seed, rand_device, vace_reference_image
    ):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        shape = (
            1,
            self.vae.model.z_dim,
            length,
            height // self.vae.upsampling_factor,
            width // self.vae.upsampling_factor,
        )
        noise = self.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return noise

    def embed_input_video(
        self, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image
    ):
        input_video = self.preprocess_video(input_video)  # B, C, T, H, W
        input_latents = self.vae.encode(
            input_video,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=self.dtype, device=self.device)
        if vace_reference_image is not None:
            vace_reference_image = self.preprocess_video([vace_reference_image])
            vace_reference_latents = self.vae.encode(
                vace_reference_image, device=self.device
            ).to(dtype=self.dtype, device=self.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        return input_latents

    def embed_image_VAE(
        self,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        image = repeat(
            input_image, f"H W C -> {PATTERN}", **({"B": 1} if "B" in PATTERN else {})
        )
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = repeat(
                end_image, f"H W C -> {PATTERN}", **({"B": 1} if "B" in PATTERN else {})
            )
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 2, height, width).to(image.device),
                    end_image.transpose(0, 1),
                ],
                dim=1,
            )
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        y = self.vae.encode(
            [vae_input.to(dtype=self.dtype, device=self.device)],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=self.dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=self.dtype, device=self.device)
        return y

    def embed_image_CLIP(self, input_image, end_image, height, width):
        image = repeat(
            input_image, f"H W C -> {PATTERN}", **({"B": 1} if "B" in PATTERN else {})
        )
        clip_context = self.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = repeat(
                end_image, f"H W C -> {PATTERN}", **({"B": 1} if "B" in PATTERN else {})
            )
            if self.dit.has_image_pos_emb:
                clip_context = torch.concat(
                    [clip_context, self.image_encoder.encode_image([end_image])], dim=1
                )
        clip_context = clip_context.to(dtype=self.dtype, device=self.device)
        return clip_context

    def embed_image_fused(
        self, input_image, latents, height, width, tiled, tile_size, tile_stride
    ):
        image = repeat(input_image, f"H W C -> C T H W", T=1)
        z = self.vae.encode(
            [image],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        latents[:, :, 0:1] = z
        return latents, z

    def fun_control(
        self,
        control_video,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        clip_feature,
        y,
        latents,
    ):
        # self.load_models_to_device(self.onload_model_names)
        control_video = self.preprocess_video(control_video)
        control_latents = self.vae.encode(
            control_video,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=self.dtype, device=self.device)
        control_latents = control_latents.to(dtype=self.dtype, device=self.device)
        y_dim = self.dit.in_channels - control_latents.shape[1] - latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros(
                (1, 257, 1280), dtype=self.dtype, device=self.device
            )
            y = torch.zeros(
                (1, y_dim, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return clip_feature, y

    def fun_reference(self, reference_image, height, width):
        reference_image = reference_image
        reference_latents = self.preprocess_video([reference_image])
        reference_latents = self.vae.encode(reference_latents, device=self.device)
        return reference_latents

    def fun_camera_control(
        self,
        height,
        width,
        num_frames,
        camera_control_direction,
        camera_control_speed,
        camera_control_origin,
        latents,
        input_image,
        tiled,
        tile_size,
        tile_stride,
    ):
        camera_control_plucker_embedding = (
            self.dit.control_adapter.process_camera_coordinates(
                camera_control_direction,
                num_frames,
                height,
                width,
                camera_control_speed,
                camera_control_origin,
            )
        )

        control_camera_video = (
            camera_control_plucker_embedding[:num_frames]
            .permute([3, 0, 1, 2])
            .unsqueeze(0)
        )
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(
                    control_camera_video[:, :, 0:1], repeats=4, dim=2
                ),
                control_camera_video[:, :, 1:],
            ],
            dim=2,
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, 4, c, h, w)
            .transpose(2, 3)
        )
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, c * 4, h, w)
            .transpose(1, 2)
        )
        control_camera_latents_input = control_camera_latents.to(
            device=self.device, dtype=self.dtype
        )

        input_image = input_image
        input_latents = self.preprocess_video([input_image])
        input_latents = self.vae.encode(input_latents, device=self.device)
        y = torch.zeros_like(latents).to(self.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=self.dtype, device=self.device)

        if y.shape[1] != self.dit.in_channels - latents.shape[1]:
            image = repeat(
                input_image,
                f"H W C -> {PATTERN}",
                **({"B": 1} if "B" in PATTERN else {}),
            )
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )
            y = self.vae.encode(
                [vae_input.to(dtype=self.dtype, device=self.device)],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )[0]
            y = y.to(dtype=self.dtype, device=self.device)
            msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat(
                [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                dim=1,
            )
            msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk, y])
            y = y.unsqueeze(0)
            y = y.to(dtype=self.dtype, device=self.device)
        return control_camera_latents_input, y

    def vace_call(
        self,
        vace_video,
        vace_video_mask,
        vace_reference_image,
        vace_scale,
        height,
        width,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        if (
            vace_video is not None
            or vace_video_mask is not None
            or vace_reference_image is not None
        ):
            self.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros(
                    (1, 3, num_frames, height, width),
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                vace_video = self.preprocess_video(vace_video)

            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = self.preprocess_video(
                    vace_video_mask, min_value=0, max_value=1
                )

            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = self.vae.encode(
                inactive,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=self.dtype, device=self.device)
            reactive = self.vae.encode(
                reactive,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=self.dtype, device=self.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)

            vace_mask_latents = rearrange(
                vace_video_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8
            )
            vace_mask_latents = F.interpolate(
                vace_mask_latents,
                size=(
                    (vace_mask_latents.shape[2] + 3) // 4,
                    vace_mask_latents.shape[3],
                    vace_mask_latents.shape[4],
                ),
                mode="nearest-exact",
            )

            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = self.preprocess_video([vace_reference_image])
                vace_reference_latents = self.vae.encode(
                    vace_reference_image,
                    device=self.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                ).to(dtype=self.dtype, device=self.device)
                vace_reference_latents = torch.concat(
                    (vace_reference_latents, torch.zeros_like(vace_reference_latents)),
                    dim=1,
                )
                vace_video_latents = torch.concat(
                    (vace_reference_latents, vace_video_latents), dim=2
                )
                vace_mask_latents = torch.concat(
                    (torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents),
                    dim=2,
                )

            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return vace_context, vace_scale
        else:
            return None, vace_scale

    def forward_preprocess(self, scheduler, data_inputs: dict[str, Any]):
        inputs = data_inputs
        height, width, num_frames = self.check_resize_height_width(
            inputs["height"], inputs["width"], inputs["num_frames"]
        )
        inputs.update({"height": height, "width": width, "num_frames": num_frames})
        noise = self.noise_initialize(
            inputs["height"],
            inputs["width"],
            inputs["num_frames"],
            inputs["seed"],
            self.device,
            inputs["vace_reference_image"],
        )
        inputs.update({"noise": noise})

        if inputs["video"] is not None:
            input_latents = self.embed_input_video(
                inputs["video"],
                noise,
                inputs["tiled"],
                inputs["tile_size"],
                inputs["tile_stride"],
                inputs["vace_reference_image"],
            )
            if not scheduler.training:
                latents = scheduler.add_noise(
                    input_latents, noise, timestep=scheduler.timesteps[0]
                )
                inputs.update({"latents": latents})
            else:
                inputs.update(
                    {"latents": noise, "input_latents": input_latents}
                )  # this 'latents' actually will not be used in training.
        else:
            inputs.update({"latents": noise})
        # might need to be checked.
        context = self.encode_prompt(
            inputs["input_ids"], inputs["attention_mask"], device=self.device
        )
        inputs.update({"context": context})

        if inputs["input_image"] is not None and self.require_vae_embedding:
            y = self.embed_image_VAE(
                inputs["input_image"],
                inputs["end_image"],
                num_frames,
                height,
                width,
                inputs["tiled"],
                inputs["tile_size"],
                inputs["tile_stride"],
            )
            inputs.update({"y": y})

        if (
            (inputs["input_image"] is not None)
            and (self.image_encoder is not None)
            and self.require_clip_embedding
        ):
            clip_feature = self.embed_image_CLIP(
                inputs["input_image"], inputs["end_image"], height, width
            )
            inputs.update({"clip_feature": clip_feature})

        if inputs["input_image"] is not None and self.fuse_vae_embedding_in_latents:
            latents, first_frame_latents = self.embed_image_fused(
                inputs["input_image"],
                inputs["latents"],
                height,
                width,
                inputs["tiled"],
                inputs["tile_size"],
                inputs["tile_stride"],
            )
            inputs.update(
                {
                    "latents": latents,
                    "fuse_vae_embedding_in_latents": True,
                    "first_frame_latents": first_frame_latents,
                }
            )

        if inputs["control_video"] is not None:
            clip_feature, y = self.fun_control(
                inputs["control_video"],
                num_frames,
                height,
                width,
                inputs["tiled"],
                inputs["tile_size"],
                inputs["tile_stride"],
                clip_feature,
                y,
                latents,
            )
            inputs.update({"clip_feature": clip_feature, "y": y})

        if inputs["reference_image"] is not None:
            reference_latents, clip_feature = self.fun_reference(
                inputs["reference_image"], height, width
            )
            if self.image_encoder is not None:
                clip_feature = repeat(
                    inputs["reference_image"],
                    f"H W C -> {PATTERN}",
                    **({"B": 1} if "B" in PATTERN else {}),
                )
                clip_feature = self.image_encoder.encode_image([clip_feature])
            inputs.update(
                {"reference_latents": reference_latents, "clip_feature": clip_feature}
            )

        if inputs["camera_control_direction"] is not None:
            control_camera_latents_input, y = self.fun_camera_control(
                inputs["height"],
                inputs["width"],
                inputs["num_frames"],
                inputs["camera_control_direction"],
                inputs["camera_control_speed"],
                inputs["camera_control_origin"],
                latents,
                inputs["input_image"],
                inputs["tiled"],
                inputs["tile_size"],
                inputs["tile_stride"],
            )
            inputs.update(
                {"control_camera_latents_input": control_camera_latents_input, "y": y}
            )

        if inputs["motion_bucket_id"] is not None:
            motion_bucket_id = torch.Tensor((inputs["motion_bucket_id"],)).to(
                dtype=self.dtype, device=self.device
            )
            inputs.update({"motion_bucket_id": motion_bucket_id})

        vace_context, vace_scale = self.vace_call(
            inputs["vace_video"],
            inputs["vace_video_mask"],
            inputs["vace_reference_image"],
            inputs["vace_scale"],
            inputs["height"],
            inputs["width"],
            inputs["num_frames"],
            inputs["tiled"],
            inputs["tile_size"],
            inputs["tile_stride"],
        )
        inputs.update({"vace_context": vace_context, "vace_scale": vace_scale})
        return inputs

    def forward(
        self,
        latents,
        context,
        timestep,
        y: Optional[torch.FloatTensor] = None,
        reference_latents: Optional[torch.Tensor] = None,
        clip_feature: Optional[torch.FloatTensor] = None,
        vace_context: Optional[torch.FloatTensor] = None,
        vace_scale: Optional[float] = 1.0,
        motion_bucket_id: Optional[int] = None,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> WanVideoOutput:
        t = self.dit.time_embedding(
            sinusoidal_embedding_1d(self.dit.freq_dim, timestep).to(
                device=self.dit.device, dtype=self.dit.dtype
            )
        )
        t_mod = self.dit.time_projection(t).unflatten(1, (6, self.dit.hidden_size))

        # Motion Controller
        if motion_bucket_id is not None and self.motion_controller is not None:
            t_mod = t_mod + self.motion_controller(motion_bucket_id).unflatten(
                1, (6, self.dit.hidden_size)
            )
        context = self.dit.text_embedding(context)

        x = latents
        # Merged cfg
        if x.shape[0] != context.shape[0]:
            x = torch.concat([x] * context.shape[0], dim=0)
        if timestep.shape[0] != context.shape[0]:
            timestep = torch.concat([timestep] * context.shape[0], dim=0)

        # Image Embedding
        if y is not None and self.require_vae_embedding:
            x = torch.cat([x, y], dim=1)

        if clip_feature is not None and self.require_clip_embedding:
            clip_embdding = self.dit.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        # Add camera control
        x, (f, h, w) = self.dit.patchify(x, control_camera_latents_input)

        # Reference image
        if reference_latents is not None:
            if len(reference_latents.shape) == 5:  # video case
                reference_latents = reference_latents[
                    :, :, 0
                ]  # only use the first frame
            reference_latents = (
                self.dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
            )
            x = torch.concat([reference_latents, x], dim=1)
            f += 1

        freqs = (
            torch.cat(
                [
                    self.dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        if vace_context is not None:
            vace_hints = self.vace(x, vace_context, context, t_mod, freqs)

        for block_id, block in enumerate(self.dit.blocks):
            x = block(x, context, t_mod, freqs)

            if vace_context is not None and block_id in self.vace.vace_layers_mapping:
                current_vace_hint = vace_hints[self.vace.vace_layers_mapping[block_id]]
                x = x + current_vace_hint * vace_scale

        x = self.dit.head(x, t)
        # Remove reference latents
        if (
            reference_latents is not None
        ):  # since we replace the first frame with the reference image, we need to remove the first frame
            x = x[:, reference_latents.shape[1] :]
            f -= 1
        x = self.dit.unpatchify(x, (f, h, w))

        return WanVideoOutput(
            noise_pred=x,
            text_embeddings=context,
        )


__all__ = [
    "WanVideoForConditionalGeneration",
    "WanVideoPreTrainedModel",
]
