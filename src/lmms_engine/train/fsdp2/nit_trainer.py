import functools
import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import AutoencoderDC, AutoencoderKL
from einops import rearrange
from loguru import logger
from torchvision.transforms.functional import hflip
from transformers import AutoModelForCausalLM, AutoTokenizer, T5EncoderModel

from lmms_engine.models.nit.utils.loss import FlowMatchingLoss
from lmms_engine.train.registry import TRAINER_REGISTER

from .fsdp2_trainer import FSDP2SFTTrainer


# dc-ae
def dc_ae_encode(dc_ae, images):
    with torch.no_grad():
        latents = dc_ae.encode(images).latent * dc_ae.config.scaling_factor
    return latents


def dc_ae_decode(dc_ae, latents):
    with torch.no_grad():
        z = latents / dc_ae.config.scaling_factor
        if dc_ae.use_slicing and z.size(0) > 1:
            decoded_slices = [dc_ae._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = dc_ae._decode(z)
        images = decoded  # decoded images
    return images


# sd-vae
def sd_vae_encode(sd_vae, images):
    with torch.no_grad():
        z = sd_vae.encode(images)
        if isinstance(z, dict):
            z = z.latent_dist.sample()
        z = sd_vae.config.scaling_factor * z
    return z


def sd_vae_decode(sd_vae, latents):
    with torch.no_grad():
        z = 1.0 / sd_vae.config.scaling_factor * latents
        out = sd_vae.decode(z)
        if isinstance(out, dict):
            out = out.sample
    return out


# load text-encoder
def load_text_encoder(text_encoder_dir, device, weight_dtype):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)
    if "gemma" in text_encoder_dir:
        tokenizer.padding_side = "right"
        text_encoder = AutoModelForCausalLM.from_pretrained(
            text_encoder_dir, attn_implementation="flash_attention_2", device_map="cpu", torch_dtype=weight_dtype
        ).get_decoder()
    elif "t5" in text_encoder_dir:
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_dir, attn_implementation="sdpa", device_map="cpu", torch_dtype=weight_dtype
        )
    else:
        raise NotImplementedError
    text_encoder.requires_grad_(False)
    text_encoder = text_encoder.eval().to(device=device, dtype=weight_dtype)

    return text_encoder, tokenizer


def encode_prompt(tokenizer, text_encoder, device, weight_dtype, captions, use_last_hidden_state, max_seq_length=256):
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
        results = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        )

        if use_last_hidden_state:
            prompt_embeds = results.last_hidden_state
        else:  # from Imagen paper
            prompt_embeds = results.hidden_states[-2]

    return prompt_embeds, prompt_masks


def prepare_null_cap_feat_mask(text_encoder_type, device, weight_dtype, use_last_hidden_state, max_seq_length=256):
    text_encoder, tokenizer = load_text_encoder(
        text_encoder_dir=text_encoder_type, device=device, weight_dtype=weight_dtype
    )
    null_cap_features, null_cap_mask = encode_prompt(
        tokenizer, text_encoder, device, weight_dtype, "", use_last_hidden_state, max_seq_length=max_seq_length
    )
    return null_cap_features, null_cap_mask


@TRAINER_REGISTER.register("nit_trainer")
class NitTrainer(FSDP2SFTTrainer):
    def __init__(
        self,
        *args,
        vae_name_or_path: str = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        vae_dtype: str = "float32",
        transport_config: dict | None = None,
        patch_size: int = 1,
        proj_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vae_name_or_path = vae_name_or_path
        self.encode_func = self.load_vae(self.vae_name_or_path)
        self.loss_fn = FlowMatchingLoss(**transport_config if transport_config is not None else {})
        self.patch_size = patch_size
        self.proj_loss_weight = proj_loss_weight

    def load_vae(self, vae_name_or_path: str):
        device = torch.cuda.current_device()
        dtype = torch.bfloat16 if self.args.bf16 else torch.float16

        if "sd-vae" in vae_name_or_path:
            sd_vae = AutoencoderKL.from_pretrained(vae_name_or_path, torch_dtype=dtype)
            sd_vae.eval()
            sd_vae.requires_grad_(False)
            sd_vae = sd_vae.to(device)
            return functools.partial(sd_vae_encode, sd_vae)
        elif "dc-ae" in vae_name_or_path:
            dc_ae = AutoencoderDC.from_pretrained(vae_name_or_path, torch_dtype=dtype)
            dc_ae.eval()
            dc_ae.requires_grad_(False)
            dc_ae = dc_ae.to(device)
            return functools.partial(dc_ae_encode, dc_ae)
        else:
            raise ValueError(f"Unsupported VAE type: {vae_name_or_path}")

    def compute_loss(self, batch):
        # Not sure if this part should move to the collator
        if len(batch) == 1 and isinstance(batch[0], list):
            batch = batch[0]
        batch_label = []
        batch_images = []
        packed_latent = []
        hw_list = []

        if self.args.bf16:
            cast_dtype = torch.bfloat16
        else:
            cast_dtype = torch.float16

        device = torch.cuda.current_device()

        # TODO: Optimize when all images have the same size
        for item in batch:
            batch_label.append(item["label"])
            image = torch.tensor(item["processed_image"], dtype=cast_dtype, device=device)
            batch_images.append(image)
            if random.randint(0, 1) == 0:
                latent = self.encode_func(image.unsqueeze(0))
            else:
                latent = self.encode_func(hflip(image).unsqueeze(0))
            B, C, H, W = latent.shape
            assert B == 1, "Batch size should be 1"
            assert C * H * W == item["num_tokens"], "Number of tokens should match"
            latent = latent.squeeze(0)
            latent = rearrange(latent, "c (h p1) (w p2) -> (h w) c p1 p2", p1=self.patch_size, p2=self.patch_size)
            packed_latent.append(latent)
            hw_list.append([H // self.patch_size, W // self.patch_size])
        packed_latent = torch.concat(packed_latent)
        noises = torch.randn_like(packed_latent)
        label = torch.tensor(batch_label)
        hw_list = torch.tensor(hw_list, dtype=torch.int32)

        # # Move tensors to model device for FSDP2 compatibility
        # device = next(self.model.parameters()).device
        # packed_latent = packed_latent.to(device)
        # noises = noises.to(device)
        # label = label.to(device)
        # hw_list = hw_list.to(device)

        model_kwargs = {
            "y": label,
            "hw_list": hw_list,
        }
        batch_size = label.shape[0]

        with torch.autocast(device_type="cuda", dtype=cast_dtype):
            fm_loss, proj_loss = self.loss_fn(
                self.model, batch_size, packed_latent, noises, model_kwargs, use_dir_loss=True
            )
            loss = fm_loss + proj_loss * self.proj_loss_weight
            return loss
