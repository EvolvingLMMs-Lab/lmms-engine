import random
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import Qwen2Tokenizer, set_seed

from lmms_engine.mapping_func import register_processor
from lmms_engine.models.bagel.data_utils import (
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    len2weight,
    patchify,
    prepare_attention_mask_per_sample,
)
from lmms_engine.models.bagel.transforms import ImageTransform

from .config import ProcessorConfig


@register_processor("bagel")
class BagelDataProcessor:
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config
        extra_kwargs = getattr(self.config, "extra_kwargs", None) or {}
        self.vit_patch_size = int(extra_kwargs.get("vit_patch_size", 14))
        self.max_num_patch_per_side = int(extra_kwargs.get("max_num_patch_per_side", 70))
        self.interpolate_pos = bool(extra_kwargs.get("interpolate_pos", False))
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate
        # Default latent * downsample = 2 * 8
        self.vae_image_downsample = int(extra_kwargs.get("vae_image_downsample", 16))
        self.max_latent_size = int(extra_kwargs.get("max_latent_size", 64))

        self.text_cond_dropout_prob = float(extra_kwargs.get("text_cond_dropout_prob", 0.0))
        self.vit_cond_dropout_prob = float(extra_kwargs.get("vit_cond_dropout_prob", 0.0))
        self.vae_cond_dropout_prob = float(extra_kwargs.get("vae_cond_dropout_prob", 0.0))

        self.user_image_as_vae_condition = extra_kwargs.get("user_image_as_vae_condition", None)
        self.set_random_seed(extra_kwargs.get("random_seed", 4396))

    def build(self):
        self.processor = self._build_processor()

    def set_random_seed(self, seed: int):
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0
        seed = seed * world_size + rank
        logger.info(f"Set random seed to {seed} for rank {rank} in world size {world_size}")
        set_seed(seed)

    def _build_processor(self):
        extra_kwargs = getattr(self.config, "extra_kwargs", None) or {}
        processor = Qwen2Tokenizer.from_pretrained(self.config.processor_name)
        processor, self.new_token_ids, self.num_new_tokens = self.add_special_tokens(processor)
        self.vae_image_stride = int(extra_kwargs.get("vae_image_stride", 16))
        self.vae_max_image_size = int(extra_kwargs.get("vae_max_image_size", 1024))
        self.vae_min_image_size = int(extra_kwargs.get("vae_min_image_size", 512))
        self.vae_image_transform = ImageTransform(
            image_stride=self.vae_image_stride,
            max_image_size=self.vae_max_image_size,
            min_image_size=self.vae_min_image_size,
        )
        self.vit_image_stride = int(extra_kwargs.get("vit_image_stride", 14))
        self.vit_max_image_size = int(extra_kwargs.get("vit_max_image_size", 512))
        self.vit_min_image_size = int(extra_kwargs.get("vit_min_image_size", 378))
        self.vit_max_pixels = int(extra_kwargs.get("vit_max_pixels", 2_007_040))
        self.vit_image_transform = ImageTransform(
            image_stride=self.vit_image_stride,
            max_image_size=self.vit_max_image_size,
            min_image_size=self.vit_min_image_size,
            max_pixels=self.vit_max_pixels,
        )
        return processor

    def save_pretrained(self, save_directory: str):
        if not hasattr(self, "processor"):
            raise ValueError("Processor has not been built yet. Please call build() first.")
        # Build a clean processor for saving
        new_processor = self._build_processor()
        new_processor.save_pretrained(save_directory)

    def process(
        self,
        images: List[Image.Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        add_system_prompt=True,
        **kwargs,
    ):
        """
        A wrapper method to process single data
        """
        image_index = 0
        # text = []
        # vae_images = []
        # vit_images = []
        sequence_status = self.set_sequence_status()
        # process_order = []
        curr = 0
        curr_rope_id = 0
        full_attn_modes = []
        split_lens = []

        use_user_vae_cond = self.user_image_as_vae_condition
        if use_user_vae_cond is None:
            has_assistant_image = any(
                (m.get("role") == "assistant") and any((c.get("type") == "image") for c in (m.get("content") or []))
                for m in hf_messages
            )
            use_user_vae_cond = has_assistant_image

        for message in hf_messages:
            role = message["role"]
            for content in message["content"]:
                ctype = content.get("type")
                if ctype == "text":
                    if role != "assistant" and self.text_cond_dropout_prob > 0:
                        if random.random() < self.text_cond_dropout_prob:
                            continue
                    curr_split_len = 0
                    curr_text = content["text"]
                    attn_modes, sequence_status, curr, curr_split_len, curr_rope_id = self.process_text(
                        curr_text,
                        role,
                        sequence_status,
                        curr,
                        curr_split_len,
                        curr_rope_id,
                    )
                    full_attn_modes.extend(attn_modes)
                    split_lens.append(curr_split_len)
                    continue
                if ctype == "image":
                    curr_image = images[image_index]
                    image_index += 1
                    if role == "assistant":
                        curr_split_len = 0
                        attn_modes, sequence_status, curr, curr_split_len, curr_rope_id = self.process_vae_image(
                            curr_image,
                            role,
                            sequence_status,
                            curr_rope_id=curr_rope_id,
                            curr=curr,
                            curr_split_len=curr_split_len,
                            need_loss=True,
                        )
                        full_attn_modes.extend(attn_modes)
                        split_lens.append(curr_split_len)
                        continue
                    if role == "user":
                        if use_user_vae_cond:
                            if self.vae_cond_dropout_prob > 0 and random.random() < self.vae_cond_dropout_prob:
                                # Match original: when dropping a VAE-condition image, still advance the rope id.
                                curr_rope_id += 1
                            else:
                                curr_split_len = 0
                                (
                                    attn_modes,
                                    sequence_status,
                                    curr,
                                    curr_split_len,
                                    curr_rope_id,
                                ) = self.process_vae_image(
                                    curr_image,
                                    role,
                                    sequence_status,
                                    curr_rope_id=curr_rope_id,
                                    curr=curr,
                                    curr_split_len=curr_split_len,
                                    need_loss=False,
                                )
                                full_attn_modes.extend(attn_modes)
                                split_lens.append(curr_split_len)
                        if self.vit_cond_dropout_prob > 0 and random.random() < self.vit_cond_dropout_prob:
                            curr_rope_id += 1
                        else:
                            curr_split_len = 0
                            attn_modes, sequence_status, curr, curr_split_len, curr_rope_id = self.process_vit_image(
                                curr_image,
                                role,
                                sequence_status,
                                curr_rope_id=curr_rope_id,
                                curr=curr,
                                curr_split_len=curr_split_len,
                            )
                            full_attn_modes.extend(attn_modes)
                            split_lens.append(curr_split_len)
                        continue
                continue

        sequence_status["attn_modes"] = full_attn_modes
        sequence_status["curr"] = curr
        sequence_status["sample_lens"].append(sum(split_lens))
        sequence_status["nested_attention_masks"].append(prepare_attention_mask_per_sample(split_lens, full_attn_modes))
        data = self.to_tensor(sequence_status)
        # Fake input ids for packing
        data["input_ids"] = torch.zeros(data["sequence_length"], dtype=torch.long)
        data["attention_mask"] = torch.ones(data["sequence_length"], dtype=torch.long)
        data["curr"] = curr
        # data["split_lens"] = split_lens

        return data

    def process_text(
        self,
        text: str,
        role: str,
        sequence_status: dict,
        curr: int,
        curr_split_len: int,
        curr_rope_id: int,
    ):
        attn_modes = []
        text_ids = self.processor.encode(text)
        shifted_text_ids = [self.bos_token_id] + text_ids
        sequence_status["packed_text_ids"].extend(shifted_text_ids)
        sequence_status["packed_text_indexes"].extend(range(curr, curr + len(shifted_text_ids)))
        if role == "assistant":
            sequence_status["ce_loss_indexes"].extend(range(curr, curr + len(shifted_text_ids)))
            sequence_status["ce_loss_weights"].extend([len2weight(len(shifted_text_ids))] * len(shifted_text_ids))
            sequence_status["packed_label_ids"].extend(text_ids + [self.eos_token_id])
        curr += len(shifted_text_ids)
        curr_split_len += len(shifted_text_ids)

        # add a <|im_end|> token
        sequence_status["packed_text_ids"].append(self.eos_token_id)
        sequence_status["packed_text_indexes"].append(curr)
        curr += 1
        curr_split_len += 1

        # update sequence status
        attn_modes.append("causal")
        sequence_status["packed_position_ids"].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
        curr_rope_id += curr_split_len
        return attn_modes, sequence_status, curr, curr_split_len, curr_rope_id

    def process_vae_image(
        self,
        # image_tensor: torch.Tensor,
        image: Image.Image,
        role: str,
        sequence_status: dict,
        curr_rope_id: int,
        curr: int,
        curr_split_len: int,
        need_loss: bool = True,
    ):
        image_tensor = self.vae_image_transform(image.convert("RGB"))
        attn_modes = []
        # add a <|startofimage|> token
        sequence_status["packed_text_ids"].append(self.start_of_image)
        sequence_status["packed_text_indexes"].append(curr)
        curr += 1
        curr_split_len += 1

        # preprocess image
        sequence_status["vae_image_tensors"].append(image_tensor)
        sequence_status["packed_latent_position_ids"].append(
            self.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.vae_image_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
        )
        H, W = image_tensor.shape[1:]
        h = H // self.vae_image_downsample
        w = W // self.vae_image_downsample
        sequence_status["vae_latent_shapes"].append((h, w))

        num_img_tokens = w * h
        sequence_status["packed_vae_token_indexes"].extend(range(curr, curr + num_img_tokens))
        if need_loss:
            # Supervised image generation/editing target (align with original: loss=1).
            sequence_status["mse_loss_indexes"].extend(range(curr, curr + num_img_tokens))
            timestep = np.random.randn()
        else:
            # VAE-condition image (align with original: loss=0, timestep=-inf -> sigmoid=0 -> clean latent condition).
            timestep = float("-inf")

        sequence_status["packed_timesteps"].extend([timestep] * num_img_tokens)
        curr += num_img_tokens
        curr_split_len += num_img_tokens

        # add a <|endofimage|> token
        sequence_status["packed_text_ids"].append(self.end_of_image)
        sequence_status["packed_text_indexes"].append(curr)
        curr += 1
        curr_split_len += 1

        sequence_status["packed_position_ids"].extend([curr_rope_id] * (num_img_tokens + 2))
        if need_loss:
            attn_modes.append("noise")
        else:
            # Match original: vae_image(loss=0) uses full attention and advances rope id by 1.
            attn_modes.append("full")
            curr_rope_id += 1

        return attn_modes, sequence_status, curr, curr_split_len, curr_rope_id

    def process_vit_image(
        self,
        image: Image.Image,
        role: str,
        sequence_status: dict,
        curr_rope_id: int,
        curr: int,
        curr_split_len: int,
    ):
        attn_modes = []
        image_tensor = self.vit_image_transform(image.convert("RGB"))

        # add a <|vision_start|> token
        sequence_status["packed_text_ids"].append(self.start_of_image)
        sequence_status["packed_text_indexes"].append(curr)
        curr += 1
        curr_split_len += 1

        # add the image tensor
        vit_tokens = patchify(image_tensor, self.vit_patch_size)
        num_img_tokens = vit_tokens.shape[0]
        sequence_status["packed_vit_token_indexes"].extend(range(curr, curr + num_img_tokens))
        curr += num_img_tokens
        curr_split_len += num_img_tokens

        sequence_status["packed_vit_tokens"].append(vit_tokens)
        sequence_status["vit_token_seqlens"].append(num_img_tokens)
        sequence_status["packed_vit_position_ids"].append(
            self.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.vit_patch_size,
                max_num_patches_per_side=self.max_num_patch_per_side,
            )
        )

        # add a <|endofimage|> token
        sequence_status["packed_text_ids"].append(self.end_of_image)
        sequence_status["packed_text_indexes"].append(curr)

        # if item['special_token_loss'] == 1: # <|endofimage|> may have loss
        #     sequence_status['ce_loss_indexes'].append(curr)
        #     sequence_status['ce_loss_weights'].append(1.0)
        #     sequence_status['packed_label_ids'].append(item['special_token_label'])

        curr += 1
        curr_split_len += 1

        # update sequence status
        attn_modes.append("full")
        sequence_status["packed_position_ids"].extend([curr_rope_id] * curr_split_len)
        curr_rope_id += 1

        return attn_modes, sequence_status, curr, curr_split_len, curr_rope_id

    def set_sequence_status(self):
        sequence_status = dict(
            curr=0,
            sample_lens=list(),
            packed_position_ids=list(),
            nested_attention_masks=list(),
            split_lens=list(),
            attn_modes=list(),
            packed_text_ids=list(),
            packed_text_indexes=list(),
            packed_label_ids=list(),
            ce_loss_indexes=list(),
            ce_loss_weights=list(),
            vae_image_tensors=list(),
            packed_latent_position_ids=list(),
            vae_latent_shapes=list(),
            packed_vae_token_indexes=list(),
            packed_timesteps=list(),
            mse_loss_indexes=list(),
            packed_vit_tokens=list(),
            vit_token_seqlens=list(),
            packed_vit_position_ids=list(),
            packed_vit_token_indexes=list(),
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status["sample_lens"]),
            sample_lens=sequence_status["sample_lens"],
            packed_text_ids=torch.tensor(sequence_status["packed_text_ids"]),
            packed_text_indexes=torch.tensor(sequence_status["packed_text_indexes"]),
            packed_position_ids=torch.tensor(sequence_status["packed_position_ids"]),
        )

        data["nested_attention_masks"] = sequence_status["nested_attention_masks"]

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status["vae_image_tensors"]) > 0:
            image_tensors = sequence_status.pop("vae_image_tensors")
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, : image_tensor.shape[1], : image_tensor.shape[2]] = image_tensor

            data["padded_images"] = padded_images
            data["patchified_vae_latent_shapes"] = sequence_status["vae_latent_shapes"]
            data["packed_latent_position_ids"] = torch.cat(sequence_status["packed_latent_position_ids"], dim=0)
            data["packed_vae_token_indexes"] = torch.tensor(sequence_status["packed_vae_token_indexes"])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status["packed_vit_tokens"]) > 0:
            data["packed_vit_tokens"] = torch.cat(sequence_status["packed_vit_tokens"], dim=0)
            data["packed_vit_position_ids"] = torch.cat(sequence_status["packed_vit_position_ids"], dim=0)
            data["packed_vit_token_indexes"] = torch.tensor(sequence_status["packed_vit_token_indexes"])
            data["vit_token_seqlens"] = torch.tensor(sequence_status["vit_token_seqlens"])

        # if the model is required to perform visual generation
        if len(sequence_status["packed_timesteps"]) > 0:
            data["packed_timesteps"] = torch.tensor(sequence_status["packed_timesteps"])
            data["mse_loss_indexes"] = torch.tensor(sequence_status["mse_loss_indexes"])

        # if the model is required to perform text generation
        if len(sequence_status["packed_label_ids"]) > 0:
            data["packed_label_ids"] = torch.tensor(sequence_status["packed_label_ids"])
            data["ce_loss_indexes"] = torch.tensor(sequence_status["ce_loss_indexes"])
            data["ce_loss_weights"] = torch.tensor(sequence_status["ce_loss_weights"])

        return data

    @property
    def image_token_id(self):
        image_token = getattr(self.processor, "image_token", None)
        if image_token is None:
            return None
        else:
            return self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)

    @property
    def video_token_id(self):
        video_token = getattr(self.processor, "video_token", None)
        if video_token is None:
            return None
        else:
            return self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)

    @property
    def tokenizer(self):
        return self.tokenizer

    def add_special_tokens(self, tokenizer):
        all_special_tokens = []
        for k, v in tokenizer.special_tokens_map.items():
            if isinstance(v, str):
                all_special_tokens.append(v)
            elif isinstance(v, list):
                all_special_tokens += v

        new_tokens = []

        if "<|im_start|>" not in all_special_tokens:
            new_tokens.append("<|im_start|>")

        if "<|im_end|>" not in all_special_tokens:
            new_tokens.append("<|im_end|>")

        if "<|vision_start|>" not in all_special_tokens:
            new_tokens.append("<|vision_start|>")

        if "<|vision_end|>" not in all_special_tokens:
            new_tokens.append("<|vision_end|>")

        num_new_tokens = tokenizer.add_tokens(new_tokens)
        bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.bos_token_id = bos_token_id
        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_token_id = eos_token_id
        start_of_image = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        end_of_image = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.start_of_image = start_of_image
        self.end_of_image = end_of_image

        new_token_ids = dict(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            start_of_image=start_of_image,
            end_of_image=end_of_image,
        )

        return tokenizer, new_token_ids, num_new_tokens
