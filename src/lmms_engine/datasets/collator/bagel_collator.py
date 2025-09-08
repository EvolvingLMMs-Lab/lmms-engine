import math
import random
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import transformers
from PIL import Image
from torch.nn.attention.flex_attention import and_masks, or_masks

from ...protocol import Processable


class DataConfig:
    def __init__(
        self,
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vae_cond_dropout_prob=0.1,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
        interpolate_pos=False,
        use_flex=True,
        max_num_tokens=16384,
    ):
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size
        self.interpolate_pos = interpolate_pos
        self.use_flex = use_flex
        


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (
            full_and_noise_seq_id[q_idx] >= 0
        )

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return ~(
            (noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])
        )

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ["full", "noise"] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == "noise" else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat(
        [torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]
    ).to(device)

    return and_masks(
        or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask
    )


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(
    img_h, img_w, patch_size, max_num_patches_per_side
):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(
    img_h, img_w, patch_size, max_num_patches_per_side
):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(
        1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side
    )
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (
        bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w
    ).flatten()
    return pos_ids


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros(
        (sample_len, sample_len), dtype=torch.bool, device=device
    )

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ["causal", "full", "noise"]
        if attn_mode == "causal":
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones(
                (s, s), device=device
            ).tril()
            attention_mask[csum : csum + s, :csum] = 1
        else:
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
            attention_mask[csum : csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    return attention_mask


def split_integer_exp_decay(S, ng_sample_decay=1.0):
    if ng_sample_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ng_sample_decay) / (1 - math.pow(ng_sample_decay, S))
        p = [base * math.pow(ng_sample_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i + 1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer):
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
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    start_of_image = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_of_image = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    new_token_ids = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        start_of_image=start_of_image,
        end_of_image=end_of_image,
    )

    return tokenizer, new_token_ids, num_new_tokens


def len2weight(x, loss_reduction="square"):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)


@dataclass
class BagelCollator:
    processor: Processable
    new_token_ids: dict | None = None
    data_config: DataConfig | None = None

    def __post_init__(self):
        if self.data_config is None:
            self.data_config = DataConfig()
        if self.new_token_ids:
            for k, v in self.new_token_ids.items():
                setattr(self, k, v)
        self.interpolate_pos = self.data_config.interpolate_pos
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate
        self.use_flex = self.data_config.use_flex
        self.max_num_tokens = self.data_config.max_num_tokens

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        return self.processor.tokenizer

    # def pad_sequence(self, input_ids, batch_first, padding_value):
    #     if self.tokenizer.padding_side == "left":
    #         input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    #     input_ids = torch.nn.utils.rnn.pad_sequence(
    #         input_ids, batch_first=batch_first, padding_value=padding_value
    #     )
    #     if self.tokenizer.padding_side == "left":
    #         input_ids = torch.flip(input_ids, [1])
    #     return input_ids

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample["image_tensor_list"]
        text_ids_list = sample["text_ids_list"]
        sequence_plan = sample["sequence_plan"]

        split_lens, attn_modes = list(), list()
        curr = sequence_status["curr"]
        curr_rope_id = 0
        sample_lens = 0

        for item in sequence_plan:
            split_start = item.get("split_start", True)
            if split_start:
                curr_split_len = 0

            if item["type"] == "text":
                text_ids = text_ids_list.pop(0)
                if (
                    item["enable_cfg"] == 1
                    and random.random() < self.data_config.text_cond_dropout_prob
                ):
                    continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status["packed_text_ids"].extend(shifted_text_ids)
                sequence_status["packed_text_indexes"].extend(
                    range(curr, curr + len(shifted_text_ids))
                )
                if item["loss"] == 1:
                    sequence_status["ce_loss_indexes"].extend(
                        range(curr, curr + len(shifted_text_ids))
                    )
                    sequence_status["ce_loss_weights"].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status["packed_label_ids"].extend(
                        text_ids + [self.eos_token_id]
                    )
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status["packed_text_ids"].append(self.eos_token_id)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:  # <|im_end|> may have loss
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(
                        item["special_token_label"]
                    )
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status["packed_position_ids"].extend(
                    range(curr_rope_id, curr_rope_id + curr_split_len)
                )
                curr_rope_id += curr_split_len

            elif item["type"] == "vit_image":
                image_tensor = image_tensor_list.pop(0)
                if (
                    item["enable_cfg"] == 1
                    and random.random() < self.data_config.vit_cond_dropout_prob
                ):
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token
                sequence_status["packed_text_ids"].append(self.start_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status["packed_vit_token_indexes"].extend(
                    range(curr, curr + num_img_tokens)
                )
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status["packed_vit_tokens"].append(vit_tokens)
                sequence_status["vit_token_seqlens"].append(num_img_tokens)
                sequence_status["packed_vit_position_ids"].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1),
                        image_tensor.size(2),
                        self.data_config.vit_patch_size,
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side,
                    )
                )

                # add a <|endofimage|> token
                sequence_status["packed_text_ids"].append(self.end_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:  # <|endofimage|> may have loss
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(
                        item["special_token_label"]
                    )
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status["packed_position_ids"].extend(
                    [curr_rope_id] * curr_split_len
                )
                curr_rope_id += 1

            elif item["type"] == "vae_image":
                image_tensor = image_tensor_list.pop(0)
                if (
                    item["enable_cfg"] == 1
                    and random.random() < self.data_config.vae_cond_dropout_prob
                ):
                    # FIXME fix vae dropout in video2video setting.
                    curr_rope_id += 1
                    continue

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
                        self.data_config.vae_image_downsample,
                        max_num_patches_per_side=self.data_config.max_latent_size,
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status["vae_latent_shapes"].append((h, w))

                num_img_tokens = w * h
                sequence_status["packed_vae_token_indexes"].extend(
                    range(curr, curr + num_img_tokens)
                )
                if item["loss"] == 1:
                    sequence_status["mse_loss_indexes"].extend(
                        range(curr, curr + num_img_tokens)
                    )
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float("-inf")

                sequence_status["packed_timesteps"].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                # add a <|endofimage|> token
                sequence_status["packed_text_ids"].append(self.end_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                # <|endofimage|> may have loss
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(
                        item["special_token_label"]
                    )
                curr += 1
                curr_split_len += 1

                # update sequence status
                if split_start:
                    if item["loss"] == 1 and "frame_delta" not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                sequence_status["packed_position_ids"].extend(
                    [curr_rope_id] * (num_img_tokens + 2)
                )
                if "frame_delta" in item.keys():
                    curr_rope_id += item["frame_delta"]
                elif item["loss"] == 0:
                    curr_rope_id += 1

            if item.get("split_end", True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        sequence_status["curr"] = curr
        sequence_status["sample_lens"].append(sample_lens)
        # prepare attention mask
        if not self.use_flex:
            sequence_status["nested_attention_masks"].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status["split_lens"].extend(split_lens)
            sequence_status["attn_modes"].extend(attn_modes)

        return sequence_status

    def __call__(
        self, instances: Sequence[Dict] | Sequence[Sequence[Dict]]
    ) -> Dict[str, torch.Tensor]:
        final_instances = []
        for instance in instances:
            if isinstance(instance, list):
                final_instances.extend(instance)
            else:
                final_instances.append(instance)
        instances = final_instances

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

        for instance in instances:
            sequence_status = self.pack_sequence(instance, sequence_status)

        data = dict(
            sequence_length=sum(sequence_status["sample_lens"]),
            sample_lens=sequence_status["sample_lens"],
            packed_text_ids=torch.tensor(sequence_status["packed_text_ids"]),
            packed_text_indexes=torch.tensor(sequence_status["packed_text_indexes"]),
            packed_position_ids=torch.tensor(sequence_status["packed_position_ids"]),
        )
        if not self.use_flex:
            data["nested_attention_masks"] = sequence_status["nested_attention_masks"]
        else:
            sequence_len = data["sequence_length"]
            pad_len = self.max_num_tokens - sequence_len
            data["split_lens"] = sequence_status["split_lens"] + [pad_len]
            data["attn_modes"] = sequence_status["attn_modes"] + ["causal"]
            data["sample_lens"] += [pad_len]

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status["vae_image_tensors"]) > 0:
            image_tensors = sequence_status.pop("vae_image_tensors")
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[
                    i, :, : image_tensor.shape[1], : image_tensor.shape[2]
                ] = image_tensor

            data["padded_images"] = padded_images
            data["patchified_vae_latent_shapes"] = sequence_status["vae_latent_shapes"]
            data["packed_latent_position_ids"] = torch.cat(
                sequence_status["packed_latent_position_ids"], dim=0
            )
            data["packed_vae_token_indexes"] = torch.tensor(
                sequence_status["packed_vae_token_indexes"]
            )

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status["packed_vit_tokens"]) > 0:
            data["packed_vit_tokens"] = torch.cat(
                sequence_status["packed_vit_tokens"], dim=0
            )
            data["packed_vit_position_ids"] = torch.cat(
                sequence_status["packed_vit_position_ids"], dim=0
            )
            data["packed_vit_token_indexes"] = torch.tensor(
                sequence_status["packed_vit_token_indexes"]
            )
            data["vit_token_seqlens"] = torch.tensor(
                sequence_status["vit_token_seqlens"]
            )

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
        return self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
