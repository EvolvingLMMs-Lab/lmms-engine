# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert LLaVa-Onevision checkpoints from the original repository.

URL: https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main

"""

import argparse
import gc
import glob
import json
import os
from pathlib import Path

import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors import safe_open
from transformers import (
    AddedToken,
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2AudioEncoder,
    Qwen2AudioForConditionalGeneration,
)

from lmms_engine.models.aero.configuration_aero import AeroConfig
from lmms_engine.models.aero.modeling_aero import (
    PROJECTOR_MAP,
    AeroForConditionalGeneration,
)
from lmms_engine.models.aero.processing_aero import AeroProcessor


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def convert_llava_to_hf(model_id, pytorch_dump_folder_path, audio_model_id, with_out_init=False):
    if not with_out_init:
        # load original config
        text_model_id = model_id

        torch.set_default_dtype(torch.float16)
        text_config = AutoConfig.from_pretrained(text_model_id)

        tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=True)
        tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
        tokenizer.add_tokens(AddedToken("<video>", special=True, normalized=False), special_tokens=True)
        tokenizer.add_tokens(AddedToken("<|AUDIO|>", special=True, normalized=False), special_tokens=True)
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        video_token_id = tokenizer.convert_tokens_to_ids("<video>")
        audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")

        audio_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        processor = AeroProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor.feature_extractor,
        )

        if "Omni" in audio_model_id:
            audio_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                audio_model_id,
                torch_dtype="auto",
                device_map="cuda:0",
            )
            audio_tower = audio_model.thinker.audio_tower
            audio_config = audio_model.config.thinker_config.audio_config
        else:
            audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                audio_model_id,
                torch_dtype="auto",
                device_map="cuda:0",
            )
            audio_tower = audio_model.audio_tower
            audio_config = audio_model.config.audio_config

        config = AeroConfig(
            text_config=text_config,
            audio_token_index=audio_token_id,
            audio_config=audio_config,
        )

        with init_empty_weights():
            model = AeroForConditionalGeneration(config)
        text_model = AutoModelForCausalLM.from_pretrained(text_model_id, torch_dtype="auto", device_map="cuda:0")
        audio_modal_projector = PROJECTOR_MAP[model.audio_tower_type](config)
        std = config.initializer_range if hasattr(config, "initializer_range") else config.text_config.initializer_range
        audio_modal_projector.linear.weight.data.normal_(mean=0.0, std=std)
        audio_modal_projector.linear.bias.data.zero_()
        model.audio_modal_projector = audio_modal_projector
        model.audio_tower = audio_tower
        model.language_model = text_model
        model.eval()

        pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
        mu = torch.mean(pre_expansion_embeddings, dim=0).float()
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

        # We add an image token so we resize the model
        # Pad to 64 for performance reasons
        # Qwen-based models have extra unused space in the vocab size already, so no need to resize
        pad_shape = 64
        vocab_size = config.text_config.vocab_size
        num_tokens = vocab_size + 3
        model.resize_token_embeddings(num_tokens, pad_to_multiple_of=pad_shape)
        model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
            tuple(
                (
                    dist.sample()
                    for _ in range(model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0])
                )
            ),
            dim=0,
        )
        model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
            tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
            dim=0,
        )

        print(f"Saving model and processor for {model_id} to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
        # print("Init Success, Push to Hub...")
        # model.push_to_hub(pytorch_dump_folder_path, private=True)
        # processor.push_to_hub(pytorch_dump_folder_path, private=True)

        # Make space so we can load the model properly now.
        del audio_model, model, processor, audio_modal_projector, text_model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="lmms-lab/llava-onevision-qwen2-0.5b-ov",
        required=False,
    )
    parser.add_argument(
        "--audio_model_id",
        help="Hub location of the audio model to convert",
        default="Qwen/Qwen2-Audio-7B-Instruct",
        required=False,
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--with_out_init",
        action="store_true",
        help="Whether init or not init but just debugging...",
    )
    args = parser.parse_args()

    convert_llava_to_hf(
        args.model_id,
        args.pytorch_dump_folder_path,
        args.audio_model_id,
        args.with_out_init,
    )
