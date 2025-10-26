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
    AutoProcessor,
    AutoTokenizer,
    Qwen2AudioEncoder,
    Qwen2AudioForConditionalGeneration,
)

from lmms_engine.models.kino.configuration_kino import KinoConfig
from lmms_engine.models.kino.modeling_kino import (
    KinoForConditionalGeneration,
    LlavaOnevisionAudioMultiModalProjector,
)
from lmms_engine.models.kino.processing_kino import KinoProcessor

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "vision_tower.",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "language_model.model.image_newline": "image_newline",
}

chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all video then #}{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}{{ '<video>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] }}{% endgeneration %}{% endfor %}{% endif %}{{'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def load_original_state_dict(model_id):
    directory_path = model_id

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    # tied wieghts so lm.head is not saved. Let's clone to load state dict
    if "lm_head.weight" not in original_state_dict:
        original_state_dict["lm_head.weight"] = original_state_dict["model.embed_tokens.weight"].clone()

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    # total_discrepancy = 0
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        # new_state_dict[key] = value.to(torch.float16)
        new_state_dict[key] = value
        # discrepancy = (new_state_dict[key] - value).sum().item()
        # total_discrepancy += discrepancy
        # print(f"Discrepancy on {key} : {discrepancy}")
    # print(f"Total Discrepancy : {total_discrepancy}")
    return new_state_dict


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def convert_llava_to_hf(model_id, pytorch_dump_folder_path, repo_id, push_to_hub=False, with_out_init=False):
    if not with_out_init:
        # load original config
        filepath = os.path.join(model_id, "config.json")
        # read json
        with open(filepath) as f:
            data = json.load(f)
            print(data)

        text_model_id = "Qwen/Qwen2.5-7B-Instruct"

        vision_model_id = data["mm_vision_tower"]
        torch.set_default_dtype(torch.float16)
        text_config = AutoConfig.from_pretrained(text_model_id)

        tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=True)
        tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
        tokenizer.add_tokens(AddedToken("<video>", special=True, normalized=False), special_tokens=True)
        tokenizer.add_tokens(AddedToken("<|AUDIO|>", special=True, normalized=False), special_tokens=True)
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        video_token_id = tokenizer.convert_tokens_to_ids("<video>")
        audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")

        qwen_vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        audio_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        processor = KinoProcessor(
            image_processor=qwen_vl_processor.image_processor,
            tokenizer=tokenizer,
            video_processor=qwen_vl_processor.image_processor,
            audio_processor=audio_processor.feature_extractor,
            num_image_tokens=None,
            vision_feature_select_strategy="navit",
        )
        vision_config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct").vision_config

        config = KinoConfig(
            vision_config=vision_config,
            text_config=text_config,
            projector_type="identity",
            vision_aspect_ratio="navit",
            image_token_index=image_token_id,
            video_token_index=video_token_id,
            audio_token_index=audio_token_id,
        )

        with init_empty_weights():
            model = KinoForConditionalGeneration(config)

        # load original state dict
        origin_state_dict = load_original_state_dict(model_id)
        state_dict = convert_state_dict_to_hf(origin_state_dict)
        audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )
        keys_in_models = set([k for k in state_dict.keys()])
        keys_in_new_models = set([k for k in model.state_dict().keys()])
        model.load_state_dict(state_dict, assign=True, strict=False)
        audio_modal_projector = LlavaOnevisionAudioMultiModalProjector(config)
        std = config.initializer_range if hasattr(config, "initializer_range") else config.text_config.initializer_range
        audio_modal_projector.linear.weight.data.normal_(mean=0.0, std=std)
        audio_modal_projector.linear.bias.data.zero_()
        model.audio_modal_projector = audio_modal_projector
        model.audio_tower = audio_model.audio_tower
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
        del state_dict
        gc.collect()

    # Load everything back for inference tests in float32 because prev script was written as that
    # Though it's mostly loaded in fp16 as original weights are in fp16
    model = KinoForConditionalGeneration.from_pretrained(
        pytorch_dump_folder_path, torch_dtype="float16", device_map="cuda:0"
    )
    processor = KinoProcessor.from_pretrained(pytorch_dump_folder_path)
    device = model.device

    # prepare inputs
    image = load_image()
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # verify generation
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
    )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print("Generated text:", repr(generated_text))

    if push_to_hub:
        model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)

    # verify batched generation
    # print("Batched generation...")
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # cats_image = Image.open(requests.get(url, stream=True).raw)

    # inputs = processor(
    # images=[image, cats_image],
    # text=[prompt, prompt],
    # padding=True,
    # return_tensors="pt",
    # )
    # inputs = {k: v.to(device) for k, v in inputs.items()}

    # for k, v in inputs.items():
    # print(k, v.shape)

    # print("Image sizes:", inputs["image_sizes"])

    # print("Batched generation...")
    # output_ids = model.generate(
    # **inputs,
    # max_new_tokens=1024,
    # use_cache=True,
    # )

    # outputs = processor.batch_decode(output_ids, skip_special_tokens=False)
    # print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="lmms-lab/llava-onevision-qwen2-0.5b-ov",
        required=False,
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="The repo id to push the mode",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub.",
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
        args.repo_id,
        args.push_to_hub,
        args.with_out_init,
    )
