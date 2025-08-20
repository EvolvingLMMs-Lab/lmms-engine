# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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


from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig


class AeroConfig(PretrainedConfig):
    model_type = "aero"
    sub_configs = {
        "text_config": AutoConfig,
        "audio_config": AutoConfig,
    }

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        audio_token_index=151648,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.audio_token_index = audio_token_index

        if isinstance(text_config, dict):
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "qwen2"
            )
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

        self.text_config = text_config

        if isinstance(audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"]
                if "model_type" in audio_config
                else "qwen2_audio_encoder"
            )
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["qwen2_audio_encoder"](
                d_model=1280,
                encoder_attention_heads=20,
                encoder_ffn_dim=5120,
                encoder_layerdrop=0.0,
                encoder_layers=32,
                num_mel_bins=128,
                max_source_positions=1500,
                scale_embedding=False,
                activation_function="gelu",
            )

        self.audio_config = audio_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
