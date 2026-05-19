# coding=utf-8
# Copyright 2025 LMMs-Lab team. All rights reserved.
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

"""AeroRealtime model configuration.

A multimodal model combining audio, vision, and language capabilities.
Inspired by VoxtralRealtime but extended with vision support for
image and video inputs.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig

from .backbone_registry import (
    BACKBONE_FAMILY,
    family_text_config_cls,
    family_text_model_type,
    family_vision_config_cls,
    family_vision_model_type,
)


class AeroRealtimeConfig(PretrainedConfig):
    r"""
    Configuration class for AeroRealtime model.

    AeroRealtime is a multimodal model that combines an audio encoder,
    a vision encoder, and a language model with an audio projector.

    Args:
        text_config (`dict` or `PretrainedConfig`, *optional*):
            Configuration for the language model backbone. When passed as a dict,
            it is resolved via `CONFIG_MAPPING` using the `model_type` key.
            Defaults to a Qwen3-style config if not provided.
        audio_config (`dict` or `PretrainedConfig`, *optional*):
            Configuration for the audio encoder tower. When passed as a dict,
            it is resolved via `CONFIG_MAPPING` using the `model_type` key.
            Defaults to a Qwen2 audio encoder config if not provided.
        vision_config (`dict` or `PretrainedConfig`, *optional*):
            Configuration for the vision encoder tower. When passed as a dict,
            it is resolved via `CONFIG_MAPPING` using the `model_type` key.
            Defaults to a Qwen3 VL vision config if not provided.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function used in the audio multi-modal projector.
        audio_length_per_tok (`int`, *optional*, defaults to `8`):
            Number of audio feature frames per text token.
        downsample_factor (`int`, *optional*, defaults to `4`):
            Factor by which audio features are downsampled (concatenated) before
            projection. The audio projector input dimension is
            `audio_config.hidden_size * downsample_factor`.
        audio_token_index (`int`, *optional*, defaults to `151671`):
            Token index for ``<|audio_pad|>`` — placeholder for audio features
            in the input sequence.
        audio_start_token_index (`int`, *optional*, defaults to `151669`):
            Token index for ``<|audio_start|>`` — opens an audio segment.
            Also exposed as ``audio_start_token_id``.
        audio_end_token_index (`int`, *optional*, defaults to `151670`):
            Token index for ``<|audio_end|>`` — closes an audio segment.
            Also exposed as ``audio_end_token_id``.
        image_token_index (`int`, *optional*, defaults to `151655`):
            Token index used as a placeholder for image features in the input sequence.
            Also exposed as ``image_token_id`` for compatibility with shared RoPE helpers.
        video_token_index (`int`, *optional*, defaults to `151656`):
            Token index used as a placeholder for video features in the input sequence.
            Also exposed as ``video_token_id`` for compatibility with shared RoPE helpers.
        vision_start_token_index (`int`, *optional*, defaults to `151652`):
            Token index for ``<|vision_start|>``. Used by the shared mrope
            position index computation (``qwen3_vl_get_rope_index``).
            Also exposed as ``vision_start_token_id``.
        rt_start_token_index (`int`, *optional*, defaults to `151672`):
            Token index for ``<|rt_start|>`` — the first token of the realtime text stream.
        rt_pad_token_index (`int`, *optional*, defaults to `151673`):
            Token index for ``<|rt_pad|>`` — silence token in the realtime text stream.
        rt_speak_token_index (`int`, *optional*, defaults to `151674`):
            Token index for ``<|rt_speak|>`` — delay boundary marker after which
            the model may begin producing text.
        rt_end_token_index (`int`, *optional*, defaults to `151675`):
            Token index for ``<|rt_end|>`` — closes one realtime speech span.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the language model's input and output word embeddings.

    Example:
        ```python
        from lmms_engine.models.aero_realtime import AeroRealtimeConfig

        config = AeroRealtimeConfig(
            text_config={"model_type": "qwen3", "hidden_size": 3584},
            audio_config={"model_type": "qwen2_audio_encoder"},
            vision_config={"hidden_size": 1152, "out_hidden_size": 3584},
        )
        ```
    """

    model_type = "aero_realtime"

    sub_configs = {
        "text_config": AutoConfig,
        "audio_config": AutoConfig,
        "vision_config": AutoConfig,
    }

    def __init__(
        self,
        backbone_family: str = "qwen3_vl",
        text_config=None,
        audio_config=None,
        vision_config=None,
        projector_hidden_act="gelu",
        audio_length_per_tok=8,
        downsample_factor=4,
        audio_token_index=151671,
        audio_start_token_index=151669,
        audio_end_token_index=151670,
        image_token_index=151655,
        video_token_index=151656,
        vision_start_token_index=151652,
        rt_start_token_index=151672,
        rt_pad_token_index=151673,
        rt_speak_token_index=151674,
        rt_end_token_index=151675,
        tie_word_embeddings=False,
        **kwargs,
    ):
        # --- Resolve backbone_family ---
        # Defaults to "qwen3_vl" so legacy ckpts (saved before this field
        # existed) and HF's no-arg ``AeroRealtimeConfig()`` (called by
        # ``to_diff_dict``) keep working.
        if backbone_family not in BACKBONE_FAMILY:
            raise ValueError(
                f"unknown backbone_family={backbone_family}; " f"expected one of {sorted(BACKBONE_FAMILY)}"
            )
        self.backbone_family = backbone_family
        text_mt = family_text_model_type(backbone_family)
        vision_mt = family_vision_model_type(backbone_family)

        # --- Plain field assignments (unchanged from previous version) ---
        self.projector_hidden_act = projector_hidden_act
        self.audio_length_per_tok = audio_length_per_tok
        self.downsample_factor = downsample_factor
        self.audio_token_index = audio_token_index
        self.audio_start_token_index = audio_start_token_index
        self.audio_end_token_index = audio_end_token_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.vision_start_token_index = vision_start_token_index
        self.rt_start_token_index = rt_start_token_index
        self.rt_pad_token_index = rt_pad_token_index
        self.rt_speak_token_index = rt_speak_token_index
        self.rt_end_token_index = rt_end_token_index
        # Aliases for shared rope helper
        self.image_token_id = image_token_index
        self.video_token_id = video_token_index
        self.vision_start_token_id = vision_start_token_index
        self.audio_token_id = audio_token_index
        self.audio_start_token_id = audio_start_token_index
        self.audio_end_token_id = audio_end_token_index

        # --- Resolve text_config (registry-driven) ---
        # Legacy ckpts may have written text_config.model_type=<family> instead
        # of "<family>_text" — accept both for back-compat.
        text_cfg_cls = family_text_config_cls(backbone_family)
        if isinstance(text_config, dict):
            mt = text_config.get("model_type", text_mt)
            if mt not in (text_mt, backbone_family):
                raise ValueError(
                    f"text_config.model_type={mt} does not match family " f"{backbone_family} (expected {text_mt})"
                )
            text_config = dict(text_config)
            text_config.pop("model_type", None)
            text_config = text_cfg_cls(**text_config)
        elif text_config is None:
            text_config = text_cfg_cls()
        self.text_config = text_config

        # --- Resolve audio_config ---
        # Defaults to voxtral_realtime_encoder. When audio_config is None we
        # let the Voxtral encoder config supply its own defaults — do not
        # duplicate kwargs here (keeps us in sync with upstream).
        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "voxtral_realtime_encoder")
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["voxtral_realtime_encoder"]()
        self.audio_config = audio_config

        # --- Resolve vision_config (registry-driven) ---
        # Legacy ckpts wrote vision_config.model_type=<family> (e.g. "qwen3_vl")
        # instead of the proper "<family>_vision" — accept both for back-compat.
        vision_cfg_cls = family_vision_config_cls(backbone_family)
        if isinstance(vision_config, dict):
            mt = vision_config.get("model_type", vision_mt)
            if mt not in (vision_mt, backbone_family):
                raise ValueError(
                    f"vision_config.model_type={mt} does not match family " f"{backbone_family} (expected {vision_mt})"
                )
            vision_config = dict(vision_config)
            vision_config.pop("model_type", None)
            vision_config = vision_cfg_cls(**vision_config)
        elif vision_config is None:
            vision_config = vision_cfg_cls()
        self.vision_config = vision_config

        # Expose text hidden size at top level for convenience
        self.hidden_size = self.text_config.hidden_size
        self.audio_hidden_size = getattr(self.audio_config, "hidden_size", None) or getattr(
            self.audio_config, "d_model", None
        )

        if tie_word_embeddings is False and getattr(self.text_config, "tie_word_embeddings", False):
            tie_word_embeddings = True

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
