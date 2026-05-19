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

"""Prepare initial weights for AeroRealtime model.

Initialises the AeroRealtime model from:
- Vision tower: Qwen3-VL (e.g. Qwen/Qwen3-VL-4B-Instruct)
- Audio tower: Qwen2Audio encoder (from Qwen/Qwen2-Audio-7B-Instruct)
- Language model: Qwen3-VL text backbone (shared with vision model)
- Audio projector: randomly initialised

Usage:
    python tools/prepare_init_weight/prepare_aero_realtime.py \
        --vision_model_id Qwen/Qwen3-VL-4B-Instruct \
        --audio_model_id Qwen/Qwen2-Audio-7B-Instruct \
        --output_dir ./data/aero_realtime_init
"""

import argparse
import gc
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Qwen2AudioForConditionalGeneration,
    WhisperFeatureExtractor,
)

from lmms_engine.models.aero_realtime.configuration_aero_realtime import (
    AeroRealtimeConfig,
)
from lmms_engine.models.aero_realtime.modeling_aero_realtime import (
    AeroRealtimeForConditionalGeneration,
)
from lmms_engine.models.aero_realtime.processing_aero_realtime import (
    AeroRealtimeProcessor,
)


def main(args):
    torch.set_default_dtype(torch.float16)

    vision_text_config = AutoConfig.from_pretrained(args.vision_model_id)

    if args.skip_weights:
        print("--skip_weights set: only regenerating tokenizer / config / processor metadata.")
        vision_text_model = None
        # Load audio config without instantiating the heavy model
        audio_config_full = AutoConfig.from_pretrained(args.audio_model_id)
        audio_config = audio_config_full.audio_config
        audio_model = None
    else:
        print(f"Loading vision/text model from {args.vision_model_id} (backbone_family={args.backbone_family})...")
        vision_text_model = AutoModelForImageTextToText.from_pretrained(
            args.vision_model_id,
            torch_dtype="auto",
            device_map="cpu",
        )

        print(f"Loading audio model from {args.audio_model_id}...")
        audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.audio_model_id,
            torch_dtype="auto",
            device_map="cpu",
        )
        audio_config = audio_model.config.audio_config

    # ----------------------------------------------------------------
    # Build AeroRealtime config
    # ----------------------------------------------------------------
    text_config = vision_text_config.text_config
    vision_config = vision_text_config.vision_config

    # Resolve token indices from the Qwen3 VL tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.vision_model_id, use_fast=True)
    # Qwen3 VL already has these special tokens
    image_token_index = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_index = tokenizer.convert_tokens_to_ids("<|video_pad|>")

    # Add new audio + realtime special tokens (in fixed id order)
    from transformers import AddedToken

    new_special_tokens = [
        "<|audio_start|>",  # 151673
        "<|audio_end|>",  # 151674
        "<|audio_pad|>",  # 151675
        "<|rt_start|>",
        "<|rt_pad|>",
        "<|rt_speak|>",
        "<|rt_end|>",
    ]
    for tok in new_special_tokens:
        if tok not in tokenizer.get_vocab():
            tokenizer.add_tokens(AddedToken(tok, special=True, normalized=False), special_tokens=True)
            print(f"Added {tok} token to tokenizer")

    audio_token_index = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    audio_start_token_index = tokenizer.convert_tokens_to_ids("<|audio_start|>")
    audio_end_token_index = tokenizer.convert_tokens_to_ids("<|audio_end|>")
    rt_start_token_index = tokenizer.convert_tokens_to_ids("<|rt_start|>")
    rt_pad_token_index = tokenizer.convert_tokens_to_ids("<|rt_pad|>")
    rt_speak_token_index = tokenizer.convert_tokens_to_ids("<|rt_speak|>")
    rt_end_token_index = tokenizer.convert_tokens_to_ids("<|rt_end|>")

    print(
        f"Token indices: image={image_token_index}, video={video_token_index}, "
        f"audio_pad={audio_token_index}, audio_start={audio_start_token_index}, "
        f"audio_end={audio_end_token_index}"
    )
    print(
        f"  rt_start={rt_start_token_index}, rt_pad={rt_pad_token_index}, "
        f"rt_speak={rt_speak_token_index}, rt_end={rt_end_token_index}"
    )

    config = AeroRealtimeConfig(
        text_config=text_config,
        audio_config=audio_config,
        vision_config=vision_config,
        image_token_index=image_token_index,
        video_token_index=video_token_index,
        audio_token_index=audio_token_index,
        audio_start_token_index=audio_start_token_index,
        audio_end_token_index=audio_end_token_index,
        rt_start_token_index=rt_start_token_index,
        rt_pad_token_index=rt_pad_token_index,
        rt_speak_token_index=rt_speak_token_index,
        rt_end_token_index=rt_end_token_index,
        downsample_factor=args.downsample_factor,
        backbone_family=args.backbone_family,
    )

    # ----------------------------------------------------------------
    # Create model and load weights (skipped when --skip_weights)
    # ----------------------------------------------------------------
    if not args.skip_weights:
        print("Creating AeroRealtime model...")
        with init_empty_weights():
            model = AeroRealtimeForConditionalGeneration(config)

        # Load vision tower from Qwen3 VL
        print("Loading vision tower weights...")
        model.vision_tower = vision_text_model.model.visual

        # Load language model (Qwen3VLTextModel) from Qwen3 VL
        print("Loading language model weights...")
        model.language_model = vision_text_model.model.language_model

        # Load lm_head from Qwen3 VL
        print("Loading lm_head weights...")
        model.lm_head = vision_text_model.lm_head

        # Load audio tower from Qwen2Audio
        print("Loading audio tower weights...")
        model.audio_tower = audio_model.audio_tower

        # Randomly init projector
        print("Initialising audio projector...")
        std = getattr(config.text_config, "initializer_range", 0.02)

        # Re-create projector on CPU (not meta device)
        from lmms_engine.models.aero_realtime.modeling_aero_realtime import (
            AeroRealtimeMultiModalProjector,
        )

        model.multi_modal_projector = AeroRealtimeMultiModalProjector(config)
        model.multi_modal_projector.linear_1.weight.data.normal_(mean=0.0, std=std)
        model.multi_modal_projector.linear_2.weight.data.normal_(mean=0.0, std=std)

        # ------------------------------------------------------------
        # Handle vocab expansion for the new audio token
        # ------------------------------------------------------------
        orig_vocab_size = text_config.vocab_size
        new_vocab_size = len(tokenizer)

        if new_vocab_size > orig_vocab_size:
            print(f"Expanding embeddings from {orig_vocab_size} to {new_vocab_size}...")
            pad_shape = 64
            model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=pad_shape)

            # Initialise new token embeddings from the distribution of existing ones
            pre_expansion_embeddings = model.language_model.embed_tokens.weight.data[:orig_vocab_size]
            mu = torch.mean(pre_expansion_embeddings, dim=0).float()
            n = pre_expansion_embeddings.size(0)
            sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
            dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

            num_new = model.language_model.embed_tokens.weight.data[orig_vocab_size:].shape[0]
            model.language_model.embed_tokens.weight.data[orig_vocab_size:] = torch.stack(
                [dist.sample() for _ in range(num_new)], dim=0
            )
            num_new_head = model.lm_head.weight.data[orig_vocab_size:].shape[0]
            model.lm_head.weight.data[orig_vocab_size:] = torch.stack(
                [dist.sample() for _ in range(num_new_head)], dim=0
            )

        model.eval()
    else:
        model = None

    # ----------------------------------------------------------------
    # Build processor
    # ----------------------------------------------------------------
    print("Building processor...")
    vision_processor = AutoProcessor.from_pretrained(args.vision_model_id)

    # WhisperFeatureExtractor with 128 mel bins for Qwen2Audio compatibility
    feature_extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
    )

    processor = AeroRealtimeProcessor(
        image_processor=vision_processor.image_processor,
        video_processor=vision_processor.video_processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        downsample_factor=args.downsample_factor,
        rt_start_token="<|rt_start|>",
        rt_pad_token="<|rt_pad|>",
        rt_speak_token="<|rt_speak|>",
        rt_end_token="<|rt_end|>",
    )

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if model is not None:
        print(f"Saving model to {args.output_dir}...")
        model.save_pretrained(args.output_dir)
    else:
        print(f"Saving config only (no weights) to {args.output_dir}...")
        config.save_pretrained(args.output_dir)
    print(f"Saving processor to {args.output_dir}...")
    processor.save_pretrained(args.output_dir)

    # Print summary
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nDone! Model saved to {args.output_dir}")
        print(f"Total parameters: {total_params / 1e9:.2f}B")
        print(f"  Vision tower: {sum(p.numel() for p in model.vision_tower.parameters()) / 1e6:.0f}M")
        print(f"  Audio tower:  {sum(p.numel() for p in model.audio_tower.parameters()) / 1e6:.0f}M")
        print(f"  Language model: {sum(p.numel() for p in model.language_model.parameters()) / 1e6:.0f}M")
        print(f"  Audio projector: {sum(p.numel() for p in model.multi_modal_projector.parameters()) / 1e6:.1f}M")
    else:
        print(f"\nDone! Metadata (config + tokenizer + processor) saved to {args.output_dir}")
        print("(model.safetensors untouched)")

    # Clean up
    del model, vision_text_model, audio_model, processor
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AeroRealtime initial weights")
    parser.add_argument(
        "--backbone_family",
        type=str,
        required=True,
        choices=["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"],
        help="Backbone family for the vision+text source checkpoint. "
        "Propagated into the generated AeroRealtimeConfig.",
    )
    parser.add_argument(
        "--vision_model_id",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="HuggingFace model ID for the vision+text backbone (Qwen3 VL).",
    )
    parser.add_argument(
        "--audio_model_id",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="HuggingFace model ID for the audio encoder (Qwen2Audio).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/aero_realtime_init",
        help="Directory to save the initialised model.",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=4,
        help="Audio downsampling factor for the projector.",
    )
    parser.add_argument(
        "--skip_weights",
        action="store_true",
        help="Skip loading Qwen3-VL / Qwen2-Audio weights and saving model.safetensors. "
        "Only regenerate tokenizer / config.json / processor_config.json / chat_template.",
    )
    args = parser.parse_args()

    main(args)
