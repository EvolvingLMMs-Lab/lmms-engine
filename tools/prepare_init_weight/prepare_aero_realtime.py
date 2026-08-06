# Copyright 2025 LMMs-Lab team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepare initial weights for AeroRealtime model.

Pulls submodule weights from independent sources:

- Vision tower + language model: Qwen3 VL / Qwen3.5 (``--vision_model_id``)
- Audio tower conv frontend: Qwen2-Audio (``--qwen2_audio_model_id``)
- Audio tower transformer layers: Qwen3-Omni (``--qwen3_omni_model_id``)
- Audio projector: random init

Audio tower weights are pulled directly from safetensors shards (no full
``from_pretrained``), so the 30B Qwen3-Omni checkpoint never gets fully
loaded into memory.

Example:
    python tools/prepare_init_weight/prepare_aero_realtime.py \\
        --backbone_family qwen3_vl \\
        --vision_model_id Qwen/Qwen3-VL-4B-Instruct \\
        --qwen2_audio_model_id Qwen/Qwen2-Audio-7B-Instruct \\
        --qwen3_omni_model_id /path/to/Qwen3-Omni-30B-A3B-Instruct \\
        --audio_padding_mode symmetric \\
        --output_dir /path/to/aero_init
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import (
    AddedToken,
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.models.voxtral_realtime.feature_extraction_voxtral_realtime import (
    VoxtralRealtimeFeatureExtractor,
)

from lmms_engine.models.aero_realtime.configuration_aero_realtime import (
    AeroRealtimeAudioEncoderConfig,
    AeroRealtimeConfig,
)
from lmms_engine.models.aero_realtime.modeling_aero_realtime import (
    AeroRealtimeForConditionalGeneration,
    AeroRealtimeMultiModalProjector,
)
from lmms_engine.models.aero_realtime.processing_aero_realtime import (
    AeroRealtimeProcessor,
)

# ---------------------------------------------------------------------------
# safetensors helpers
# ---------------------------------------------------------------------------


def _resolve_local_dir(model_id_or_path: str) -> Path:
    """Return a local directory for ``model_id_or_path``; download if needed."""
    p = Path(model_id_or_path)
    if p.exists() and p.is_dir():
        return p
    print(f"  snapshot_download({model_id_or_path}) ...")
    return Path(snapshot_download(model_id_or_path, allow_patterns=["*.safetensors", "*.json"]))


def _load_subset_from_safetensors(model_dir: Path, key_filter) -> dict[str, torch.Tensor]:
    """Load only the tensors whose key passes ``key_filter(key) -> bool``.

    Reads the safetensors index when present (sharded ckpt); falls back to
    iterating shards directly when no index is found.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            if key_filter(key):
                shards.setdefault(shard, []).append(key)
    else:
        shards = {p.name: None for p in sorted(model_dir.glob("*.safetensors"))}

    out: dict[str, torch.Tensor] = {}
    for shard_name, keys in shards.items():
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as f:
            iter_keys = keys if keys is not None else [k for k in f.keys() if key_filter(k)]
            for k in iter_keys:
                out[k] = f.get_tensor(k)
    return out


# ---------------------------------------------------------------------------
# audio tower rename tables
# ---------------------------------------------------------------------------


def _build_audio_state_dict(
    qwen2_audio_dir: Path,
    qwen3_omni_dir: Path,
) -> dict[str, torch.Tensor]:
    """Assemble audio_tower state_dict: conv from Qwen2-Audio, layers from Qwen3-Omni."""

    # --- conv frontend from Qwen2-Audio ---
    print("Pulling conv frontend from Qwen2-Audio...")

    def _q2a_filter(k: str) -> bool:
        return k.startswith("audio_tower.conv1.") or k.startswith("audio_tower.conv2.")

    q2a = _load_subset_from_safetensors(qwen2_audio_dir, _q2a_filter)
    if not q2a:
        raise RuntimeError(f"No audio_tower.conv* keys found in {qwen2_audio_dir}")

    state: dict[str, torch.Tensor] = {}
    for k, v in q2a.items():
        # audio_tower.conv1.weight -> embedder.conv1.weight
        new_k = k.replace("audio_tower.conv1.", "embedder.conv1.").replace("audio_tower.conv2.", "embedder.conv2.")
        state[new_k] = v

    # --- transformer layers + ln_post from Qwen3-Omni ---
    print("Pulling transformer layers + ln_post from Qwen3-Omni...")
    prefix = "thinker.audio_tower."

    def _q3o_filter(k: str) -> bool:
        if not k.startswith(prefix):
            return False
        tail = k[len(prefix) :]
        return tail.startswith("layers.") or tail.startswith("ln_post.")

    q3o = _load_subset_from_safetensors(qwen3_omni_dir, _q3o_filter)
    if not q3o:
        raise RuntimeError(f"No thinker.audio_tower.layers/ln_post keys found in {qwen3_omni_dir}")

    for k, v in q3o.items():
        tail = k[len(prefix) :]
        # ln_post -> norm
        if tail.startswith("ln_post."):
            new_k = tail.replace("ln_post.", "norm.", 1)
        else:
            # layers.N.self_attn.out_proj.* -> layers.N.self_attn.o_proj.*
            # layers.N.fc{1,2}.*           -> layers.N.mlp.fc{1,2}.*
            new_k = tail.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
            new_k = new_k.replace(".fc1.", ".mlp.fc1.").replace(".fc2.", ".mlp.fc2.")
        state[new_k] = v

    return state


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(args):
    torch.set_default_dtype(torch.float16)

    vision_text_config = AutoConfig.from_pretrained(args.vision_model_id)
    text_config = vision_text_config.text_config
    vision_config = vision_text_config.vision_config
    # Backbones declare tying at the top level only; the sub text_config falls
    # back to its class default, which is wrong for e.g. Qwen3-VL-30B-A3B.
    text_config.tie_word_embeddings = vision_text_config.tie_word_embeddings

    # ----- audio config: Qwen3-Omni layer dims + Qwen2-Audio mel + LayerNorm/GELU -----
    audio_config = AeroRealtimeAudioEncoderConfig(
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        num_key_value_heads=20,
        head_dim=64,
        num_mel_bins=128,
        max_position_embeddings=1500,
        activation_function="gelu",
        norm_type="layer_norm",
        mlp_type="gelu",
        conv_padding=args.audio_padding_mode,
        k_proj_bias=True,
        sliding_window=None,
        attention_window_left=-1,
        attention_window_right=-1,
    )

    # ----- tokenizer + token indices -----
    tokenizer = AutoTokenizer.from_pretrained(args.vision_model_id, use_fast=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_index = tokenizer.convert_tokens_to_ids("<|video_pad|>")

    new_special_tokens = [
        "<|audio_start|>",
        "<|audio_end|>",
        "<|audio_pad|>",
        "<|rt_start|>",
        "<|rt_pad|>",
        "<|rt_speak|>",
        "<|rt_end|>",
    ]
    for tok in new_special_tokens:
        if tok not in tokenizer.get_vocab():
            tokenizer.add_tokens(AddedToken(tok, special=True, normalized=False), special_tokens=True)
            print(f"Added {tok}")

    audio_token_index = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    audio_start_token_index = tokenizer.convert_tokens_to_ids("<|audio_start|>")
    audio_end_token_index = tokenizer.convert_tokens_to_ids("<|audio_end|>")
    rt_start_token_index = tokenizer.convert_tokens_to_ids("<|rt_start|>")
    rt_pad_token_index = tokenizer.convert_tokens_to_ids("<|rt_pad|>")
    rt_speak_token_index = tokenizer.convert_tokens_to_ids("<|rt_speak|>")
    rt_end_token_index = tokenizer.convert_tokens_to_ids("<|rt_end|>")

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

    if args.skip_weights:
        print("--skip_weights: saving metadata only.")
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(args.output_dir)
        _save_processor(args, tokenizer)
        return

    # ----- load vision+text model -----
    print(f"Loading vision/text model from {args.vision_model_id}...")
    vision_text_model = AutoModelForImageTextToText.from_pretrained(
        args.vision_model_id, torch_dtype="auto", device_map="cpu"
    )

    # ----- assemble audio tower state_dict from Qwen2-Audio + Qwen3-Omni -----
    qwen2_audio_dir = _resolve_local_dir(args.qwen2_audio_model_id)
    qwen3_omni_dir = _resolve_local_dir(args.qwen3_omni_model_id)
    audio_state = _build_audio_state_dict(qwen2_audio_dir, qwen3_omni_dir)

    # ----- build Aero model and graft submodules -----
    print("Creating AeroRealtime model on meta device...")
    with init_empty_weights():
        model = AeroRealtimeForConditionalGeneration(config)

    print("Grafting vision tower / language model / lm_head...")
    model.vision_tower = vision_text_model.model.visual
    model.language_model = vision_text_model.model.language_model
    model.lm_head = vision_text_model.lm_head

    print("Loading audio tower state_dict...")
    # audio_tower is still on meta; materialise it first.
    model.audio_tower.to_empty(device="cpu")
    missing, unexpected = model.audio_tower.load_state_dict(audio_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading audio_tower: {unexpected[:8]}...")
    # Anything missing must be either rotary_emb (no params) or a parameter we
    # explicitly skipped — fail loudly if anything else slipped through.
    bad_missing = [k for k in missing if "rotary_emb" not in k]
    if bad_missing:
        raise RuntimeError(f"Missing audio_tower keys after load: {bad_missing[:8]}...")
    print(f"  audio_tower loaded ({len(audio_state)} tensors; {len(missing)} missing rotary buffers)")

    print("Initialising audio projector (random)...")
    model.multi_modal_projector = AeroRealtimeMultiModalProjector(config)
    std = getattr(config.text_config, "initializer_range", 0.02)
    model.multi_modal_projector.linear_1.weight.data.normal_(mean=0.0, std=std)
    model.multi_modal_projector.linear_2.weight.data.normal_(mean=0.0, std=std)

    # ----- vocab expansion -----
    orig_vocab_size = text_config.vocab_size
    new_vocab_size = len(tokenizer)
    if new_vocab_size > orig_vocab_size:
        print(f"Expanding embeddings {orig_vocab_size} -> {new_vocab_size}...")
        model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=64)

        pre = model.language_model.embed_tokens.weight.data[:orig_vocab_size]
        mu = torch.mean(pre, dim=0).float()
        n = pre.size(0)
        sigma = ((pre - mu).T @ (pre - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

        num_new = model.language_model.embed_tokens.weight.data[orig_vocab_size:].shape[0]
        model.language_model.embed_tokens.weight.data[orig_vocab_size:] = torch.stack(
            [dist.sample() for _ in range(num_new)], dim=0
        )
        num_new_head = model.lm_head.weight.data[orig_vocab_size:].shape[0]
        model.lm_head.weight.data[orig_vocab_size:] = torch.stack([dist.sample() for _ in range(num_new_head)], dim=0)

    model.eval()

    # ----- save -----
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    _save_processor(args, tokenizer)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nDone! Total params: {total / 1e9:.2f}B")
    print(f"  Vision tower:    {sum(p.numel() for p in model.vision_tower.parameters()) / 1e6:.0f}M")
    print(f"  Audio tower:     {sum(p.numel() for p in model.audio_tower.parameters()) / 1e6:.0f}M")
    print(f"  Language model:  {sum(p.numel() for p in model.language_model.parameters()) / 1e6:.0f}M")
    print(f"  Audio projector: {sum(p.numel() for p in model.multi_modal_projector.parameters()) / 1e6:.1f}M")

    del model, vision_text_model
    torch.cuda.empty_cache()
    gc.collect()


def _save_processor(args, tokenizer):
    vision_processor = AutoProcessor.from_pretrained(args.vision_model_id)
    feature_extractor = VoxtralRealtimeFeatureExtractor()
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
    print(f"Saving processor to {args.output_dir}...")
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AeroRealtime initial weights")
    parser.add_argument(
        "--backbone_family",
        type=str,
        required=True,
        choices=["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"],
    )
    parser.add_argument("--vision_model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--qwen2_audio_model_id", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--qwen3_omni_model_id", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument(
        "--audio_padding_mode",
        type=str,
        default="symmetric",
        choices=["causal", "symmetric"],
        help="Conv padding mode for audio embedder.",
    )
    parser.add_argument("--output_dir", type=str, default="./data/aero_realtime_init")
    parser.add_argument("--downsample_factor", type=int, default=4)
    parser.add_argument("--skip_weights", action="store_true")
    args = parser.parse_args()

    main(args)
