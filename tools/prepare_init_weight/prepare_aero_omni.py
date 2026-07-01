# Copyright 2025 LMMs-Lab team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepare initial weights for the AeroRealtime *Omni* (thinker + talker) model.

Grafts two pretrained sources into a fresh
``AeroRealtimeOmniForConditionalGeneration``:

- thinker  <- an existing aero-realtime stage-2 checkpoint (``--thinker_ckpt``),
  whose keys are ``audio_tower.* / language_model.* / multi_modal_projector.* /
  vision_tower.*`` (no ``thinker.`` prefix; ``lm_head`` is tied, hence absent).
- talker   <- Qwen3-TTS-12Hz-0.6B (``--qwen3_tts_id``) ``talker.*`` tensors. The
  ``speaker_encoder.*`` tensors are ignored. The ``text_projection.linear_fc1``
  weights are dropped (in-dim mismatch 2048 vs 2560) and left randomly
  initialized; ``text_projection.linear_fc2`` is loaded.

Example:
    python tools/prepare_init_weight/prepare_aero_omni.py \\
        --output_dir ./data/aero_omni_init
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open

from lmms_engine.models.aero_realtime.configuration_aero_realtime import (
    AeroRealtimeConfig,
)
from lmms_engine.models.aero_realtime.processing_aero_realtime import (
    AeroRealtimeProcessor,
)
from lmms_engine.models.aero_realtime_omni.configuration_aero_realtime_omni import (
    AeroRealtimeOmniConfig,
)
from lmms_engine.models.aero_realtime_omni.configuration_aero_realtime_talker import (
    AeroRealtimeTalkerConfig,
)
from lmms_engine.models.aero_realtime_omni.modeling_aero_realtime_omni import (
    AeroRealtimeOmniForConditionalGeneration,
)

DEFAULT_THINKER_CKPT = (
    "/data/v-kaichen/azure_blob/output/"
    "aero_realtime_qwen3vl_4b_stage2_v3_from_stage1_v3_4x8_a100_40g_lr5e_5_constant"
)
DEFAULT_QWEN3_TTS_ID = "/data/v-kaichen/azure_blob/pretrained_models/huggingface/Qwen3-TTS-12Hz-0.6B-Base"


def _resolve_local_dir(model_id_or_path: str) -> Path:
    p = Path(model_id_or_path)
    if p.exists() and p.is_dir():
        return p
    print(f"  snapshot_download({model_id_or_path}) ...")
    return Path(snapshot_download(model_id_or_path, allow_patterns=["*.safetensors", "*.json"]))


def _load_subset_from_safetensors(model_dir: Path, key_filter) -> dict[str, torch.Tensor]:
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


def _build_thinker_state_dict(thinker_dir: Path) -> dict[str, torch.Tensor]:
    print("Pulling thinker weights (audio_tower / language_model / vision_tower / projector)...")

    def _keep(k: str) -> bool:
        return k.startswith(("audio_tower.", "language_model.", "multi_modal_projector.", "vision_tower."))

    sd = _load_subset_from_safetensors(thinker_dir, _keep)
    if not sd:
        raise RuntimeError(f"No thinker tensors found in {thinker_dir}")
    return sd


def _build_talker_state_dict(qwen3_tts_dir: Path) -> dict[str, torch.Tensor]:
    print("Pulling talker weights from Qwen3-TTS (talker.* only; speaker_encoder.* ignored)...")

    def _keep(k: str) -> bool:
        return k.startswith("talker.")

    raw = _load_subset_from_safetensors(qwen3_tts_dir, _keep)
    if not raw:
        raise RuntimeError(f"No talker.* tensors found in {qwen3_tts_dir}")

    dropped = []
    sd: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        tail = k[len("talker.") :]
        if tail in ("text_projection.linear_fc1.weight", "text_projection.linear_fc1.bias"):
            dropped.append(tail)
            continue
        sd[tail] = v
    print(f"  dropped (re-init) text_projection.linear_fc1: {sorted(dropped)}")
    return sd


def main(args):
    torch.set_default_dtype(torch.float16)

    thinker_dir = _resolve_local_dir(args.thinker_ckpt)
    qwen3_tts_dir = _resolve_local_dir(args.qwen3_tts_id)

    print("Building AeroRealtimeOmniConfig...")
    thinker_config = AeroRealtimeConfig.from_pretrained(thinker_dir)
    thinker_hidden = thinker_config.text_config.hidden_size
    talker_config = AeroRealtimeTalkerConfig(
        thinker_hidden_size=thinker_hidden,
    )
    config = AeroRealtimeOmniConfig(
        thinker_config=thinker_config,
        talker_config=talker_config,
        codec_loss_weight=args.codec_loss_weight,
    )

    print("Creating AeroRealtimeOmni model on meta device...")
    with init_empty_weights():
        model = AeroRealtimeOmniForConditionalGeneration(config)

    thinker_sd = _build_thinker_state_dict(thinker_dir)
    print("Materialising + loading thinker submodule...")
    model.thinker.to_empty(device="cpu")
    t_missing, t_unexpected = model.thinker.load_state_dict(thinker_sd, strict=False)
    model.thinker.tie_weights()
    if t_unexpected:
        raise RuntimeError(f"Unexpected thinker keys: {t_unexpected[:8]}...")
    bad_t_missing = [k for k in t_missing if ("rotary_emb" not in k and not k.endswith("lm_head.weight"))]
    if bad_t_missing:
        raise RuntimeError(f"Unexpected MISSING thinker keys: {bad_t_missing[:8]}...")
    print(f"  thinker loaded ({len(thinker_sd)} tensors; {len(t_missing)} missing = lm_head(tied)+rotary)")

    talker_sd = _build_talker_state_dict(qwen3_tts_dir)
    print("Materialising + loading talker submodule...")
    model.talker.to_empty(device="cpu")
    k_missing, k_unexpected = model.talker.load_state_dict(talker_sd, strict=False)
    if k_unexpected:
        raise RuntimeError(f"Unexpected talker keys (every kept tensor must map): {k_unexpected[:8]}...")
    fc1_missing = any(k.startswith("text_projection.linear_fc1") for k in k_missing)
    fc2_missing = any(k.startswith("text_projection.linear_fc2") for k in k_missing)
    if not fc1_missing:
        raise RuntimeError("Expected text_projection.linear_fc1.* to be MISSING (re-init), but it is not.")
    if fc2_missing:
        raise RuntimeError("text_projection.linear_fc2.* unexpectedly MISSING (should load from Qwen3-TTS).")
    bad_k_missing = [k for k in k_missing if not (k.startswith("text_projection.linear_fc1") or "rotary_emb" in k)]
    if bad_k_missing:
        raise RuntimeError(f"Unexpected MISSING talker keys: {bad_k_missing[:8]}...")
    print(
        f"  talker loaded ({len(talker_sd)} tensors; missing only linear_fc1 (+rotary): "
        f"{[k for k in k_missing if 'rotary_emb' not in k]})"
    )

    model.eval()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)

    print("Saving thinker processor...")
    try:
        processor = AeroRealtimeProcessor.from_pretrained(thinker_dir)
        processor.save_pretrained(args.output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  from_pretrained failed ({e}); copying processor files directly.")
        for name in (
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "chat_template.jinja",
            "preprocessor_config.json",
            "vocab.json",
            "merges.txt",
        ):
            src = thinker_dir / name
            if src.exists():
                shutil.copy2(src, output_path / name)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nDone! Total params: {total / 1e9:.2f}B")
    print(f"  Thinker: {sum(p.numel() for p in model.thinker.parameters()) / 1e9:.2f}B")
    print(f"  Talker:  {sum(p.numel() for p in model.talker.parameters()) / 1e6:.0f}M")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AeroRealtime Omni (thinker+talker) init weights")
    parser.add_argument("--thinker_ckpt", type=str, default=DEFAULT_THINKER_CKPT)
    parser.add_argument("--qwen3_tts_id", type=str, default=DEFAULT_QWEN3_TTS_ID)
    parser.add_argument("--output_dir", type=str, default="./data/aero_omni_init")
    parser.add_argument("--codec_loss_weight", type=float, default=1.0)
    args = parser.parse_args()

    main(args)
