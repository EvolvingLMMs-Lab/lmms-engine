"""Prepare standalone AeroRealtime talker weights from Qwen3-TTS.

The output contains only the Qwen3-TTS talker. AeroRealtime processor assets
are copied from ``--processor_ckpt`` so the dataset can produce aligned
``text_stream_ids``.
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

from lmms_engine.models.aero_realtime.processing_aero_realtime import (
    AeroRealtimeProcessor,
)
from lmms_engine.models.aero_realtime_talker.configuration_aero_realtime_talker import (
    AeroRealtimeTalkerConfig,
)
from lmms_engine.models.aero_realtime_talker.modeling_aero_realtime_talker import (
    AeroRealtimeTalkerForConditionalGeneration,
)

DEFAULT_PROCESSOR_CKPT = (
    "/data/v-kaichen/azure_blob/output/"
    "aero_realtime_qwen3vl_4b_stage2_v3_from_stage1_v3_4x8_a100_40g_lr5e_5_constant"
)
DEFAULT_QWEN3_TTS_ID = "/data/v-kaichen/azure_blob/pretrained_models/huggingface/Qwen3-TTS-12Hz-0.6B-Base"


def _resolve_local_dir(model_id_or_path: str) -> Path:
    path = Path(model_id_or_path)
    if path.exists() and path.is_dir():
        return path
    print(f"  snapshot_download({model_id_or_path}) ...")
    return Path(snapshot_download(model_id_or_path, allow_patterns=["*.safetensors", "*.json"]))


def _load_talker_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            if key.startswith("talker."):
                shards.setdefault(shard, []).append(key)
    else:
        shards = {path.name: None for path in sorted(model_dir.glob("*.safetensors"))}

    state_dict = {}
    for shard_name, keys in shards.items():
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
            shard_keys = keys if keys is not None else [key for key in handle.keys() if key.startswith("talker.")]
            for key in shard_keys:
                state_dict[key.removeprefix("talker.")] = handle.get_tensor(key)
    if not state_dict:
        raise RuntimeError(f"No talker.* tensors found in {model_dir}")
    return state_dict


def _build_config(qwen3_tts_dir: Path) -> AeroRealtimeTalkerConfig:
    source_config = json.loads((qwen3_tts_dir / "config.json").read_text())["talker_config"]
    source_config.pop("model_type", None)
    source_config.pop("spk_id", None)
    source_config["codec_eos_id"] = source_config.pop("codec_eos_token_id")
    source_config["speaker_id"] = {"ryan": 3061}
    return AeroRealtimeTalkerConfig(**source_config)


def _save_processor(processor_dir: Path, output_dir: Path) -> None:
    print("Saving AeroRealtime processor...")
    try:
        processor = AeroRealtimeProcessor.from_pretrained(processor_dir)
        processor.save_pretrained(output_dir)
    except Exception as error:  # noqa: BLE001
        print(f"  from_pretrained failed ({error}); copying processor files directly.")
        for name in (
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "chat_template.jinja",
            "preprocessor_config.json",
            "vocab.json",
            "merges.txt",
        ):
            source = processor_dir / name
            if source.exists():
                shutil.copy2(source, output_dir / name)


def main(args) -> None:
    processor_dir = _resolve_local_dir(args.processor_ckpt)
    qwen3_tts_dir = _resolve_local_dir(args.qwen3_tts_id)

    config = _build_config(qwen3_tts_dir)
    print("Creating standalone AeroRealtime talker on meta device...")
    with init_empty_weights():
        model = AeroRealtimeTalkerForConditionalGeneration(config)

    print("Loading Qwen3-TTS talker weights...")
    state_dict = _load_talker_state_dict(qwen3_tts_dir)
    model.to_empty(device="cpu")
    model.apply(model._init_weights)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"Unexpected talker keys: {unexpected[:8]}")
    bad_missing = [key for key in missing if "rotary_emb" not in key]
    if bad_missing:
        raise RuntimeError(f"Unexpected missing talker keys: {bad_missing[:8]}")
    print(f"  loaded {len(state_dict)} tensors; missing only non-persistent rotary buffers: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(output_dir)
    _save_processor(processor_dir, output_dir)

    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Done. Talker parameters: {total / 1e6:.0f}M")
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare standalone AeroRealtime talker init weights")
    parser.add_argument("--processor_ckpt", type=str, default=DEFAULT_PROCESSOR_CKPT)
    parser.add_argument("--qwen3_tts_id", type=str, default=DEFAULT_QWEN3_TTS_ID)
    parser.add_argument("--output_dir", type=str, default="./data/aero_realtime_talker_init")
    main(parser.parse_args())
