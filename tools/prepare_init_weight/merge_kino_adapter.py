import argparse
import glob
import os

import requests
from accelerate import init_empty_weights
from PIL import Image
from safetensors import safe_open

from lmms_engine.models.qwen2_5_vl_audio import (
    KinoQwen2_5_VLConfig,
    KinoQwen2_5_VLForConditionalGeneration,
    KinoQwen2_5_VLProcessor,
)


def load_state_dict(directory_path):
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


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def merge_adapters(
    vision_adapter_path,
    audio_adapter_path,
    text_adapter_path,
    output_path,
    processor_path,
):
    vision_adapter_state_dict = load_state_dict(vision_adapter_path)
    audio_adapter_state_dict = load_state_dict(audio_adapter_path)
    text_adapter_state_dict = load_state_dict(text_adapter_path)

    vision_config = KinoQwen2_5_VLConfig.from_pretrained(vision_adapter_path)
    audio_config = KinoQwen2_5_VLConfig.from_pretrained(audio_adapter_path)
    text_config = KinoQwen2_5_VLConfig.from_pretrained(text_adapter_path)

    # Let's use vision config as merged config
    vision_config.audio_lora = audio_config.audio_lora
    vision_config.text_lora = text_config.text_lora

    with init_empty_weights():
        model = KinoQwen2_5_VLForConditionalGeneration(vision_config)

    model.load_state_dict(vision_adapter_state_dict, strict=False, assign=True)
    model.load_state_dict(audio_adapter_state_dict, strict=False, assign=True)
    model.load_state_dict(text_adapter_state_dict, strict=False, assign=True)

    model.save_pretrained(output_path)

    model = model.to("cuda")
    device = model.device

    image = load_image()
    processor = KinoQwen2_5_VLProcessor.from_pretrained(processor_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "content": "what is in this image?"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
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
    processor.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-adapter-path", "-v", type=str, required=True)
    parser.add_argument("--audio-adapter-path", "-a", type=str, required=True)
    parser.add_argument("--text-adapter-path", "-t", type=str, required=True)
    parser.add_argument("--output-path", "-o", type=str, required=True)
    parser.add_argument("--processor-path", "-p", type=str, required=True)
    args = parser.parse_args()

    merge_adapters(
        args.vision_adapter_path,
        args.audio_adapter_path,
        args.text_adapter_path,
        args.output_path,
        args.processor_path,
    )
