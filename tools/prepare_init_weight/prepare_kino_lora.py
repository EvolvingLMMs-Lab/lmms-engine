import argparse
import gc
from pathlib import Path

import requests
import torch
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    AddedToken,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2AudioForConditionalGeneration,
)

from lmms_engine.models.qwen2_5_vl_audio import (
    KinoQwen2_5_VLConfig,
    KinoQwen2_5_VLForConditionalGeneration,
    KinoQwen2_5_VLProcessor,
)


def load_pretrained_weights(repo_id):
    model = KinoQwen2_5_VLForConditionalGeneration.from_pretrained(repo_id, torch_dtype="auto", device_map="cuda:0")
    return model


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def load_processor(repo_id):
    processor = KinoQwen2_5_VLProcessor.from_pretrained(repo_id)
    return processor


def prepare_weights_for_kino(
    repo_id,
    pytorch_dump_folder_path,
    with_out_init=False,
):
    if not with_out_init:
        kino_model = load_pretrained_weights(repo_id)
        processor = load_processor(repo_id)
        vision_lora = LoraConfig(
            r=256,
            target_modules="layers.*((self_attn\\.((q|k|v|o)_proj))|(mlp\\.(gate|down)_proj))",
            lora_alpha=512,
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
        )
        audio_lora = LoraConfig(
            r=256,
            target_modules="layers.*((self_attn\\.((q|k|v|o)_proj))|(mlp\\.(gate|down)_proj))",
            lora_alpha=512,
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
        )
        text_lora = LoraConfig(
            r=256,
            target_modules="layers.*((self_attn\\.((q|k|v|o)_proj))|(mlp\\.(gate|down)_proj))",
            lora_alpha=512,
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
        )

        vision_adapter = get_peft_model(kino_model.model, peft_config=vision_lora, adapter_name="vision")
        audio_adapter = get_peft_model(kino_model.model, peft_config=audio_lora, adapter_name="audio")
        text_adapter = get_peft_model(kino_model.model, peft_config=text_lora, adapter_name="text")
        config = kino_model.config
        config.vision_lora = vision_lora.to_dict()
        config.audio_lora = audio_lora.to_dict()
        config.text_lora = text_lora.to_dict()

        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        kino_model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
        config.save_pretrained(pytorch_dump_folder_path)
        del kino_model, processor, config
        gc.collect()
        torch.cuda.empty_cache()

    model = KinoQwen2_5_VLForConditionalGeneration.from_pretrained(
        pytorch_dump_folder_path, torch_dtype="auto", device_map="cuda:0"
    )
    processor = KinoQwen2_5_VLProcessor.from_pretrained(pytorch_dump_folder_path)

    device = model.device

    # prepare inputs
    image = load_image()
    # model.unset_lora_adapter()
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        "-v",
        help="Hub location of the model to convert",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
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
    prepare_weights_for_kino(
        args.repo_id,
        args.pytorch_dump_folder_path,
        args.with_out_init,
    )
