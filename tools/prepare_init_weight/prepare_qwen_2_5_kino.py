import argparse
from pathlib import Path

import requests
import torch
from accelerate import init_empty_weights
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
from lmms_engine.models.qwen2_5_vl_audio.modeling_qwen2_5_vl import (
    AudioMultiModalProjector,
)


def load_pretrained_vl_model(repo_id):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo_id, torch_dtype="auto", device_map="cuda:0")
    return model


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def load_pretrained_audio_model(repo_id):
    model = Qwen2AudioForConditionalGeneration.from_pretrained(repo_id, torch_dtype="auto", device_map="cuda:1")
    return model


def load_processor(repo_id):
    processor = AutoProcessor.from_pretrained(repo_id)
    return processor


def prepare_weights_for_kino(
    qwen2_5_vl_repo_id,
    qwen2_5_audio_repo_id,
    pytorch_dump_folder_path,
    with_out_init=False,
):
    if not with_out_init:
        qwen2_5_vl_model = load_pretrained_vl_model(qwen2_5_vl_repo_id)
        qwen2_5_audio_model = load_pretrained_audio_model(qwen2_5_audio_repo_id)
        qwen2_5_processor = load_processor(qwen2_5_vl_repo_id)
        qwen2_5_audio_processor = load_processor(qwen2_5_audio_repo_id)

        tokenizer = qwen2_5_processor.tokenizer
        tokenizer.add_tokens(AddedToken("<|AUDIO|>", special=True, normalized=False), special_tokens=True)

        audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        config = KinoQwen2_5_VLConfig(
            audio_token_id=audio_token_id,
            audio_config=qwen2_5_audio_model.config.audio_config,
            **qwen2_5_vl_model.config.to_dict(),
        )

        processor = KinoQwen2_5_VLProcessor(
            image_processor=qwen2_5_processor.image_processor,
            audio_processor=qwen2_5_audio_processor.feature_extractor,
            tokenizer=tokenizer,
        )

        with init_empty_weights():
            model = KinoQwen2_5_VLForConditionalGeneration(config)

        model.load_state_dict(qwen2_5_vl_model.state_dict(), assign=True, strict=False)
        model.audio_tower = qwen2_5_audio_model.audio_tower

        audio_modal_projector = AudioMultiModalProjector(config)
        std = config.initializer_range

        audio_modal_projector.linear.weight.data.normal_(mean=0.0, std=std)
        audio_modal_projector.linear.bias.data.zero_()
        model.audio_modal_projector = audio_modal_projector

        pre_expansion_embeddings = model.model.embed_tokens.weight.data
        mu = torch.mean(pre_expansion_embeddings, dim=0).float()
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

        # We add an audio token so we resize the model
        # Pad to 64 for performance reasons
        pad_shape = 64
        vocab_size = config.vocab_size
        num_tokens = vocab_size + 1
        model.resize_token_embeddings(num_tokens, pad_to_multiple_of=pad_shape)
        model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
            tuple((dist.sample() for _ in range(model.model.embed_tokens.weight.data[vocab_size:].shape[0]))),
            dim=0,
        )
        model.lm_head.weight.data[vocab_size:] = torch.stack(
            tuple((dist.sample() for _ in range(model.lm_head.weight.data[vocab_size:].shape[0]))),
            dim=0,
        )

        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    model = KinoQwen2_5_VLForConditionalGeneration.from_pretrained(
        pytorch_dump_folder_path, torch_dtype="auto", device_map="cuda:0"
    )
    processor = KinoQwen2_5_VLProcessor.from_pretrained(pytorch_dump_folder_path)

    device = model.device

    # prepare inputs
    image = load_image()
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
        "--qwen2_5_vl_repo_id",
        "-v",
        help="Hub location of the model to convert",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--qwen2_5_audio_repo_id",
        "-a",
        help="Hub location of the model to convert",
        default="Qwen/Qwen2-Audio-7B-Instruct",
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
        args.qwen2_5_vl_repo_id,
        args.qwen2_5_audio_repo_id,
        args.pytorch_dump_folder_path,
        args.with_out_init,
    )
