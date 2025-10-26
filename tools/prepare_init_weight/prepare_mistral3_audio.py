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
    Mistral3ForConditionalGeneration,
    Qwen2AudioForConditionalGeneration,
)

from lmms_engine.models.mistral3_audio import (
    Mistral3AudioConfig,
    Mistral3AudioForConditionalGeneration,
    Mistral3AudioProcessor,
)
from lmms_engine.models.mistral3_audio.modeling_mistral3_audio import (
    Mistral3AudioProjector,
)


def load_pretrained_vl_model(repo_id):
    model = Mistral3ForConditionalGeneration.from_pretrained(repo_id, torch_dtype="auto", device_map="cuda:0")
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
    vl_repo_id,
    qwen2_5_audio_repo_id,
    pytorch_dump_folder_path,
    with_out_init=False,
):
    if not with_out_init:
        mistral3_model = load_pretrained_vl_model(vl_repo_id)
        qwen2_5_audio_model = load_pretrained_audio_model(qwen2_5_audio_repo_id)
        mistral3_processor = load_processor(vl_repo_id)
        qwen2_5_audio_processor = load_processor(qwen2_5_audio_repo_id)

        tokenizer = mistral3_processor.tokenizer
        tokenizer.add_tokens(AddedToken("[AUDIO]", special=True, normalized=False), special_tokens=True)
        tokenizer.add_tokens(AddedToken("[VIDEO]", special=True, normalized=False), special_tokens=True)

        audio_token_id = tokenizer.convert_tokens_to_ids("[AUDIO]")
        video_token_id = tokenizer.convert_tokens_to_ids("[VIDEO]")
        config = Mistral3AudioConfig(
            audio_token_index=audio_token_id,
            video_token_index=video_token_id,
            audio_config=qwen2_5_audio_model.config.audio_config,
            **mistral3_model.config.to_dict(),
        )

        processor = Mistral3AudioProcessor(
            image_processor=mistral3_processor.image_processor,
            audio_processor=qwen2_5_audio_processor.feature_extractor,
            tokenizer=tokenizer,
            patch_size=mistral3_processor.patch_size,
            spatial_merge_size=mistral3_processor.spatial_merge_size,
        )

        with init_empty_weights():
            model = Mistral3AudioForConditionalGeneration(config)

        model.load_state_dict(mistral3_model.state_dict(), assign=True, strict=False)
        model.audio_tower = qwen2_5_audio_model.audio_tower

        audio_modal_projector = Mistral3AudioProjector(config)
        std = config.text_config.initializer_range

        model.audio_modal_projector = audio_modal_projector

        print("Expanding the token embeddings")
        pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
        mu = torch.mean(pre_expansion_embeddings, dim=0).float()
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

        # We add an audio token and a video token so we resize the model
        # Pad to 64 for performance reasons
        pad_shape = 64
        vocab_size = config.text_config.vocab_size
        num_tokens = vocab_size + 2
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

        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
        del model, processor, mistral3_model, qwen2_5_audio_model
        torch.cuda.empty_cache()

    processor = Mistral3AudioProcessor.from_pretrained(pytorch_dump_folder_path)
    model = Mistral3AudioForConditionalGeneration.from_pretrained(
        pytorch_dump_folder_path, torch_dtype="auto", device_map="cuda:0"
    )

    device = model.device

    # prepare inputs
    image = load_image()
    prompt = "What is shown in this image"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image_url": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages)
    print(prompt)
    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True)
    print([k for k in inputs.keys()])

    # verify generation
    output_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device, dtype=model.dtype),
        image_sizes=inputs["image_sizes"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_new_tokens=100,
        use_cache=True,
    )

    generated_text = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=False)[
        0
    ].strip()

    print("Generated text:", repr(generated_text))

    print("\n\n Test Chat template \n\n")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image_url": image},
                {"type": "audio", "audio_url": image},
                {"type": "video", "video_url": image},
                {
                    "type": "text",
                    "text": "What is shown in this image, audio, and video?",
                },
            ],
        }
    ]

    print(processor.apply_chat_template(messages))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vl_repo_id",
        "-v",
        help="Hub location of the model to convert",
        default="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
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
        args.vl_repo_id,
        args.qwen2_5_audio_repo_id,
        args.pytorch_dump_folder_path,
        args.with_out_init,
    )
