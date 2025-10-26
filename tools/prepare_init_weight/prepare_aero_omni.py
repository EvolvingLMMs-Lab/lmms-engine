import argparse
import glob

import librosa
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from tqdm import tqdm
from transformers import AddedToken

from lmms_engine.models.aero import AeroProcessor
from lmms_engine.models.aero_omni import (
    AeroOmniConfig,
    AeroOmniForConditionalGeneration,
    AeroOmniProcessor,
)


def load_original_state_dict(model_id):
    directory_path = model_id

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def load_audio():
    return librosa.load(librosa.ex("libri1"), sr=16000)[0]


def main(args):
    if not args.with_out_init:
        # Load the original state dict
        original_state_dict = load_original_state_dict(args.model_id)

        # Load the config
        config = AeroOmniConfig.from_pretrained(args.model_id)
        processor = AeroProcessor.from_pretrained(args.model_id)
        tokenizer = processor.tokenizer

        # Create the model
        with init_empty_weights():
            model = AeroOmniForConditionalGeneration(config)
        # Load the state dict into the model
        model.load_state_dict(original_state_dict, assign=True)

        tokenizer.add_tokens(
            AddedToken("<|audio_pad|>", special=True, normalized=False),
            special_tokens=True,
        )

        # Hardcode snac code book size
        codebook_size = 4096
        num_codebooks = 7

        pbar = tqdm(total=codebook_size * num_codebooks, desc="Adding audio tokens")
        for i in range(codebook_size * num_codebooks):
            tokenizer.add_tokens(
                AddedToken(f"<|audio_token_{i}|>", special=True, normalized=False),
                special_tokens=True,
            )
            pbar.update(1)
        pbar.close()
        tokenizer.add_tokens(
            AddedToken("<|audio_bos|>", special=True, normalized=False),
            special_tokens=True,
        )
        tokenizer.add_tokens(
            AddedToken("<|audio_eos|>", special=True, normalized=False),
            special_tokens=True,
        )

        audio_pad_token_idx = processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        audio_token_start_from = processor.tokenizer.convert_tokens_to_ids("<|audio_token_0|>")
        audio_bos_token_idx = processor.tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        audio_eos_token_idx = processor.tokenizer.convert_tokens_to_ids("<|audio_eos|>")

        config.audio_pad_token_index = audio_pad_token_idx
        config.audio_token_start_from = audio_token_start_from
        config.audio_bos_token_index = audio_bos_token_idx
        config.audio_eos_token_index = audio_eos_token_idx
        new_processor = AeroOmniProcessor(
            tokenizer=tokenizer,
            audio_processor=processor.audio_processor,
            audio_pad_token="<|audio_pad|>",
            audio_special_token_prefix="<|audio_token_",
        )

        # Expand the model's embeddings
        print(f"Expanding embeddings")
        pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
        mu = torch.mean(pre_expansion_embeddings, dim=0).float()
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

        # We add an image token so we resize the model
        # Pad to 64 for performance reasons
        # Qwen-based models have extra unused space in the vocab size already, so no need to resize
        pad_shape = 64
        vocab_size = config.text_config.vocab_size
        # Adding one pad
        num_tokens = vocab_size + 1 + codebook_size * num_codebooks
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

        # Save the model
        model.save_pretrained(args.output_dir)
        new_processor.save_pretrained(args.output_dir)

    model = AeroOmniForConditionalGeneration.from_pretrained(args.output_dir, device_map="cuda", torch_dtype="auto")
    processor = AeroOmniProcessor.from_pretrained(args.output_dir)

    audio = load_audio()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": "placeholder"},
                {"type": "text", "text": "Please transcribe this audio into text"},
            ],
        },
    ]

    text = processor.apply_chat_template(messages)
    audio_pad_id = processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    inputs = processor(text=text, audios=[audio], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    audio_input_ids = inputs["input_ids"].clone()
    audio_input_ids.fill_(audio_pad_id)
    inputs["audio_input_ids"] = audio_input_ids

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
        )

    # Decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        "-i",
        type=str,
        default="path/to/original/model",
        help="Path to the original model directory.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="path/to/output/model",
        help="Path to save the converted model.",
    )
    parser.add_argument(
        "--with_out_init",
        default=False,
        action="store_true",
        help="Init the model or not",
    )
    args = parser.parse_args()

    main(args)
