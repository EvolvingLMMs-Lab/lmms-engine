import argparse
from copy import deepcopy

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.image_utils import load_image

from lmms_engine.models.rae_siglip.configuration_rae_siglip import RaeSiglipConfig
from lmms_engine.models.rae_siglip.modeling_rae_siglip import RaeSiglipModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare RAE SigLIP model weights aligned with the original Stage-1 architecture."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="google/siglip2-base-patch16-256",
        help="HuggingFace model checkpoint to use as base",
    )
    parser.add_argument(
        "--decoder_config",
        type=str,
        default=None,
        help="Configuration name or path for the MAE decoder (defaults to local ViTXL config).",
    )
    parser.add_argument(
        "--noise_tau",
        type=float,
        default=0.8,
        help="Latent noise strength recorded in the saved configuration (train-time default 0.8, inference 0.0).",
    )
    parser.add_argument(
        "--decoder_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained MAE decoder weights (defaults to local ViTXL model).",
    )
    parser.add_argument(
        "--latent_stats",
        type=str,
        default=None,
        help="Optional path to latent normalization statistics (expects keys 'mean' and 'var').",
    )
    parser.add_argument(
        "--encoder_input_size",
        type=int,
        default=None,
        help="Override SigLIP encoder input resolution (defaults to checkpoint resolution).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Epsilon used when applying latent normalization.",
    )
    parser.add_argument(
        "--no_reshape_to_2d",
        action="store_true",
        help="Disable reshaping latents to BCHW before normalization/decoding.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/rae_siglip2",
        help="Output path for saved model (default: ./data/rae_siglip2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for model initialization",
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default="https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg",
        help="Test image URL to verify model works",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set output path
    output_path = args.output_path

    print(f"Loading base model from: {args.checkpoint}")
    print(f"Output path: {output_path}")

    # Load base model
    base_model = AutoModel.from_pretrained(args.checkpoint, device_map=args.device).eval()
    siglip_config = base_model.config.vision_config
    encoder_input_size = (
        int(args.encoder_input_size) if args.encoder_input_size is not None else int(siglip_config.image_size)
    )

    if args.decoder_config is not None:
        # Use custom decoder config from HuggingFace or path
        decoder_config = AutoConfig.from_pretrained(args.decoder_config)
        decoder_config.hidden_size = siglip_config.hidden_size
    else:
        # Use local ViTXL decoder architecture (matches the default checkpoint)
        decoder_config = AutoConfig.from_pretrained("facebook/vit-mae-base")
        decoder_config.decoder_hidden_size = 1152
        decoder_config.decoder_num_hidden_layers = 28
        decoder_config.decoder_num_attention_heads = 16  # Fixed: align with original RAE ViTXL config
        decoder_config.decoder_intermediate_size = 4096
        decoder_config.hidden_size = siglip_config.hidden_size

    latent_mean = latent_var = None
    if args.latent_stats:
        stats = torch.load(args.latent_stats, map_location="cpu")
        latent_mean = stats.get("mean")
        latent_var = stats.get("var")
        if latent_mean is not None:
            latent_mean = latent_mean.to(dtype=torch.float32).cpu()
        if latent_var is not None:
            latent_var = latent_var.to(dtype=torch.float32).cpu()

    rae_config = RaeSiglipConfig(
        encoder_config=siglip_config,
        decoder_config=decoder_config,
        encoder_processor_path=args.checkpoint,
        noise_tau=args.noise_tau,
        encoder_input_size=encoder_input_size,
        reshape_to_2d=not args.no_reshape_to_2d,
        latent_mean=latent_mean.tolist() if latent_mean is not None else None,
        latent_var=latent_var.tolist() if latent_var is not None else None,
        eps=args.eps,
    )

    # Initialize RAE model
    print("Initializing RAE model...")
    rae_model = RaeSiglipModel(rae_config)

    # Load pretrained encoder weights
    print("Loading pretrained encoder weights...")
    encoder_state = deepcopy(base_model.vision_model.state_dict())
    encoder_state.pop("post_layernorm.weight", None)
    encoder_state.pop("post_layernorm.bias", None)
    missing, unexpected = rae_model.encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"Missing encoder keys: {missing}")
    if unexpected:
        print(f"Unexpected encoder keys: {unexpected}")

    if args.decoder_checkpoint is not None:
        print(f"Loading pretrained decoder weights from: {args.decoder_checkpoint}")
        decoder_state = torch.load(args.decoder_checkpoint, map_location="cpu")
        decoder_missing, decoder_unexpected = rae_model.decoder.load_state_dict(decoder_state, strict=False)
        if decoder_missing:
            print(f"Missing decoder keys: {decoder_missing}")
        if decoder_unexpected:
            print(f"Unexpected decoder keys: {decoder_unexpected}")

    # Move model to device
    rae_model.to(base_model.device).eval()

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.checkpoint)

    # Test model with image
    print(f"Testing model with image: {args.test_image}")
    test_image = load_image(args.test_image)
    inputs = processor(images=[test_image], return_tensors="pt").to(base_model.device)

    with torch.no_grad():
        outputs = rae_model(**inputs)
        print(f"Initial test successful! Output: {outputs}")

    # Save model, processor, and config
    print(f"\nSaving to {output_path}...")
    rae_model.save_pretrained(output_path, safe_serialization=True)
    processor.save_pretrained(output_path)
    rae_config.save_pretrained(output_path)
    print("Model saved successfully!")

    # Verify saved model works
    print("\nVerifying saved model...")
    loaded_model = RaeSiglipModel.from_pretrained(output_path)
    loaded_config = RaeSiglipConfig.from_pretrained(output_path)

    loaded_model.to(base_model.device).eval()

    with torch.no_grad():
        outputs = loaded_model(**inputs)
        output_shape = outputs.out_pixels.shape if hasattr(outputs, "out_pixels") else "No out_pixels attribute"
        print(f"Verification successful! Output shape: {output_shape}")

    print(f"\n✓ RAE SigLIP model saved to: {output_path}")
    print(
        f"✓ Decoder image size: {loaded_config.decoder_config.image_size}, "
        f"patch size: {loaded_config.decoder_config.patch_size}"
    )


if __name__ == "__main__":
    main()
