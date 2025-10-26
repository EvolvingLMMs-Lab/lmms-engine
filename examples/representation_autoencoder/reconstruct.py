#!/usr/bin/env python3
"""
RAE Image Reconstruction Script

This script loads a pre-trained RAE-SigLIP model and reconstructs a fixed COCO image.
The reconstructed image is saved to the specified output path.

Usage:
    python reconstruct.py --model_path /path/to/model --output_path /path/to/output.png
"""

import argparse
import os
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor
from transformers.image_utils import load_image

from lmms_engine.models.rae_siglip.modeling_rae_siglip import RaeSiglipModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reconstruct an image using a pre-trained RAE-SigLIP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained model checkpoint directory",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the reconstructed image will be saved",
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA (Exponential Moving Average) weights for inference (recommended for better quality)",
    )

    return parser.parse_args()


def load_ema_weights(model, ema_state_path: str):
    """Load EMA weights into the model."""
    try:
        print(f"Loading EMA weights from: {ema_state_path}")
        ema_state = torch.load(ema_state_path, map_location="cpu")

        # Load EMA weights into model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema_state:
                    param.copy_(ema_state[name])
                else:
                    print(f"Warning: EMA state missing for parameter: {name}")

        print("EMA weights loaded successfully!")
        return model

    except Exception as e:
        print(f"Error loading EMA weights: {e}")
        print("Continuing with regular model weights...")
        return model


def load_model_and_processor(model_path: str, use_ema: bool = False):
    """Load the RAE model and processor from the checkpoint path."""
    try:
        print(f"Loading model from: {model_path}")
        model = RaeSiglipModel.from_pretrained(model_path)

        # Load EMA weights if requested
        if use_ema:
            ema_path = os.path.join(model_path, "ema_state.pt")
            if os.path.exists(ema_path):
                model = load_ema_weights(model, ema_path)
            else:
                print(f"Warning: EMA weights not found at {ema_path}")
                print("Using regular model weights instead.")

        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path)

        print("Model and processor loaded successfully!")
        return model, processor

    except Exception as e:
        print(f"Error loading model or processor: {e}")
        print(f"Make sure the model path '{model_path}' is correct and contains the required files.")
        exit(1)


def load_input_image():
    """Load the fixed COCO input image from URL."""
    try:
        print("Loading COCO image from URL...")
        image_url = "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg"
        image = load_image(image_url)
        print(f"Image loaded successfully! Size: {image.size}, Mode: {image.mode}")
        return image

    except Exception as e:
        print(f"Error loading image from URL: {e}")
        print("Make sure you have internet connection and the URL is accessible.")
        exit(1)


def reconstruct_image(model, processor, image, device):
    """Reconstruct the image using the RAE model."""
    print("Processing image with model...")

    # Set model to evaluation mode
    model.eval()

    # Prepare inputs
    inputs = processor(images=[image], return_tensors="pt")

    # Move inputs to the same device as the model
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    print("Image reconstruction completed!")
    return outputs


def save_reconstructed_image(outputs, output_path: str):
    """Save the reconstructed image to the specified path."""
    try:
        print(f"Saving reconstructed image to: {output_path}")

        # Extract the reconstructed pixels
        out_pixels = outputs.out_pixels.squeeze(0)

        # The model already outputs in [0, 1] range after denormalization
        # No need to convert from [-1, 1] - just clamp to ensure valid range
        img_tensor = out_pixels.clamp(0, 1)

        # Convert to PIL image
        img = to_pil_image(img_tensor)

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save the image
        img.save(output_path)
        print(f"Reconstructed image saved successfully to: {output_path}")

    except Exception as e:
        print(f"Error saving reconstructed image: {e}")
        exit(1)


def main():
    """Main function to run the image reconstruction pipeline."""
    # Parse command line arguments
    args = parse_args()

    # Load model and processor
    model, processor = load_model_and_processor(args.model_path, use_ema=args.use_ema)

    # Load input image
    input_image = load_input_image()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    # Reconstruct the image
    outputs = reconstruct_image(model, processor, input_image, device)

    # Save the reconstructed image
    save_reconstructed_image(outputs, args.output_path)

    print("Image reconstruction pipeline completed successfully!")


if __name__ == "__main__":
    main()
