#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 WanVideo team. All rights reserved.
"""
Test script to verify WanVideo model loading and basic functionality.
"""

import sys
from pathlib import Path
import torch
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lmms_engine.models.wanvideo import (
    WanVideoConfig,
    WanVideoForConditionalGeneration,
    WanVideoProcessor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config_creation():
    """Test WanVideoConfig creation."""
    logger.info("Testing WanVideoConfig creation...")
    
    # Test default config
    config = WanVideoConfig()
    assert config.model_type == "wanvideo"
    assert config.dit_hidden_size == 3072
    logger.info("✓ Default config created successfully")
    
    # Test 1.3B config
    config_1_3b = WanVideoConfig(
        model_size="1.3B",
        dit_hidden_size=2432,
        dit_num_layers=28,
        dit_num_heads=19,
        dit_enable_flash_attn=False,  # Disable for CPU testing
    )
    assert config_1_3b.model_size == "1.3B"
    assert config_1_3b.dit_hidden_size == 2432
    logger.info("✓ 1.3B config created successfully")
    
    # Test 14B config
    config_14b = WanVideoConfig(
        model_size="14B",
        dit_hidden_size=5120,
        dit_num_layers=48,
        dit_num_heads=40,
    )
    assert config_14b.model_size == "14B"
    assert config_14b.dit_hidden_size == 5120
    logger.info("✓ 14B config created successfully")
    
    return config_1_3b


def test_model_creation(config):
    """Test WanVideoForConditionalGeneration model creation."""
    logger.info("Testing model creation...")
    
    # Create model
    model = WanVideoForConditionalGeneration(config)
    assert model is not None
    assert model.config.model_type == "wanvideo"
    logger.info("✓ Model created successfully")
    
    # Check model components
    assert model.dit is not None
    assert len(model.dit.blocks) == config.dit_num_layers
    logger.info(f"✓ DiT model has {len(model.dit.blocks)} blocks")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def test_processor_creation():
    """Test WanVideoProcessor creation."""
    logger.info("Testing processor creation...")
    
    # Create processor
    processor = WanVideoProcessor()
    assert processor is not None
    assert processor.image_processor is not None
    logger.info("✓ Processor created successfully")
    
    # Test image processing
    import numpy as np
    dummy_image = np.random.randint(0, 255, (480, 832, 3), dtype=np.uint8)
    processed = processor.image_processor.preprocess(
        dummy_image,
        return_tensors="pt"
    )
    assert "pixel_values" in processed
    assert processed["pixel_values"].shape[0] == 1  # Batch size
    logger.info("✓ Image processing works")
    
    return processor


def test_forward_pass(model, config):
    """Test model forward pass."""
    logger.info("Testing forward pass...")
    
    # Create dummy inputs
    batch_size = 1
    num_frames = 8  # Small for testing
    height = 32  # Small for testing
    width = 32  # Small for testing
    
    # Create dummy latents
    latents = torch.randn(
        batch_size,
        config.dit_in_channels,
        num_frames // 4,
        height // 8,
        width // 8,
    )
    
    # Create dummy timesteps
    timesteps = torch.tensor([500])  # Middle of diffusion process
    
    # Forward pass
    with torch.no_grad():
        output = model(
            latents=latents,
            timesteps=timesteps,
            return_dict=True,
        )
    
    assert output is not None
    assert output.noise_pred is not None
    assert output.noise_pred.shape == latents.shape
    logger.info("✓ Forward pass successful")
    
    return output


def test_training_step(model, config):
    """Test training step with loss computation."""
    logger.info("Testing training step...")
    
    # Create dummy inputs
    batch_size = 1
    num_frames = 8
    height = 32
    width = 32
    
    # Create dummy latents and noise
    latents = torch.randn(
        batch_size,
        config.dit_in_channels,
        num_frames // 4,
        height // 8,
        width // 8,
    )
    noise = torch.randn_like(latents)
    
    # Create dummy text prompt
    prompt = ["A test video"]
    
    # Training step
    model.train()
    output = model(
        latents=latents,
        noise=noise,
        prompt=prompt,
        return_dict=True,
    )
    
    assert output.loss is not None
    assert output.loss.requires_grad
    logger.info(f"✓ Training loss computed: {output.loss.item():.4f}")
    
    # Test backward pass
    output.loss.backward()
    logger.info("✓ Backward pass successful")
    
    return output


def test_generation(model, config):
    """Test video generation."""
    logger.info("Testing video generation...")
    
    # Set to eval mode
    model.eval()
    
    # Generate video
    with torch.no_grad():
        video = model.generate(
            prompt="A beautiful sunset over the ocean",
            num_frames=8,  # Small for testing
            height=32,  # Small for testing
            width=32,  # Small for testing
            num_inference_steps=5,  # Few steps for testing
            guidance_scale=5.0,
        )
    
    assert video is not None
    assert video.dim() == 5  # (B, T, C, H, W)
    logger.info(f"✓ Generated video shape: {video.shape}")
    
    return video


def main():
    """Run all tests."""
    logger.info("="*50)
    logger.info("Starting WanVideo model tests...")
    logger.info("="*50)
    
    try:
        # Test configuration
        config = test_config_creation()
        logger.info("")
        
        # Test model creation
        model = test_model_creation(config)
        logger.info("")
        
        # Test processor
        processor = test_processor_creation()
        logger.info("")
        
        # Test forward pass
        output = test_forward_pass(model, config)
        logger.info("")
        
        # Test training step
        training_output = test_training_step(model, config)
        logger.info("")
        
        # Test generation
        video = test_generation(model, config)
        logger.info("")
        
        logger.info("="*50)
        logger.info("✅ All tests passed successfully!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
