#!/usr/bin/env python
# Test script to verify CLI integration with WanVideo model

import os
import sys
import tempfile
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from lmms_engine.utils.config_loader import load_config
from lmms_engine.models import ModelConfig
from lmms_engine.datasets import DatasetConfig
from lmms_engine.train import TrainerConfig, TrainingArguments
from lmms_engine.mapping_func import create_model_from_config

def test_config_loading():
    """Test loading WanVideo configuration."""
    print("Testing config loading...")
    config_path = "examples/wanvideo/configs/wan2.1_t2v_1.3b.yaml"
    configs = load_config(config_path)
    assert len(configs) > 0
    assert configs[0].get("task_type") == "trainer"
    print("✓ Config loaded successfully")
    return configs[0]["config"]

def test_model_creation(config):
    """Test model creation from config."""
    print("\nTesting model creation from config...")
    model_config = config["model_config"]
    model_type = model_config["load_from_config"]["model_type"]
    init_config = model_config["load_from_config"]
    
    # Test the mapping function
    model_class, m_config = create_model_from_config(model_type, init_config)
    assert model_class is not None
    assert m_config is not None
    print(f"✓ Model class: {model_class.__name__}")
    print(f"✓ Model config type: {type(m_config).__name__}")
    
    # Create model instance
    model = model_class(m_config)
    assert model is not None
    print(f"✓ Model created: {type(model).__name__}")
    
    # Check model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    return model

def test_dataset_config(config):
    """Test dataset configuration."""
    print("\nTesting dataset configuration...")
    dataset_config = config["dataset_config"]
    
    # Create DatasetConfig object
    ds_config = DatasetConfig(**dataset_config)
    assert ds_config.dataset_type == "vision"
    assert ds_config.processor_config["processor_type"] == "wanvideo"
    print(f"✓ Dataset type: {ds_config.dataset_type}")
    print(f"✓ Processor type: {ds_config.processor_config['processor_type']}")
    print(f"✓ Video backend: {ds_config.video_backend}")
    
    return ds_config

def test_trainer_config(config):
    """Test trainer configuration."""
    print("\nTesting trainer configuration...")
    
    dataset_config = DatasetConfig(**config["dataset_config"])
    model_config = ModelConfig(**config["model_config"])
    trainer_type = config["trainer_type"]
    
    # Extract trainer args, handling the nested structure
    trainer_args_dict = config.copy()
    trainer_args_dict.pop("dataset_config", None)
    trainer_args_dict.pop("model_config", None)
    trainer_args_dict.pop("trainer_type", None)
    
    # If trainer_args exists as a key, use it; otherwise use the remaining config
    if "trainer_args" in trainer_args_dict:
        trainer_args_dict = trainer_args_dict["trainer_args"]
    
    # Set required fields if not present
    if "output_dir" not in trainer_args_dict:
        trainer_args_dict["output_dir"] = "./test_output"
    
    trainer_args = TrainingArguments(**trainer_args_dict)
    
    trainer_config = TrainerConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        trainer_type=trainer_type,
        trainer_args=trainer_args,
    )
    
    assert trainer_config.trainer_type == "hf_trainer"
    print(f"✓ Trainer type: {trainer_config.trainer_type}")
    print(f"✓ Output dir: {trainer_config.trainer_args.output_dir}")
    print(f"✓ Learning rate: {trainer_config.trainer_args.learning_rate}")
    
    return trainer_config

def test_processor_registration():
    """Test that WanVideo processor is registered."""
    print("\nTesting processor registration...")
    from lmms_engine.mapping_func import DATAPROCESSOR_MAPPING
    
    assert "wanvideo" in DATAPROCESSOR_MAPPING
    processor_class = DATAPROCESSOR_MAPPING["wanvideo"]
    print(f"✓ WanVideo processor registered: {processor_class.__name__}")
    
    # Test processor instantiation
    from lmms_engine.datasets.processor import ProcessorConfig
    proc_config = ProcessorConfig(processor_name="wanvideo", processor_type="wanvideo")
    processor = processor_class(proc_config)
    assert processor is not None
    print("✓ Processor instantiated successfully")
    
    return processor

def main():
    """Run all integration tests."""
    print("="*60)
    print("Testing WanVideo CLI Integration")
    print("="*60)
    
    try:
        # Test config loading
        config = test_config_loading()
        
        # Test model creation
        model = test_model_creation(config)
        del model  # Free memory
        
        # Test dataset config
        dataset_config = test_dataset_config(config)
        
        # Test trainer config
        trainer_config = test_trainer_config(config)
        
        # Test processor registration
        processor = test_processor_registration()
        
        print("\n" + "="*60)
        print("✅ All CLI integration tests passed!")
        print("="*60)
        print("\nYou can now run training with:")
        print("python -m lmms_engine.launch.cli --config examples/wanvideo/configs/wan2.1_t2v_1.3b.yaml")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
