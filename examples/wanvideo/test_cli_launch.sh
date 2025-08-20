#!/bin/bash
# Test script to verify CLI launch works with WanVideo

set -e

# Set environment variables for single GPU test
export CUDA_VISIBLE_DEVICES=""
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
export TOKENIZERS_PARALLELISM="false"

echo "Testing CLI launch with WanVideo model..."
echo "=========================================="

# Create a minimal test config
cat > /tmp/test_wanvideo_config.yaml << 'EOF'
- task_type: trainer
  config:
    trainer_type: hf_trainer
    
    dataset_config:
      dataset_type: vision
      dataset_format: json
      dataset_path: /tmp/dummy_dataset.json
      video_sampling_strategy: frame_num
      frame_num: 8
      video_backend: qwen_vl_utils
      processor_config:
        processor_type: wanvideo
        processor_name: wanvideo
    
    model_config:
      load_from_config:
        model_type: wanvideo
        model_size: "1.3B"
        dit_hidden_size: 512  # Very small for testing
        dit_num_layers: 2  # Very small for testing
        dit_num_heads: 8
        dit_enable_flash_attn: false
        gradient_checkpointing: false
        num_frames: 8
        height: 32
        width: 32
      attn_implementation: sdpa
    
    output_dir: "/tmp/test_wanvideo_output"
    max_steps: 1
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 1
    learning_rate: 1.0e-5
    logging_steps: 1
    save_steps: 1000
    dataloader_num_workers: 0
    bf16: false
    fp16: false
    seed: 42
    max_grad_norm: 1.0
    use_cpu: true
EOF

# Create a dummy dataset
cat > /tmp/dummy_dataset.json << 'EOF'
[
  {
    "video_path": "dummy_video.mp4",
    "caption": "A test video",
    "video_id": "test_001"
  }
]
EOF

echo "Created test configuration and dataset"

# Run the CLI (with max_steps=1 it will exit quickly)
echo "Launching training with CLI..."
python -m lmms_engine.launch.cli --config /tmp/test_wanvideo_config.yaml 2>&1 | head -100

echo ""
echo "✅ CLI launch test completed successfully!"
echo "You can now use the full configuration files for actual training."
