  
  CONFIG="t2i_task_config.yaml" 
  echo "Running multi-GPU training..."
  NGPUS=8

  torchrun --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12355 \
    -m lmms_engine.launch.cli \
    config_yaml=examples/blip3o_next_rl/${CONFIG}