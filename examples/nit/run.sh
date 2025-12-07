NGPUS=2

# . .venv/bin/activate

# Training command
# /mnt/umm/users/pufanyi/workspace/lmms-engine/.venv/bin/torchrun --nproc_per_node=${NGPUS} \
torchrun --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12358 \
  -m lmms_engine.launch.cli \
  config_yaml=examples/nit/example_config.yaml
