# . ".venv/bin/activate"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12357 -m lmms_engine.launch.cli --config /mnt/aigc/users/pufanyi/workspace/lmms-engine-mini/examples/bagel/bagel_example.yaml
