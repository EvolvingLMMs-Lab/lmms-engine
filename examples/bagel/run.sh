# . ".venv/bin/activate"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12357 -m lmms_engine.launch.cli --config examples/bagel/bagel_example.yaml
