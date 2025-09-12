rm /mnt/aigc/users/pufanyi/workspace/lmms-engine-mini/output/output_test.txt
srun \
	-p vigen \
	-j pufanyi-bagel-test \
	--framework pytorch \
	-o /mnt/aigc/users/pufanyi/workspace/lmms-engine-mini/output/output_test.txt \
	--workspace-id aigc \
	-r N6lS.Iq.I10.4 \
    --priority=HIGHEST \
    /usr/bin/zsh -c 'cd /mnt/aigc/users/pufanyi/workspace/lmms-engine-mini && /mnt/aigc/users/pufanyi/workspace/lmms-engine-mini/.venv/bin/torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12357 -m lmms_engine.launch.cli --config /mnt/aigc/users/pufanyi/workspace/lmms-engine-mini/examples/bagel/bagel_example.yaml'
