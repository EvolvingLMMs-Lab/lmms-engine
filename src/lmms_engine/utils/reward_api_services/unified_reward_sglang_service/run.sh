# source $(conda info --base)/etc/profile.d/conda.sh


# conda activate sglang_ur

# python -m sglang.launch_server --model-path ./UnifiedReward-7b-v1.5 --api-key reward_api --port 17140 --chat-template chatml-llava --enable-p2p-check --mem-fraction-static 0.85 

# /home/ubuntu/miniconda3/envs/sglang_ur/bin/gunicorn -c gunicorn.conf.py "app:create_app()"

#!/bin/bash

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source $(conda info --base)/etc/profile.d/conda.sh
conda activate sglang_ur

# head num should be divisible by tp-size
nohup python -m sglang.launch_server --model-path "CodeGoat24/UnifiedReward-7b-v1.5" --api-key reward_api --port 17140 --chat-template chatml-llava --enable-p2p-check --mem-fraction-static 0.85 --tp-size 4 > sglang.log 2>&1 &

echo "Waiting for sglang server..."
for i in {1..30}; do
    if curl -s http://localhost:17140 > /dev/null; then
        echo "Starting gunicorn..."
        /home/ubuntu/miniconda3/envs/sglang_ur/bin/gunicorn -c gunicorn.conf.py "app:create_app()"
        exit 0
    fi
    sleep 2
done

echo "Error: sglang server failed to start."
kill $!
exit 1