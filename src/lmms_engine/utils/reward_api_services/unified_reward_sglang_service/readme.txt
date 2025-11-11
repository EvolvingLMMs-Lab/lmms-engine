conda create -n sglang_ur python=3.10
conda activate sglang_ur

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install "sglang[all]>=0.4.7"
pip install gunicorn flask