source $(conda info --base)/etc/profile.d/conda.sh

export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib:$LD_LIBRARY_PATH

conda activate ocr

python -c 'from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)'

/home/ubuntu/miniconda3/envs/ocr/bin/gunicorn -c gunicorn.conf.py "app:create_app()"

# pkill gunicorn to stop