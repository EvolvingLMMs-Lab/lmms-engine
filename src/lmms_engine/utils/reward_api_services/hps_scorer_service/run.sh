source $(conda info --base)/etc/profile.d/conda.sh


conda activate hps

/home/ubuntu/miniconda3/envs/hps/bin/gunicorn -c gunicorn.conf.py "app:create_app()"

# pkill gunicorn to stop