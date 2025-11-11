source $(conda info --base)/etc/profile.d/conda.sh


conda activate pickscore

/home/ubuntu/miniconda3/envs/pickscore/bin/gunicorn -c gunicorn.conf.py "app:create_app()"

# pkill gunicorn to stop