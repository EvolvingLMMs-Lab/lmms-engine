source $(conda info --base)/etc/profile.d/conda.sh


conda activate imagereward

/home/ubuntu/miniconda3/envs/imagereward/bin/gunicorn -c gunicorn.conf.py "app:create_app()"

# pkill gunicorn to stop