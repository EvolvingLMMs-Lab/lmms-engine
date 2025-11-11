source $(conda info --base)/etc/profile.d/conda.sh


conda activate geneval

/home/ubuntu/miniconda3/envs/geneval/bin/gunicorn -c gunicorn.conf.py "app:create_app()"
