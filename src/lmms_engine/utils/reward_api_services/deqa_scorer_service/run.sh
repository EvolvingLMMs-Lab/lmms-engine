source $(conda info --base)/etc/profile.d/conda.sh


conda activate deqa

/home/ubuntu/miniconda3/envs/deqa/bin/gunicorn -c gunicorn.conf.py "app:create_app()"
