source $(conda info --base)/etc/profile.d/conda.sh


conda activate sd3

/home/ubuntu/miniconda3/envs/sd3/bin/gunicorn -c gunicorn.conf.py "app:create_app()"
