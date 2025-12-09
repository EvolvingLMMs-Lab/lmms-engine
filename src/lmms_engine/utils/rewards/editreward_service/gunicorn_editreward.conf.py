"""
Gunicorn configuration for EditReward service.

Required environment variables:
    EDITREWARD_MODEL_PATH      : Path to base model
    EDITREWARD_CHECKPOINT_PATH : Path to checkpoint

Optional environment variables:
    EDITREWARD_PORT        : Server port (default: 18087)
    EDITREWARD_NUM_WORKERS : Number of workers (default: 8)

Usage:
    export EDITREWARD_MODEL_PATH=/path/to/base_model
    export EDITREWARD_CHECKPOINT_PATH=/path/to/checkpoint
    gunicorn -c gunicorn_editreward.conf.py 'app_editreward:create_app()'
"""

import os

# App settings
wsgi_app = "app_editreward:create_app()"
bind = f"127.0.0.1:{os.getenv('EDITREWARD_PORT', '18087')}"
workers = int(os.getenv("EDITREWARD_NUM_WORKERS", "8"))
worker_class = "sync"
timeout = 300

# GPU assignment for workers
NUM_DEVICES = workers
USED_DEVICES = set()


def pre_fork(server, worker):
    global USED_DEVICES
    worker.device_id = next(i for i in range(NUM_DEVICES) if i not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)


def post_fork(server, worker):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)


def child_exit(server, worker):
    global USED_DEVICES
    USED_DEVICES.discard(worker.device_id)
