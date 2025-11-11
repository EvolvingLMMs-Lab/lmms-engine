import os
import sys

NUM_DEVICES = 8 
USED_DEVICES = set()

port = 18084

def pre_fork(server, worker):
    global USED_DEVICES
    worker.device_id = next(i for i in range(NUM_DEVICES) if i not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)
    print(f"Worker {worker.pid} assigned device {worker.device_id}", file=sys.stderr)


def post_fork(server, worker):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)
    print(f"Worker {worker.pid} active on CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", file=sys.stderr)


def child_exit(server, worker):
    global USED_DEVICES
    if hasattr(worker, 'device_id'):
        USED_DEVICES.discard(worker.device_id) 
        
bind = f"0.0.0.0:{port}" 
workers = NUM_DEVICES 
worker_class = "sync" 
timeout = 300 