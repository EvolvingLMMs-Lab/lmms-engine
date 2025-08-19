# Adapted from https://github.com/JiuhaiChen/BLIP3o/blob/BLIP3o-NEXT/blip3o/utils.py

import torch.distributed as dist

from lmms_engine.utils import Logging


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            Logging.info(f"Rank {dist.get_rank()}: ", *args)
    else:
        Logging.info(*args)


def rank_print(*args):
    if dist.is_initialized():
        Logging.info(f"Rank {dist.get_rank()}: ", *args)
    else:
        Logging.info(*args)
