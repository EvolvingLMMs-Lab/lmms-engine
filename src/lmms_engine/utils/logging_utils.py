import logging as _stdlib_logging
from contextlib import redirect_stdout
from typing import Any, Dict

import torch.distributed as dist
from loguru import logger
from rich.logging import RichHandler

# Third-party loggers that emit too-verbose warnings during multimodal data
# loading (e.g. decord/torchvision fallback noise on every video). These are
# silenced at startup by ``setup_distributed_logging`` to keep training logs
# readable.
_NOISY_LOGGERS = (
    "qwen_vl_utils",
    "qwen_vl_utils.vision_process",
)


def distributed_filter(record: Dict[str, Any]) -> bool:
    """
    Filter function for distributed training.
    Only allows logs from rank 0 when distributed training is initialized.
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def setup_distributed_logging():
    """
    Setup loguru logger with distributed training filter.
    Call this function once at the beginning of your program.
    """
    # Remove default handler
    logger.remove()

    # Add handler with distributed filter and RichHandler for beautiful logging
    logger.add(
        RichHandler(rich_tracebacks=True, show_path=True, omit_repeated_times=False),
        format="{message}",
        filter=distributed_filter,
        level="DEBUG",
    )

    # Silence noisy third-party loggers (e.g. qwen_vl_utils warns on every
    # video decode fallback). These warnings flood the log without being
    # actionable.
    for name in _NOISY_LOGGERS:
        _stdlib_logging.getLogger(name).setLevel(_stdlib_logging.ERROR)


class Logging:
    """
    Legacy Logging class for backward compatibility.
    Recommend using loguru logger directly with setup_distributed_logging().
    """

    @staticmethod
    def show_deprecation_warning():
        logger.warning("Logging is deprecated. Use loguru logger directly.")

    @staticmethod
    def info(msg: str):
        Logging.show_deprecation_warning()
        if dist.is_initialized():
            if dist.get_rank() == 0:
                logger.info(msg)
        else:
            logger.info(msg)

    @staticmethod
    def error(msg: str):
        Logging.show_deprecation_warning()
        if dist.is_initialized():
            if dist.get_rank() == 0:
                logger.error(msg)
        else:
            logger.error(msg)

    @staticmethod
    def warning(msg: str):
        Logging.show_deprecation_warning()
        if dist.is_initialized():
            if dist.get_rank() == 0:
                logger.warning(msg)
        else:
            logger.warning(msg)

    @staticmethod
    def debug(msg: str):
        Logging.show_deprecation_warning()
        if dist.is_initialized():
            if dist.get_rank() == 0:
                logger.debug(msg)
        else:
            logger.debug(msg)

    @staticmethod
    def null_logging(msg):
        Logging.show_deprecation_warning()
        with redirect_stdout(None):
            print(msg)
