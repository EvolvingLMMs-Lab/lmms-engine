from contextlib import redirect_stdout
from typing import Any, Dict

import torch.distributed as dist
from loguru import logger
from rich.logging import RichHandler


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
