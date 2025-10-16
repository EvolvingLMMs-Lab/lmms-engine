"""Utility functions for handling optional imports."""

from importlib import import_module
from typing import Optional


def try_import(module_name: str) -> Optional[object]:
    """
    Try to import a module, returning None if it's not available.

    Args:
        module_name: The name of the module to import.

    Returns:
        The imported module if available, None otherwise.
    """
    try:
        return import_module(module_name)
    except ImportError:
        return None
