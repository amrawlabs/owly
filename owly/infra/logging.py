"""Logging utilities for Owly runtime."""

from __future__ import annotations

import logging


def get_logger(name: str = "owly") -> logging.Logger:
    """Return a configured logger without global side effects."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger
