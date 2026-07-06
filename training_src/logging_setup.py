"""Centralized logging configuration."""
import logging
import sys
from datetime import datetime


def setup_logging(name="llm-server", level=logging.INFO):
    """Configure structured logging with timestamps and levels."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"Logging initialized: level={logging.getLevelName(level)}")
    return logger


def get_logger(module_name):
    """Get a named logger for a module."""
    return logging.getLogger(f"llm-server.{module_name}")
