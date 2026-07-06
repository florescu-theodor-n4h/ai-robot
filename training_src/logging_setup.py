"""Re-export logging utilities from main module.

All logging is configured in runLLMAgentForAgenticDevs.py.
This module provides compatibility with modular imports.
"""

import logging

def get_logger(module_name):
    """Get a named logger for a module."""
    return logging.getLogger(f"llm-server.{module_name}")


def setup_logging(config=None, level=logging.DEBUG):
    """Setup logging (forwarded to main module)."""
    return logging.getLogger("llm-server")

