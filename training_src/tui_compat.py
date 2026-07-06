"""Re-export TUI from training_src module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_src.tui import ServerStatusTUI, get_tui

__all__ = ["ServerStatusTUI", "get_tui"]
