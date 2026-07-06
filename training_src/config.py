"""Re-export configuration from main runLLMAgentForAgenticDevs module.

This module provides compatibility with modular imports.
All configuration is source-controlled in runLLMAgentForAgenticDevs.py.
"""

import sys
import os

# Add parent directory to path to import main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runLLMAgentForAgenticDevs import (
    CONFIG,
    EnvironmentConfig,
    ModelConfig,
    ServerConfig,
    apply_environment_config,
)

__all__ = [
    "CONFIG",
    "EnvironmentConfig",
    "ModelConfig",
    "ServerConfig",
    "apply_environment_config",
]
