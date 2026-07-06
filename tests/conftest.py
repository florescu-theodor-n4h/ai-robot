"""Test configuration and fixtures."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def config():
    """Fixture providing test configuration."""
    from runLLMAgentForAgenticDevs import Config, EnvironmentConfig, ModelConfig, ServerConfig
    
    return Config(
        env=EnvironmentConfig(
            omp_threads=2,
            tokenizer_parallel=False,
            rocm_enabled=False,
            cuda_enabled=False,
        ),
        model=ModelConfig(),
        server=ServerConfig(port=9999, max_messages=3),
    )


@pytest.fixture
def tui():
    """Fixture providing TUI instance."""
    from training_src.tui import ServerStatusTUI
    
    return ServerStatusTUI()


@pytest.fixture
def logger():
    """Fixture providing logger."""
    import logging
    
    return logging.getLogger("test-logger")
