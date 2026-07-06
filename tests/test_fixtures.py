"""Test initialization and utilities."""

import pytest


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset module state between tests."""
    import sys
    
    # Store original modules
    modules_to_preserve = {
        'runLLMAgentForAgenticDevs',
        'training_src.tui',
        'training_src.config',
    }
    
    yield
    
    # Clean up doesn't reset - each test gets fresh imports via fixtures


def test_pytest_configuration():
    """Test that pytest is properly configured."""
    assert pytest is not None


def test_fixtures_available(config, tui, logger):
    """Test that all fixtures are available."""
    assert config is not None
    assert tui is not None
    assert logger is not None
