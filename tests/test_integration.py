"""Integration and smoke tests."""

import pytest


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_config_to_dict_roundtrip(self, config):
        """Test converting config to dict and back."""
        config_dict = config.to_dict()
        
        # Verify structure
        assert "env" in config_dict
        assert "model" in config_dict
        assert "server" in config_dict
        
        # Verify values preserved
        assert config_dict["env"]["omp_threads"] == config.env.omp_threads
        assert config_dict["model"]["ctx_size"] == config.model.ctx_size
        assert config_dict["server"]["port"] == config.server.port

    def test_environment_setup_complete(self, config):
        """Test complete environment setup."""
        from runLLMAgentForAgenticDevs import apply_environment_config
        import os
        
        apply_environment_config(config)
        
        # Verify environment is properly configured
        assert "OMP_NUM_THREADS" in os.environ
        assert "TOKENIZERS_PARALLELISM" in os.environ


class TestTextProcessingIntegration:
    """Integration tests for text processing."""

    def test_text_extraction_pipeline(self):
        """Test complete text extraction pipeline."""
        from runLLMAgentForAgenticDevs import extract_text, hard_clip
        
        # Test various input types
        inputs = [
            "Simple string",
            None,
            ["List", "of", "strings"],
            {"type": "text", "text": "Dict with text"},
            [{"text": "Mixed"}, "types"],
        ]
        
        for inp in inputs:
            result = extract_text(inp)
            assert isinstance(result, str)
            clipped = hard_clip(result, max_chars=100)
            assert isinstance(clipped, str)

    def test_large_text_handling(self):
        """Test handling of very large text."""
        from runLLMAgentForAgenticDevs import extract_text, hard_clip
        
        # Create a large text
        large_text = "x" * 10000
        
        # Extract and clip
        extracted = extract_text(large_text)
        clipped = hard_clip(extracted, max_chars=1000)
        
        assert len(clipped) == 1000
        assert clipped == large_text[-1000:]


class TestTUIIntegrationFull:
    """Full integration tests for TUI system."""

    def test_tui_startup_to_status_flow(self, tui, config):
        """Test complete flow from startup to status tracking."""
        # Startup sequence
        assert not tui.model_loaded
        assert not tui.server_running
        
        # Mark completion
        tui.mark_model_loaded()
        tui.mark_server_running()
        
        assert tui.model_loaded
        assert tui.server_running
        assert tui.start_time is not None
        
        # Track requests
        tui.record_request(success=True, tokens=128)
        tui.record_request(success=True, tokens=256)
        tui.record_request(success=False, tokens=0)
        
        assert tui.request_count == 3
        assert tui.error_count == 1
        assert tui.total_tokens == 384
        
        # Build status
        status = tui.build_status_table()
        assert status is not None


class TestSmokeTests:
    """Smoke tests - basic functionality verification."""

    def test_imports_successful(self):
        """Test that all main modules can be imported."""
        try:
            from runLLMAgentForAgenticDevs import (
                Config, EnvironmentConfig, ModelConfig, ServerConfig,
                extract_text, hard_clip, setup_logging
            )
            from training_src.tui import ServerStatusTUI, get_tui
            from training_src.config import CONFIG
            
            assert Config is not None
            assert ServerStatusTUI is not None
            assert CONFIG is not None
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_config_creation(self):
        """Test that configs can be created."""
        from runLLMAgentForAgenticDevs import Config
        
        config = Config()
        assert config is not None
        assert config.env is not None
        assert config.model is not None
        assert config.server is not None

    def test_tui_creation(self):
        """Test that TUI can be created."""
        from training_src.tui import ServerStatusTUI
        
        tui = ServerStatusTUI()
        assert tui is not None
        assert hasattr(tui, 'show_welcome_banner')
        assert hasattr(tui, 'record_request')

    def test_text_utils_functions_exist(self):
        """Test that text utility functions are callable."""
        from runLLMAgentForAgenticDevs import extract_text, hard_clip
        
        assert callable(extract_text)
        assert callable(hard_clip)
        
        # Basic smoke test
        result1 = extract_text("test")
        result2 = hard_clip("test", max_chars=10)
        
        assert result1 == "test"
        assert result2 == "test"

    def test_logging_setup(self):
        """Test logging can be set up."""
        from runLLMAgentForAgenticDevs import setup_logging, Config
        
        config = Config()
        logger = setup_logging(config)
        
        assert logger is not None
        assert callable(logger.info)
        assert callable(logger.error)
        assert callable(logger.debug)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_extract_text_with_empty_list(self):
        """Test extracting text from empty list."""
        from runLLMAgentForAgenticDevs import extract_text
        
        result = extract_text([])
        assert result == ""

    def test_extract_text_with_nested_structures(self):
        """Test extracting text from complex nested structures."""
        from runLLMAgentForAgenticDevs import extract_text
        
        content = [
            "Start",
            [{"text": "Nested"}],  # Will be stringified
            {"type": "text", "text": "End"},
        ]
        result = extract_text(content)
        # Should not raise error
        assert isinstance(result, str)

    def test_hard_clip_with_zero_chars(self):
        """Test hard_clip with zero character limit."""
        from runLLMAgentForAgenticDevs import hard_clip
        
        result = hard_clip("Hello", max_chars=0)
        assert result == ""

    def test_hard_clip_with_negative_chars(self):
        """Test hard_clip with negative character limit."""
        from runLLMAgentForAgenticDevs import hard_clip
        
        # Should handle gracefully
        result = hard_clip("Hello", max_chars=-1)
        # Should return empty or last 0 chars
        assert isinstance(result, str)


class TestDataValidation:
    """Test data validation and constraints."""

    def test_config_ranges(self, config):
        """Test that config values are in reasonable ranges."""
        assert config.env.omp_threads > 0
        assert config.server.port > 0
        assert config.server.port < 65536
        assert 0 <= config.server.temperature <= 1.0

    def test_tui_metrics_sanity(self, tui):
        """Test that TUI metrics stay sane."""
        for _ in range(100):
            tui.record_request(success=True, tokens=50)
        
        assert tui.request_count == 100
        assert tui.total_tokens == 5000
        assert tui.error_count == 0
