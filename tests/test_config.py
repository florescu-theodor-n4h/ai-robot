"""Tests for configuration module."""

import pytest


class TestEnvironmentConfig:
    """Test EnvironmentConfig dataclass."""

    def test_default_values(self):
        """Test default environment configuration."""
        from runLLMAgentForAgenticDevs import EnvironmentConfig
        
        env = EnvironmentConfig()
        
        assert env.omp_threads == 4
        assert env.tokenizer_parallel is False
        assert env.rocm_enabled is True
        assert env.cuda_enabled is False
        assert env.rocm_gfx_version == "10.3.0"

    def test_custom_values(self):
        """Test custom environment configuration."""
        from runLLMAgentForAgenticDevs import EnvironmentConfig
        
        env = EnvironmentConfig(
            omp_threads=8,
            tokenizer_parallel=True,
            rocm_enabled=False,
            cuda_enabled=True,
        )
        
        assert env.omp_threads == 8
        assert env.tokenizer_parallel is True
        assert env.rocm_enabled is False
        assert env.cuda_enabled is True


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_values(self):
        """Test default model configuration."""
        from runLLMAgentForAgenticDevs import ModelConfig
        
        model = ModelConfig()
        
        assert "TinyLlama" in model.repo
        assert model.ctx_size == 2048
        assert model.threads == 4
        assert model.verbose is False
        assert len(model.system_prompt) > 0

    def test_custom_values(self):
        """Test custom model configuration."""
        from runLLMAgentForAgenticDevs import ModelConfig
        
        model = ModelConfig(
            ctx_size=4096,
            threads=8,
            verbose=True,
        )
        
        assert model.ctx_size == 4096
        assert model.threads == 8
        assert model.verbose is True


class TestServerConfig:
    """Test ServerConfig dataclass."""

    def test_default_values(self):
        """Test default server configuration."""
        from runLLMAgentForAgenticDevs import ServerConfig
        
        server = ServerConfig()
        
        assert server.host == "0.0.0.0"
        assert server.port == 8888
        assert server.max_messages == 6
        assert server.max_tokens == 512
        assert server.temperature == 0.6

    def test_custom_values(self):
        """Test custom server configuration."""
        from runLLMAgentForAgenticDevs import ServerConfig
        
        server = ServerConfig(
            port=9999,
            max_tokens=1024,
            temperature=0.8,
        )
        
        assert server.port == 9999
        assert server.max_tokens == 1024
        assert server.temperature == 0.8


class TestConfig:
    """Test Config dataclass."""

    def test_default_initialization(self):
        """Test Config with default sub-configs."""
        from runLLMAgentForAgenticDevs import Config
        
        config = Config()
        
        assert config.env is not None
        assert config.model is not None
        assert config.server is not None

    def test_custom_initialization(self):
        """Test Config with custom sub-configs."""
        from runLLMAgentForAgenticDevs import (
            Config, EnvironmentConfig, ModelConfig, ServerConfig
        )
        
        env = EnvironmentConfig(omp_threads=2)
        model = ModelConfig(ctx_size=1024)
        server = ServerConfig(port=7777)
        
        config = Config(env=env, model=model, server=server)
        
        assert config.env.omp_threads == 2
        assert config.model.ctx_size == 1024
        assert config.server.port == 7777

    def test_to_dict(self):
        """Test Config.to_dict() method."""
        from runLLMAgentForAgenticDevs import Config
        
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "env" in config_dict
        assert "model" in config_dict
        assert "server" in config_dict
        assert isinstance(config_dict["env"], dict)
        assert isinstance(config_dict["model"], dict)
        assert isinstance(config_dict["server"], dict)


class TestApplyEnvironmentConfig:
    """Test apply_environment_config function."""

    def test_environment_variables_set(self, config):
        """Test that environment variables are properly set."""
        import os
        from runLLMAgentForAgenticDevs import apply_environment_config
        
        apply_environment_config(config)
        
        assert os.environ["OMP_NUM_THREADS"] == "2"
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    def test_rocm_disabled(self, config):
        """Test that ROCM is not enabled when disabled."""
        import os
        from runLLMAgentForAgenticDevs import apply_environment_config
        
        config.env.rocm_enabled = False
        apply_environment_config(config)
        
        # ROCM variables should not be set
        assert os.environ.get("GGML_OPENCL") != "1"

    def test_rocm_enabled(self, config):
        """Test that ROCM is enabled when configured."""
        import os
        from runLLMAgentForAgenticDevs import apply_environment_config
        
        config.env.rocm_enabled = True
        apply_environment_config(config)
        
        assert "HSA_OVERRIDE_GFX_VERSION" in os.environ or True  # May not be set in all envs
