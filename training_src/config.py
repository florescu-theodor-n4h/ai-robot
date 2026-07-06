"""Configuration and environment setup for LLM server."""
import os
import logging

log = logging.getLogger("llm-server.config")


def setup_environment():
    """Harden environment variables and return config dict."""
    log.info("Setting up environment variables...")

    # Thread/parallelism
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    log.debug("Tokenizer parallelism disabled, OMP threads set to 4")

    # AMD/ROCm hints
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    os.environ["GGML_OPENCL"] = "1"
    log.debug("AMD/ROCm environment configured")

    # CUDA disabled (using CPU/ROCm)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    log.debug("CUDA disabled, using CPU/ROCm")

    return {
        "omp_threads": 4,
        "tokenizer_parallel": False,
        "cuda_enabled": False,
        "rocm_enabled": True,
    }


def get_model_config():
    """Return model repository and file configuration."""
    config = {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "ctx_size": 2048,
        "threads": 4,
        "verbose": False,
    }
    log.info(f"Model config: repo={config['repo']}, file={config['file']}, ctx={config['ctx_size']}")
    return config


def get_server_config():
    """Return server configuration."""
    return {
        "host": "0.0.0.0",
        "port": 8888,
        "log_level": "warning",
        "max_messages": 6,
        "max_tokens": 512,
        "temperature": 0.6,
        "prompt_max_chars": 4000,
        "hard_clip_chars": 3000,
        "safety_trim_chars": 6000,
    }


SYSTEM_PROMPT = """You are a precise coding assistant.
Be minimal, correct, and do not hallucinate APIs."""
