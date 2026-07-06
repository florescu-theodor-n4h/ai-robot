#!/bin/bash ./runVenv-inference.sh

import os
import sys
import signal
import logging
import traceback
from dataclasses import dataclass
from typing import Dict, Any

from fastapi import FastAPI, Request
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# =========================
# CONFIGURATION (SOURCE OF TRUTH)
# =========================

@dataclass
class EnvironmentConfig:
    """Environment and hardware settings."""
    omp_threads: int = 4
    tokenizer_parallel: bool = False
    rocm_enabled: bool = True
    cuda_enabled: bool = False
    rocm_gfx_version: str = "10.3.0"

@dataclass
class ModelConfig:
    """Model and inference settings."""
    repo: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    file: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    ctx_size: int = 2048
    threads: int = 4
    verbose: bool = False
    system_prompt: str = """You are a precise coding assistant.
Be minimal, correct, and do not hallucinate APIs."""

@dataclass
class ServerConfig:
    """Server and API settings."""
    host: str = "0.0.0.0"
    port: int = 8888
    log_level: str = "warning"
    max_messages: int = 6
    max_tokens: int = 512
    temperature: float = 0.6
    prompt_max_chars: int = 4000
    hard_clip_chars: int = 3000
    safety_trim_chars: int = 6000

@dataclass
class Config:
    """Complete server configuration."""
    env: EnvironmentConfig = None
    model: ModelConfig = None
    server: ServerConfig = None

    def __post_init__(self):
        if self.env is None:
            self.env = EnvironmentConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.server is None:
            self.server = ServerConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "env": self.env.__dict__,
            "model": self.model.__dict__,
            "server": self.server.__dict__,
        }


# Global config instance
CONFIG = Config()


def apply_environment_config(config: Config):
    """Apply environment configuration to os.environ."""
    os.environ["OMP_NUM_THREADS"] = str(config.env.omp_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if config.env.tokenizer_parallel else "false"

    if config.env.rocm_enabled:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = config.env.rocm_gfx_version
        os.environ["GGML_OPENCL"] = "1"

    if not config.env.cuda_enabled:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""




# =========================
# LOGGING
# =========================

def setup_logging(config: Config, level=logging.DEBUG):
    """Setup structured logging."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("llm-server")

# =========================
# APP
# =========================

log = None
app = None
llm = None


def initialize_app(config: Config):
    """Initialize FastAPI app and model."""
    global log, app, llm

    log = setup_logging(config)
    log.info("=" * 60)
    log.info("LLM Agentic Dev Server Starting")
    log.info("=" * 60)

    apply_environment_config(config)
    log.debug(f"Environment config applied: {config.env.__dict__}")

    # Load model
    log.info(f"Loading model: {config.model.repo}/{config.model.file}")
    try:
        model_path = hf_hub_download(
            repo_id=config.model.repo,
            filename=config.model.file
        )
        log.info(f"Model downloaded to: {model_path}")

        llm = Llama(
            model_path=model_path,
            n_ctx=config.model.ctx_size,
            n_threads=config.model.threads,
            verbose=config.model.verbose
        )
        log.info("Model loaded successfully")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        log.debug(traceback.format_exc())
        raise

    # Create FastAPI app
    app = FastAPI(title="LLM Agentic Dev Server")
    log.info("FastAPI application created")

    # Setup routes
    _setup_routes(app, config)
    log.info("Routes registered")

    return app, llm, config

# =========================
# TEXT UTILITIES
# =========================

def extract_text(content):
    """Safely extract text from various content types."""
    if content is None:
        log.debug("extract_text: content is None")
        return ""

    if isinstance(content, str):
        log.debug(f"extract_text: string content, len={len(content)}")
        return content

    if isinstance(content, list):
        out = []
        log.debug(f"extract_text: list with {len(content)} items")
        for i, c in enumerate(content):
            if isinstance(c, str):
                out.append(c)
                log.debug(f"  [{i}] string: {len(c)} chars")
            elif isinstance(c, dict):
                if c.get("type") == "text":
                    text = c.get("text", "")
                    out.append(text)
                    log.debug(f"  [{i}] dict type=text: {len(text)} chars")
                elif "text" in c:
                    text = str(c["text"])
                    out.append(text)
                    log.debug(f"  [{i}] dict with text key: {len(text)} chars")
        result = "\n".join(out)
        log.debug(f"extract_text: total {len(result)} chars")
        return result

    log.debug(f"extract_text: unknown type, converting to string")
    return str(content)


def hard_clip(text, max_chars=3000):
    """Aggressively clip text to max_chars from the end."""
    if not text:
        return ""
    if len(text) > max_chars:
        log.debug(f"hard_clip: {len(text)} -> {max_chars}")
        return text[-max_chars:]
    return text

# =========================
# LEGACY API
# =========================

class Prompt(BaseModel):
    prompt: str
    system: str | None = None
    temperature: float | None = None


def generate(prompt, system=None, temperature=None, config: Config = None):
    """Generate text using the model."""
    if config is None:
        config = CONFIG
    if temperature is None:
        temperature = config.server.temperature
    if system is None:
        system = config.model.system_prompt

    if log:
        log.info(f"generate(): prompt_len={len(prompt)}, temp={temperature}")

    full_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"

    try:
        result = llm(
            full_prompt,
            max_tokens=config.server.max_tokens,
            temperature=temperature,
            stop=["</s>", "<|user|>"]
        )
        text = result["choices"][0]["text"]
        if log:
            log.debug(f"generate(): produced {len(text)} chars")
        return text

    except Exception as e:
        if log:
            log.error(f"generate() failed: {e}")
            log.debug(traceback.format_exc())
        return "generation error"


def _setup_routes(app, config):
    """Register all routes with the app."""

    @app.post("/chat")
    def chat(data: Prompt):
        if log:
            log.info("POST /chat")
        return {"response": generate(data.prompt, data.system, data.temperature, config)}

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        if log:
            log.info("POST /v1/chat/completions")
        try:
            data = await request.json()
            messages = data.get("messages", [])
            temperature = float(data.get("temperature", config.server.temperature))
            max_tokens = int(data.get("max_tokens", config.server.max_tokens))

            if log:
                log.debug(f"OpenAI request: messages={len(messages)}, temp={temperature}, max_tokens={max_tokens}")

            # Context limit
            max_messages = config.server.max_messages
            if len(messages) > max_messages:
                if log:
                    log.warning(f"Truncating {len(messages)} messages to {max_messages}")
                messages = messages[-max_messages:]

            system = config.model.system_prompt

            # Build prompt
            prompt_parts = []
            for i, msg in enumerate(messages):
                role = msg.get("role", "")
                content = extract_text(msg.get("content"))

                if not content:
                    if log:
                        log.debug(f"  msg[{i}]: skipping empty content")
                    continue

                if role == "system":
                    system = content
                    if log:
                        log.debug(f"  msg[{i}]: system prompt updated")
                elif role == "user":
                    prompt_parts.append("User: " + content)
                    if log:
                        log.debug(f"  msg[{i}]: user message added")
                elif role == "assistant":
                    prompt_parts.append("Assistant: " + content)
                    if log:
                        log.debug(f"  msg[{i}]: assistant message added")

            prompt = "\n".join(prompt_parts)
            if log:
                log.debug(f"Combined prompt: {len(prompt)} chars")

            # Safety trims
            max_chars = config.server.prompt_max_chars
            if len(prompt) > max_chars:
                if log:
                    log.warning(f"Hard clip prompt: {len(prompt)} -> {max_chars}")
                prompt = prompt[-max_chars:]

            full_prompt = f"""<|system|>
{system}

<|assistant|>
{prompt}

<|assistant|>
"""

            full_prompt = hard_clip(full_prompt, config.server.hard_clip_chars)
            if log:
                log.debug(f"Final prompt: {len(full_prompt)} chars")

            if len(full_prompt) > config.server.safety_trim_chars:
                if log:
                    log.warning(f"Emergency trim: {len(full_prompt)} -> {config.server.safety_trim_chars}")
                full_prompt = full_prompt[-config.server.safety_trim_chars:]

            text = llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "<|user|>"]
            )["choices"][0]["text"]

            if log:
                log.info(f"OpenAI response: {len(text)} chars")

            return {
                "id": "local",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text
                        }
                    }
                ]
            }

        except Exception as e:
            if log:
                log.error(f"OpenAI request failed: {e}")
                log.debug(traceback.format_exc())
            return {"error": "generation_failed", "detail": str(e)}

    @app.get("/v1/models")
    def models():
        if log:
            log.debug("GET /v1/models")
        return {
            "object": "list",
            "data": [
                {"id": "local-llama", "object": "model"}
            ]
        }

    @app.get("/health")
    def health():
        if log:
            log.debug("GET /health")
        return {
            "status": "ok",
            "model": config.model.repo
        }

# =========================
# RUN
# =========================

def setup_signal_handlers(log):
    """Register signal handlers for graceful shutdown."""
    def shutdown_handler(sig, frame):
        log.warning(f"Received signal {sig}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    log.debug("Signal handlers registered")


if __name__ == "__main__":
    import uvicorn

    # Initialize everything
    app, llm, config = initialize_app(CONFIG)
    setup_signal_handlers(log)

    log.info("=" * 60)
    log.info(f"Starting server on {config.server.host}:{config.server.port}")
    log.info("=" * 60)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
    )
