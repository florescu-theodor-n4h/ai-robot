"""Re-export model utilities from main module.

All model loading is defined in runLLMAgentForAgenticDevs.py.
This module provides compatibility with modular imports.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("llm-server.model")


class ModelLoader:
    """Wrapper for model access."""

    def __init__(self, repo_id, filename, ctx_size=2048, n_threads=4, verbose=False):
        self.repo_id = repo_id
        self.filename = filename
        self.ctx_size = ctx_size
        self.n_threads = n_threads
        self.verbose = verbose
        self.model = None

    def load(self):
        """Load model from main module."""
        from runLLMAgentForAgenticDevs import llm as main_llm
        self.model = main_llm
        log.info(f"Model loaded: {self.repo_id}/{self.filename}")
        return self.model

    def generate(self, prompt, max_tokens=512, temperature=0.6, stop_sequences=None):
        """Generate text using the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if stop_sequences is None:
            stop_sequences = ["</s>", "<|user|>"]

        log.debug(f"Generating: {len(prompt)} chars, max_tokens={max_tokens}")
        result = self.model(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop_sequences)
        return result["choices"][0]["text"]

