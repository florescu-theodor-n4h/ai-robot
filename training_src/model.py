"""Model loading and inference utilities."""
import logging
import traceback
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

log = logging.getLogger("llm-server.model")


class ModelLoader:
    """Handles model download and initialization."""

    def __init__(self, repo_id, filename, ctx_size=2048, n_threads=4, verbose=False):
        self.repo_id = repo_id
        self.filename = filename
        self.ctx_size = ctx_size
        self.n_threads = n_threads
        self.verbose = verbose
        self.model = None
        self.model_path = None

    def load(self):
        """Download and load the model."""
        log.info(f"Loading model: {self.repo_id}/{self.filename}")
        try:
            log.debug(f"Downloading from HuggingFace: {self.repo_id}")
            self.model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
            )
            log.info(f"Model downloaded to: {self.model_path}")

            log.debug(
                f"Initializing Llama with ctx={self.ctx_size}, threads={self.n_threads}"
            )
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.ctx_size,
                n_threads=self.n_threads,
                verbose=self.verbose,
            )
            log.info("Model loaded successfully")
            return self.model

        except Exception as e:
            log.error(f"Failed to load model: {e}")
            log.debug(traceback.format_exc())
            raise

    def generate(self, prompt, max_tokens=512, temperature=0.6, stop_sequences=None):
        """Generate text from the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if stop_sequences is None:
            stop_sequences = ["</s>", "<|user|>"]

        log.debug(
            f"Generating: max_tokens={max_tokens}, temperature={temperature}, "
            f"prompt_len={len(prompt)}"
        )

        try:
            result = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
            )
            text = result["choices"][0]["text"]
            log.debug(f"Generated {len(text)} characters")
            return text

        except Exception as e:
            log.error(f"Generation failed: {e}")
            log.debug(traceback.format_exc())
            raise
