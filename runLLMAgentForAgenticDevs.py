#!/usr/bin/env python3

import os
import sys
import time
import signal
import logging
import traceback

from fastapi import FastAPI, Request
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# =========================
# ENVIRONMENT HARDENING
# =========================

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# AMD / ROCm hints (safe if ignored)
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0" # Compatibilitate Beige Goby
os.environ["GGML_OPENCL"] = "1"

# prevent CUDA confusion
os.environ["CUDA_VISIBLE_DEVICES"] = ""







# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)

log = logging.getLogger("llm-server")

# =========================
# SHUTDOWN
# =========================

def shutdown_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# =========================
# MODEL
# =========================

MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

# =========================
# PROMPT
# =========================

SYSTEM_PROMPT = """
You are a precise coding assistant.
Be minimal, correct, and do not hallucinate APIs.
"""

# =========================
# APP
# =========================

app = FastAPI(title="LLM Server")

# =========================
# SAFE TEXT EXTRACTION (IMPORTANT FIX)
# =========================

def extract_text(content):
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        out = []
        for c in content:
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, dict):
                if c.get("type") == "text":
                    out.append(c.get("text", ""))
                elif "text" in c:
                    out.append(str(c["text"]))
        return "\n".join(out)

    return str(content)

# =========================
# LEGACY API
# =========================

class Prompt(BaseModel):
    prompt: str
    system: str | None = None
    temperature: float | None = 0.6


def generate(prompt, system=None, temperature=0.6):
    system = system or SYSTEM_PROMPT

    full_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"

    try:
        result = llm(
            full_prompt,
            max_tokens=512,
            temperature=temperature,
            stop=["</s>", "<|user|>"]
        )

        return result["choices"][0]["text"]

    except Exception:
        log.error(traceback.format_exc())
        return "generation error"


@app.post("/chat")
def chat(data: Prompt):
    return {"response": generate(data.prompt, data.system, data.temperature)}

# =========================
# OPENAI COMPATIBLE ENDPOINT (FIXED)
# =========================

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    temperature = float(data.get("temperature", 0.6))

    system = SYSTEM_PROMPT

    # =========================
    # AGGRESSIVE CONTEXT LIMIT (CRITICAL FIX)
    # =========================

    MAX_MESSAGES = 6  # keeps agent working but prevents explosion
    messages = messages[-MAX_MESSAGES:]

    def safe_extract(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            out = []
            for c in content:
                if isinstance(c, dict):
                    out.append(c.get("text", ""))
                else:
                    out.append(str(c))
            return "\n".join(out)
        return str(content)

    prompt_parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = safe_extract(msg.get("content"))

        if not content:
            continue

        if role == "system":
            system = content

        elif role == "user":
            prompt_parts.append("User: " + content)

        elif role == "assistant":
            prompt_parts.append("Assistant: " + content)

    # =========================
    # HARD SAFETY TRIM (IMPORTANT)
    # =========================

    prompt = "\n".join(prompt_parts)

    # hard cut to prevent 14k+ token explosions
    prompt = prompt[-4000:]

    full_prompt = f"""<|system|>
    {system}

    <|assistant|>
    {prompt}

    <|assistant|>
    """

    def hard_clip(text, max_chars=3000):
        if not text:
            return ""
        return text[-max_chars:]

    # FINAL EMERGENCY TRIM (DO NOT RELY ON EARLIER STEPS)
    full_prompt = hard_clip(full_prompt)

    # EXTRA SAFETY: if still too big, aggressively shrink again
    if len(full_prompt) > 6000:
        full_prompt = full_prompt[-6000:]


    try:
        result = llm(
            full_prompt,
            max_tokens=int(data.get("max_tokens", 512)),
            temperature=temperature,
            stop=["</s>", "<|user|>"]
        )

        text = result["choices"][0]["text"]

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

    except Exception:
        log.error(traceback.format_exc())
        return {"error": "generation_failed"}

# =========================
# MODELS
# =========================

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": "local-llama", "object": "model"}
        ]
    }

# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_REPO
    }

# =========================
# RUN
# =========================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8888,
        log_level="warning"
    )
