"""FastAPI routes and endpoints."""
import logging
import traceback
from fastapi import FastAPI, Request
from pydantic import BaseModel

log = logging.getLogger("llm-server.api")


class Prompt(BaseModel):
    prompt: str
    system: str | None = None
    temperature: float | None = 0.6


def create_app(model_loader, config):
    """Create and configure FastAPI app."""
    log.info("Creating FastAPI application...")

    app = FastAPI(title="LLM Server")
    server_config = config["server"]

    @app.post("/chat")
    def chat(data: Prompt):
        """Simple chat endpoint (legacy API)."""
        log.info(f"POST /chat: prompt_len={len(data.prompt)}")
        try:
            response = model_loader.generate(
                prompt=data.prompt,
                temperature=data.temperature or 0.6,
            )
            log.debug(f"Chat response: {len(response)} chars")
            return {"response": response}
        except Exception as e:
            log.error(f"Chat request failed: {e}")
            log.debug(traceback.format_exc())
            return {"error": "generation_failed", "detail": str(e)}

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        """OpenAI-compatible chat endpoint."""
        log.info("POST /v1/chat/completions (OpenAI compatible)")
        try:
            data = await request.json()
            messages = data.get("messages", [])
            temperature = float(data.get("temperature", server_config["temperature"]))
            max_tokens = int(data.get("max_tokens", server_config["max_tokens"]))

            log.debug(
                f"OpenAI request: messages={len(messages)}, "
                f"temperature={temperature}, max_tokens={max_tokens}"
            )

            # Context limit
            max_messages = server_config["max_messages"]
            if len(messages) > max_messages:
                log.warning(
                    f"Truncating messages from {len(messages)} to {max_messages}"
                )
                messages = messages[-max_messages:]

            system = config["system_prompt"]

            # Build prompt from messages
            prompt_parts = []
            for i, msg in enumerate(messages):
                role = msg.get("role", "")
                content = _safe_extract(msg.get("content"))

                if not content:
                    log.debug(f"  message[{i}]: empty content, skipping")
                    continue

                if role == "system":
                    system = content
                    log.debug(f"  message[{i}]: system prompt updated")
                elif role == "user":
                    prompt_parts.append("User: " + content)
                    log.debug(f"  message[{i}]: user message added")
                elif role == "assistant":
                    prompt_parts.append("Assistant: " + content)
                    log.debug(f"  message[{i}]: assistant message added")

            prompt = "\n".join(prompt_parts)
            log.debug(f"Combined prompt: {len(prompt)} chars")

            # Safety trims
            max_chars = server_config["prompt_max_chars"]
            if len(prompt) > max_chars:
                log.warning(f"Hard clipping prompt from {len(prompt)} to {max_chars}")
                prompt = prompt[-max_chars:]

            full_prompt = f"""<|system|>
{system}

<|assistant|>
{prompt}

<|assistant|>
"""

            hard_clip_limit = server_config["hard_clip_chars"]
            full_prompt = full_prompt[-hard_clip_limit:]
            log.debug(f"Final prompt: {len(full_prompt)} chars")

            # Extra safety check
            safety_limit = server_config["safety_trim_chars"]
            if len(full_prompt) > safety_limit:
                log.warning(f"Emergency trim: {len(full_prompt)} -> {safety_limit}")
                full_prompt = full_prompt[-safety_limit:]

            text = model_loader.generate(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            log.info(f"OpenAI response generated: {len(text)} chars")

            return {
                "id": "local",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            }

        except Exception as e:
            log.error(f"OpenAI request failed: {e}")
            log.debug(traceback.format_exc())
            return {"error": "generation_failed", "detail": str(e)}

    @app.get("/v1/models")
    def models():
        """List available models."""
        log.debug("GET /v1/models")
        return {
            "object": "list",
            "data": [{"id": "local-llama", "object": "model"}],
        }

    @app.get("/health")
    def health():
        """Health check endpoint."""
        log.debug("GET /health")
        return {"status": "ok", "model": config["model"]["repo"]}

    log.info("FastAPI application created successfully")
    return app


def _safe_extract(content):
    """Safely extract text from various content types."""
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
