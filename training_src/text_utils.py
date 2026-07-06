"""Text processing utilities."""
import logging

log = logging.getLogger("llm-server.text")


def extract_text(content):
    """Safely extract text from various content types."""
    if content is None:
        log.debug("extract_text: content is None, returning empty string")
        return ""

    if isinstance(content, str):
        log.debug(f"extract_text: string content, len={len(content)}")
        return content

    if isinstance(content, list):
        out = []
        log.debug(f"extract_text: list content with {len(content)} items")
        for i, c in enumerate(content):
            if isinstance(c, str):
                out.append(c)
                log.debug(f"  [{i}] string: {len(c)} chars")
            elif isinstance(c, dict):
                if c.get("type") == "text":
                    text = c.get("text", "")
                    out.append(text)
                    log.debug(f"  [{i}] dict with type=text: {len(text)} chars")
                elif "text" in c:
                    text = str(c["text"])
                    out.append(text)
                    log.debug(f"  [{i}] dict with text key: {len(text)} chars")
        result = "\n".join(out)
        log.debug(f"extract_text: extracted {len(result)} chars total")
        return result

    log.debug(f"extract_text: unknown type {type(content)}, converting to string")
    return str(content)


def hard_clip(text, max_chars=3000):
    """Aggressively clip text to max_chars from the end."""
    if not text:
        return ""

    if len(text) > max_chars:
        clipped = text[-max_chars:]
        log.debug(f"hard_clip: clipped from {len(text)} to {len(clipped)} chars")
        return clipped

    log.debug(f"hard_clip: text within limit ({len(text)} <= {max_chars})")
    return text
