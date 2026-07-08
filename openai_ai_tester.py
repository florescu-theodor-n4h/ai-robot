#!/usr/bin/env python3
"""Curses-based OpenAI-compatible chat tester for localhost:8080."""

from __future__ import annotations

import argparse
import curses
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_ENDPOINT = "http://localhost:8080/v1"
DEFAULT_MODEL = "local-model"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful coding assistant. Answer clearly and focus on practical "
    "Python feedback when the user shares a .py file."
)


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class TesterConfig:
    base_url: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    timeout: float = 60.0
    temperature: float = 0.2
    max_tokens: int = 800
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


TesterConfig.__test__ = False


def normalize_base_url(base_url: str) -> str:
    """Normalize a base URL to an OpenAI-compatible /v1 endpoint."""
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("Endpoint cannot be empty.")
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


def stringify_content(content: Any) -> str:
    """Convert OpenAI-style message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return json.dumps(content, ensure_ascii=False)


def build_chat_payload(config: TesterConfig, messages: list[ChatMessage]) -> dict[str, Any]:
    """Build the JSON payload for an OpenAI-compatible chat request."""
    return {
        "model": config.model,
        "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }


def extract_assistant_text(data: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-compatible response."""
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Malformed response: {data!r}") from exc

    content = stringify_content(message.get("content", ""))
    if content.strip():
        return content.strip()

    finish_reason = choice.get("finish_reason")
    raise ValueError(f"Assistant returned no text (finish_reason={finish_reason!r}).")


def load_python_file(path_text: str) -> tuple[Path, str]:
    """Read and validate a Python file for /test commands."""
    path = Path(path_text).expanduser()
    if path.suffix != ".py":
        raise ValueError("Only .py files are supported by /test.")
    if not path.exists():
        raise FileNotFoundError(f"Python file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")
    return path, path.read_text(encoding="utf-8")


def build_python_test_prompt(path: Path, source: str) -> str:
    """Build a prompt that asks the model to inspect a Python file."""
    return (
        f"Test and review this Python file named {path.name}.\n"
        "Focus on runtime errors, edge cases, correctness, and quick improvement ideas.\n\n"
        f"```python\n{source}\n```"
    )


class OpenAICompatClient:
    """Small OpenAI-compatible HTTP client."""

    def __init__(self, config: TesterConfig):
        self.config = TesterConfig(
            base_url=normalize_base_url(config.base_url),
            model=config.model,
            timeout=config.timeout,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
        )

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        http_request = request.Request(
            url=f"{self.config.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )

        try:
            with request.urlopen(http_request, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {path}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach {self.config.base_url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Server returned invalid JSON for {path}.") from exc

    def chat(self, messages: list[ChatMessage]) -> str:
        payload = build_chat_payload(self.config, messages)
        data = self._request("POST", "/chat/completions", payload=payload)
        return extract_assistant_text(data)


class ChatSession:
    """Conversation state and command handling."""

    def __init__(self, client: OpenAICompatClient):
        self.client = client
        self.messages: list[ChatMessage] = []
        self.status = (
            "Connected to "
            f"{self.client.config.base_url} using model {self.client.config.model}. "
            "Use /test path/to/file.py, /clear, or /quit."
        )
        self.reset()

    def reset(self) -> None:
        self.messages = [
            ChatMessage(role="system", content=self.client.config.system_prompt),
            ChatMessage(
                role="assistant",
                content=(
                    "OpenAI AI Tester is ready. Type a message to chat, or run "
                    "/test path/to/file.py to inspect Python code."
                ),
            ),
        ]
        self.status = "Conversation cleared."

    def process_input(self, raw_text: str) -> str | None:
        text = raw_text.strip()
        if not text:
            return None

        if text in {"/quit", "/exit"}:
            return "__QUIT__"

        if text == "/clear":
            self.reset()
            return None

        if text.startswith("/test "):
            _, _, path_text = text.partition(" ")
            path, source = load_python_file(path_text)
            prompt = build_python_test_prompt(path, source)
            return self._send(prompt, label=f"Testing {path}")

        return self._send(text)

    def _send(self, text: str, label: str | None = None) -> str:
        user_text = label or text
        self.messages.append(ChatMessage(role="user", content=user_text))
        assistant_text = self.client.chat(self.messages[:-1] + [ChatMessage(role="user", content=text)])
        self.messages.append(ChatMessage(role="assistant", content=assistant_text))
        self.status = "Last request completed."
        return assistant_text


class ChatTUI:
    """Minimal ChatGPT-style curses UI."""

    def __init__(self, stdscr: Any, session: ChatSession):
        self.stdscr = stdscr
        self.session = session
        self.input_buffer = ""
        self.scroll_offset = 0

    def run(self) -> None:
        curses.curs_set(1)
        self.stdscr.keypad(True)
        self._init_colors()

        while True:
            self.render()
            key = self.stdscr.get_wch()
            if key == curses.KEY_RESIZE:
                continue
            if key in ("\n", "\r") or key == curses.KEY_ENTER:
                self._submit_input()
                continue
            if key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                self.input_buffer = self.input_buffer[:-1]
                continue
            if key == curses.KEY_PPAGE:
                self.scroll_offset += 8
                continue
            if key == curses.KEY_NPAGE:
                self.scroll_offset = max(0, self.scroll_offset - 8)
                continue
            if key == "\x03":
                break
            if isinstance(key, str) and key.isprintable():
                self.input_buffer += key

    def _submit_input(self) -> None:
        pending = self.input_buffer
        self.input_buffer = ""
        try:
            result = self.session.process_input(pending)
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            self.session.status = str(exc)
            return

        self.scroll_offset = 0
        if result == "__QUIT__":
            raise SystemExit(0)

    def _init_colors(self) -> None:
        if not curses.has_colors():
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)

    def render(self) -> None:
        self.stdscr.erase()
        max_y, max_x = self.stdscr.getmaxyx()
        title = f" OpenAI AI Tester | {self.session.client.config.base_url} | {self.session.client.config.model} "
        self._add_line(0, 0, title[: max_x - 1], curses.color_pair(1) | curses.A_BOLD)

        transcript_height = max(3, max_y - 4)
        lines = self._format_messages(max_x)
        visible_end = max(0, len(lines) - self.scroll_offset)
        visible_start = max(0, visible_end - transcript_height)
        visible_lines = lines[visible_start:visible_end]

        for index, (text, attr) in enumerate(visible_lines, start=1):
            if index >= max_y - 2:
                break
            self._add_line(index, 0, text[: max_x - 1], attr)

        help_line = " Enter=send  PgUp/PgDn=scroll  /test file.py  /clear  /quit "
        self._add_line(max_y - 2, 0, help_line[: max_x - 1], curses.color_pair(3))

        prompt = f"> {self.input_buffer}"
        self._add_line(max_y - 1, 0, prompt[: max_x - 1], curses.color_pair(2))
        cursor_x = min(len(prompt), max_x - 1)
        self.stdscr.move(max_y - 1, cursor_x)
        self.stdscr.refresh()

    def _format_messages(self, width: int) -> list[tuple[str, int]]:
        wrapped: list[tuple[str, int]] = []
        content_width = max(20, width - 2)

        for message in self.session.messages[1:]:
            role_label = message.role.capitalize()
            attr = 0
            if message.role == "user":
                attr = curses.color_pair(2) | curses.A_BOLD
            elif message.role == "assistant":
                attr = curses.color_pair(1)

            body_lines = textwrap.wrap(
                message.content,
                width=max(10, content_width - len(role_label) - 2),
                replace_whitespace=False,
                drop_whitespace=False,
            ) or [""]

            wrapped.append((f"{role_label}: {body_lines[0]}", attr))
            for body_line in body_lines[1:]:
                wrapped.append((f"{' ' * (len(role_label) + 2)}{body_line}", attr))
            wrapped.append(("", 0))

        if not wrapped:
            wrapped.append((self.session.status, 0))
        return wrapped

    def _add_line(self, y: int, x: int, text: str, attr: int = 0) -> None:
        try:
            self.stdscr.addnstr(y, x, text, max(0, self.stdscr.getmaxyx()[1] - x - 1), attr)
        except curses.error:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curses OpenAI-compatible tester for localhost:8080")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="OpenAI-compatible base URL. Defaults to http://localhost:8080/v1",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to send in requests.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("python_files", nargs="*", help="Optional .py files to inspect on startup.")
    return parser.parse_args()


def run_app(config: TesterConfig, python_files: list[str] | None = None) -> None:
    client = OpenAICompatClient(config)
    session = ChatSession(client)

    for python_file in python_files or []:
        session.process_input(f"/test {python_file}")

    curses.wrapper(lambda stdscr: ChatTUI(stdscr, session).run())


def main() -> None:
    args = parse_args()
    config = TesterConfig(
        base_url=args.endpoint,
        model=args.model,
        timeout=args.timeout,
    )
    run_app(config, python_files=args.python_files)


if __name__ == "__main__":
    main()
