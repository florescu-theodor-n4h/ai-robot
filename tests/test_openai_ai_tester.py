"""Tests for the OpenAI AI tester script."""

import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai_ai_tester import (  # noqa: E402
    ChatMessage,
    ChatSession,
    TesterConfig,
    build_chat_payload,
    build_python_test_prompt,
    extract_assistant_text,
    load_python_file,
    normalize_base_url,
)


class FakeClient:
    def __init__(self):
        self.config = TesterConfig()
        self.requests = []

    def chat(self, messages):
        self.requests.append(messages)
        return "assistant reply"


def test_normalize_base_url_appends_v1():
    assert normalize_base_url("http://localhost:8080") == "http://localhost:8080/v1"


def test_normalize_base_url_preserves_existing_v1():
    assert normalize_base_url("http://localhost:8080/v1/") == "http://localhost:8080/v1"


def test_build_chat_payload_uses_messages():
    config = TesterConfig(model="demo-model", temperature=0.3, max_tokens=42)
    payload = build_chat_payload(
        config,
        [ChatMessage(role="user", content="hello"), ChatMessage(role="assistant", content="hi")],
    )

    assert payload["model"] == "demo-model"
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 42
    assert payload["messages"][0]["content"] == "hello"


def test_extract_assistant_text_supports_multimodal_content():
    data = {
        "choices": [
            {
                "message": {
                    "content": [{"type": "text", "text": "line one"}, {"text": "line two"}]
                }
            }
        ]
    }

    assert extract_assistant_text(data) == "line one\nline two"


def test_build_python_test_prompt_includes_filename_and_source():
    prompt = build_python_test_prompt(Path("demo.py"), "print('hi')")
    assert "demo.py" in prompt
    assert "```python" in prompt
    assert "print('hi')" in prompt


def test_load_python_file_requires_py_suffix(tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError):
        load_python_file(str(other))


def test_chat_session_sends_plain_message():
    client = FakeClient()
    session = ChatSession(client)

    reply = session.process_input("hello model")

    assert reply == "assistant reply"
    assert session.messages[-2].role == "user"
    assert session.messages[-2].content == "hello model"
    assert session.messages[-1].role == "assistant"
    assert client.requests[-1][-1].content == "hello model"


def test_chat_session_tests_python_file(tmp_path):
    client = FakeClient()
    session = ChatSession(client)
    script = tmp_path / "sample.py"
    script.write_text("value = 1\nprint(value)\n", encoding="utf-8")

    reply = session.process_input(f"/test {script}")

    assert reply == "assistant reply"
    assert session.messages[-2].content == f"Testing {script}"
    sent_prompt = client.requests[-1][-1].content
    assert "sample.py" in sent_prompt
    assert "print(value)" in sent_prompt


def test_chat_session_clear_resets_history():
    client = FakeClient()
    session = ChatSession(client)
    session.process_input("hello model")

    session.process_input("/clear")

    assert len(session.messages) == 2
    assert session.messages[0].role == "system"
    assert session.messages[1].role == "assistant"
