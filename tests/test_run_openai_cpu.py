"""Tests for the CPU OpenAI server launcher."""

import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_openai_cpu  # noqa: E402


def test_discover_default_adapter_skips_inaccessible_saved_path(tmp_path, monkeypatch):
    accessible_adapter = tmp_path / "runs_cpu" / "adapters"
    accessible_adapter.mkdir(parents=True)
    (accessible_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (accessible_adapter / "adapter_model.bin").write_bytes(b"")

    monkeypatch.setattr(run_openai_cpu, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(run_openai_cpu, "RUN_DIR", tmp_path / "runs_cpu")

    inaccessible_path = "/root/AI/runs_cpu/adapters"

    def fake_read_json(path, default=None):
        if path.name == "current_state.json":
            return {"adapter_dir": inaccessible_path}
        return {}

    original_is_dir = Path.is_dir

    def fake_is_dir(self):
        if str(self) == inaccessible_path:
            raise PermissionError(13, "Permission denied", inaccessible_path)
        return original_is_dir(self)

    monkeypatch.setattr(run_openai_cpu, "read_json", fake_read_json)
    monkeypatch.setattr(Path, "is_dir", fake_is_dir)

    assert run_openai_cpu.discover_default_adapter() == accessible_adapter.resolve()
