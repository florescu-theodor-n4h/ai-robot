#!/bin/bash ./runVenv-qlora.sh
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os

# ============================================================
# CPU-ONLY ENV HARDENING — before torch/transformers imports
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import curses
import errno
import importlib.util
import json
import multiprocessing as mp
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# SIGNAL HARDENING
# ============================================================

def harden_signals():
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.default_int_handler)


harden_signals()


# ============================================================
# PATHS / DEFAULTS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parent
RUN_DIR = PROJECT_DIR / "runs_cpu"
RUN_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_STATE_FILE = RUN_DIR / "current_state.json"
TRAIN_CONFIG_FILE = RUN_DIR / "current_config.json"

STATE_FILE = RUN_DIR / "openai_server_state.json"
LOG_FILE = RUN_DIR / "openai_server_log.txt"
PID_FILE = RUN_DIR / "openai_server.pid"
CONFIG_FILE = RUN_DIR / "openai_server_config.json"

DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_MODEL_NAME = "tinyllama-qlora-cpu"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_MESSAGES = 8
DEFAULT_PROMPT_MAX_CHARS = 6000
DEFAULT_SYSTEM_PROMPT = (
    "You are a recruiting decision-support assistant. "
    "Compare a CV against a job description using only job-relevant skills, "
    "experience, responsibilities, and tools. "
    "Return concise, valid JSON when the user asks for a fit assessment."
)


# ============================================================
# DEPENDENCY CHECK
# ============================================================

REQUIRED_IMPORTS = {
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "uvicorn": "uvicorn",
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
}


def missing_dependencies():
    missing = []
    for import_name, pip_name in REQUIRED_IMPORTS.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)
    return missing


# ============================================================
# SAFE FILE HELPERS
# ============================================================

def atomic_write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default=None):
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def append_log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def is_pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def current_worker_alive() -> bool:
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    return is_pid_alive(pid)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_elapsed(seconds):
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def tail_lines(path: Path, limit=8):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[-limit:]
    except Exception:
        return []


def run_command(command: list[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"Command not found: {command[0]}"
    except Exception as exc:
        return 1, "", str(exc)


def port_bind_error(host: str, port: int) -> str | None:
    try:
        addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    except socket.gaierror as exc:
        return str(exc)

    last_error = None
    for family, socktype, proto, _, sockaddr in addrinfos:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            return None
        except OSError as exc:
            last_error = exc
            if exc.errno == errno.EADDRINUSE:
                return str(exc)
        finally:
            sock.close()

    return str(last_error) if last_error else None


def find_listener_on_port(port: int) -> dict | None:
    code, stdout, stderr = run_command(["ss", "-H", "-ltnp"])
    if code != 0:
        return {"port": int(port), "error": stderr or stdout or "Unable to inspect listeners with ss"}

    for line in stdout.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 5:
            continue
        local_address = parts[3]
        if not local_address.endswith(f":{port}"):
            continue

        process_info = parts[5] if len(parts) > 5 else ""
        pid_match = re.search(r"pid=(\d+)", process_info)
        name_match = re.search(r'"([^"]+)"', process_info)
        pid = int(pid_match.group(1)) if pid_match else None
        return {
            "port": int(port),
            "local_address": local_address,
            "pid": pid,
            "process_name": name_match.group(1) if name_match else None,
            "process_info": process_info or "-",
        }

    return None


def process_command_line(pid: int | None) -> str:
    if not pid:
        return "-"
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").strip()
        if raw:
            return raw.decode("utf-8", errors="replace")
    except Exception:
        pass
    try:
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
    except Exception:
        return "-"


def find_systemd_service_for_pid(pid: int | None) -> str | None:
    if not pid:
        return None
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines():
            matches = re.findall(r"([A-Za-z0-9_.@-]+\.service)\b", line)
            if matches:
                return matches[-1]
    except Exception:
        pass
    return None


def wait_for_pid_exit(pid: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not is_pid_alive(pid):
            return True
        time.sleep(0.2)
    return not is_pid_alive(pid)


def stop_systemd_service(service_name: str) -> tuple[bool, str]:
    code, stdout, stderr = run_command(["systemctl", "stop", service_name])
    message = stdout or stderr or f"systemctl stop {service_name} returned {code}"
    return code == 0, message


def systemd_service_status(service_name: str) -> list[str]:
    code, stdout, stderr = run_command(["systemctl", "--no-pager", "--lines=14", "status", service_name])
    text = stdout or stderr or f"systemctl status {service_name} returned {code}"
    return text.splitlines() or [text]


def kill_process_tree_root(pid: int) -> tuple[bool, str]:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True, f"Process {pid} is already gone."
    except PermissionError as exc:
        return False, f"Permission denied while stopping PID {pid}: {exc}"
    except OSError as exc:
        return False, f"Could not signal PID {pid}: {exc}"

    if wait_for_pid_exit(pid, 3.0):
        return True, f"PID {pid} exited after SIGTERM."

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True, f"Process {pid} exited while escalating."
    except PermissionError as exc:
        return False, f"Permission denied while force-killing PID {pid}: {exc}"
    except OSError as exc:
        return False, f"Could not SIGKILL PID {pid}: {exc}"

    if wait_for_pid_exit(pid, 2.0):
        return True, f"PID {pid} was force-killed."
    return False, f"PID {pid} is still alive after SIGKILL."


# ============================================================
# ADAPTER DISCOVERY
# ============================================================

def is_adapter_dir(path: Path) -> bool:
    try:
        return (
            path.is_dir()
            and (path / "adapter_config.json").is_file()
            and any(
                (path / filename).is_file()
                for filename in ("adapter_model.safetensors", "adapter_model.bin")
            )
        )
    except OSError:
        return False


def discover_default_adapter() -> Path | None:
    candidates = []

    def add(pathlike):
        if not pathlike:
            return
        path = Path(pathlike).expanduser()
        if path not in candidates:
            candidates.append(path)

    training_state = read_json(TRAIN_STATE_FILE, {}) or {}
    training_config = read_json(TRAIN_CONFIG_FILE, {}) or {}

    add(training_state.get("adapter_dir"))
    add(training_config.get("adapter_dir"))
    add(RUN_DIR / "adapters")
    add(PROJECT_DIR / "lora_adapter")
    add(PROJECT_DIR / "AI_weight_lora")

    for pattern in ("runs_cpu/outputs/checkpoint-*", "output/checkpoint-*", "lora_out/checkpoint-*"):
        matches = sorted(
            PROJECT_DIR.glob(pattern),
            key=lambda item: item.stat().st_mtime if item.exists() else 0,
            reverse=True,
        )
        for match in matches:
            add(match)

    for candidate in candidates:
        if is_adapter_dir(candidate):
            return candidate.resolve()

    return None


def resolve_base_model(adapter_dir: Path) -> str:
    adapter_config = read_json(adapter_dir / "adapter_config.json", {}) or {}
    base_model = adapter_config.get("base_model_name_or_path")
    if base_model:
        return str(base_model)

    training_config = read_json(TRAIN_CONFIG_FILE, {}) or {}
    if training_config.get("model_id"):
        return str(training_config["model_id"])

    training_state = read_json(TRAIN_STATE_FILE, {}) or {}
    if training_state.get("model_id"):
        return str(training_state["model_id"])

    return DEFAULT_BASE_MODEL


def default_model_name(adapter_dir: Path) -> str:
    name = adapter_dir.name.strip().lower().replace(" ", "-")
    return name or DEFAULT_MODEL_NAME


# ============================================================
# CURSES HELPERS
# ============================================================

def init_colors():
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)
        curses.init_pair(6, curses.COLOR_BLUE, -1)
        curses.init_pair(7, curses.COLOR_WHITE, -1)
    except Exception:
        pass


def color_attr(index: int):
    try:
        return curses.color_pair(index)
    except Exception:
        return 0


def draw_box(stdscr, title: str, footer: str | None = None):
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    stdscr.border()
    stdscr.addstr(0, 2, f" {title} ")
    if footer:
        stdscr.addstr(height - 2, 2, footer[: max(0, width - 4)])
    stdscr.refresh()


def safe_addstr(stdscr, y, x, text, attr=0):
    height, width = stdscr.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    available = max(0, width - x - 1)
    try:
        stdscr.addstr(y, x, str(text)[:available], attr)
    except curses.error:
        pass


def show_message_box(stdscr, title: str, lines: list[str], footer: str = "Press any key to continue"):
    draw_box(stdscr, title, footer)
    for index, line in enumerate(lines):
        safe_addstr(stdscr, 3 + index, 4, line)
    stdscr.refresh()
    stdscr.getch()


def ask_text(stdscr, title, question, default):
    curses.echo()
    draw_box(stdscr, title, "Enter = continue | Q = quit")
    safe_addstr(stdscr, 3, 4, question)
    safe_addstr(stdscr, 5, 4, f"Default: {default}")
    safe_addstr(stdscr, 7, 4, "> ")
    stdscr.refresh()

    value = stdscr.getstr(7, 6, 400).decode("utf-8").strip()
    curses.noecho()

    if value.lower() == "q":
        raise KeyboardInterrupt

    return value or str(default)


def ask_choice(stdscr, title, question, choices, default_index=0):
    selected = default_index

    while True:
        draw_box(stdscr, title, "Arrows/J/K = move | Enter = continue | Q = quit")
        safe_addstr(stdscr, 3, 4, question)

        for index, choice in enumerate(choices):
            marker = ">" if index == selected else " "
            attr = curses.A_REVERSE if index == selected else 0
            safe_addstr(stdscr, 5 + index, 6, f"{marker} {choice}", attr)

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            raise KeyboardInterrupt
        if key in (curses.KEY_UP, ord("k"), ord("K")):
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
            selected = min(len(choices) - 1, selected + 1)
        elif key in (10, 13):
            return selected


def ask_yes_no(stdscr, title, question, default=True):
    selected = 0 if default else 1
    return ask_choice(stdscr, title, question, ["Yes", "No"], selected) == 0


def ask_port_conflict_action(stdscr, host: str, port: int, conflict: dict, service_name: str | None):
    choices = [
        "Find the systemd service responsible",
        "Kill service",
        "Kill offending hijacked process",
        "Exit",
    ]
    selected = 0
    pid = conflict.get("pid")
    process_name = conflict.get("process_name") or "unknown"
    command_line = process_command_line(pid)
    conflict_line = conflict.get("error") or f"Listener: {conflict.get('local_address', '-')}"
    service_line = service_name or "No systemd service detected for that PID."

    while True:
        draw_box(
            stdscr,
            "Port squatter showdown",
            "Arrows/J/K = choose | Enter = act | Q = exit like a sensible grandparent",
        )
        safe_addstr(stdscr, 2, 4, "Well now, hold your horses...", color_attr(3) | curses.A_BOLD)
        safe_addstr(stdscr, 3, 4, f"Port {port} for {host} is already spoken for.", color_attr(4) | curses.A_BOLD)
        safe_addstr(stdscr, 5, 4, "Your polite old machine found this culprit:", color_attr(5) | curses.A_BOLD)
        safe_addstr(stdscr, 6, 6, conflict_line, color_attr(1))
        safe_addstr(stdscr, 7, 6, f"PID: {pid or '-'}")
        safe_addstr(stdscr, 8, 6, f"Process: {process_name}")
        safe_addstr(stdscr, 9, 6, f"Service: {service_line}")
        safe_addstr(stdscr, 10, 6, f"Command: {command_line}")
        safe_addstr(stdscr, 12, 4, "Choose your remedy, youngster:", color_attr(2) | curses.A_BOLD)

        for index, choice in enumerate(choices):
            marker = ">" if index == selected else " "
            attr = curses.A_BOLD
            if index == 0:
                attr |= color_attr(1)
            elif index == 1:
                attr |= color_attr(3)
            elif index == 2:
                attr |= color_attr(4)
            else:
                attr |= color_attr(6)
            if index == selected:
                attr |= curses.A_REVERSE
            safe_addstr(stdscr, 14 + index, 6, f"{marker} {choice}", attr)

        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            raise KeyboardInterrupt
        if key in (curses.KEY_UP, ord("k"), ord("K")):
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
            selected = min(len(choices) - 1, selected + 1)
        elif key in (10, 13):
            return selected


def resolve_port_conflict(stdscr, host: str, port: int):
    while True:
        bind_error = port_bind_error(host, int(port))
        if bind_error is None:
            return

        conflict = find_listener_on_port(int(port)) or {"port": int(port), "error": bind_error}
        pid = conflict.get("pid")
        service_name = find_systemd_service_for_pid(pid)
        action = ask_port_conflict_action(stdscr, host, int(port), conflict, service_name)

        if action == 0:
            if service_name:
                show_message_box(
                    stdscr,
                    f"systemd: {service_name}",
                    systemd_service_status(service_name),
                )
            else:
                show_message_box(
                    stdscr,
                    "No systemd service found",
                    [
                        "I could not trace that listener back to a .service unit.",
                        "It may be a manual process, a user session, or something outside systemd.",
                    ],
                )
            continue

        if action == 1:
            if not service_name:
                show_message_box(
                    stdscr,
                    "No service to stop",
                    [
                        "This listener does not appear to belong to a systemd .service unit.",
                        "Use the process-kill option if you want to evict it anyway.",
                    ],
                )
                continue
            ok, message = stop_systemd_service(service_name)
            show_message_box(
                stdscr,
                "Service stop request",
                [
                    f"Service: {service_name}",
                    message,
                    "I'll re-check the port when you close this box.",
                ],
                "Press any key to re-check the port",
            )
            if not ok:
                continue
            time.sleep(0.5)
            continue

        if action == 2:
            if not pid:
                show_message_box(
                    stdscr,
                    "No PID found",
                    [
                        "I found the bound port, but ss did not reveal a PID for it.",
                        "Try the service lookup or free the port manually.",
                    ],
                )
                continue
            ok, message = kill_process_tree_root(int(pid))
            show_message_box(
                stdscr,
                "Process eviction",
                [
                    f"PID: {pid}",
                    message,
                    "I'll re-check the port when you close this box.",
                ],
                "Press any key to re-check the port",
            )
            if not ok:
                continue
            time.sleep(0.5)
            continue

        raise KeyboardInterrupt


def prompt_for_existing_adapter(stdscr, proposed: str) -> Path:
    current_default = proposed
    while True:
        chosen = Path(ask_text(
            stdscr,
            "QLoRA adapter",
            "Path to the trained adapter directory:",
            current_default,
        )).expanduser()
        if is_adapter_dir(chosen):
            return chosen.resolve()

        draw_box(stdscr, "Adapter not found", "Press any key to try again")
        safe_addstr(stdscr, 3, 4, f"Not a valid adapter directory: {chosen}")
        safe_addstr(stdscr, 5, 4, "Expected files:")
        safe_addstr(stdscr, 6, 6, "- adapter_config.json")
        safe_addstr(stdscr, 7, 6, "- adapter_model.safetensors or adapter_model.bin")
        safe_addstr(stdscr, 9, 4, "Finish a training run first, or point this script at the exported adapter.")
        stdscr.getch()
        current_default = str(chosen)


def wizard(stdscr, args):
    curses.curs_set(0)
    init_colors()

    missing = missing_dependencies()
    if missing:
        draw_box(stdscr, "Missing dependencies", "Press any key to exit")
        safe_addstr(stdscr, 3, 4, "The current Python environment is missing:")
        for index, package in enumerate(missing):
            safe_addstr(stdscr, 5 + index, 6, f"- {package}")
        safe_addstr(stdscr, 7 + len(missing), 4, "Install with:")
        safe_addstr(stdscr, 9 + len(missing), 6, f"{sys.executable} -m pip install {' '.join(missing)}")
        stdscr.getch()
        raise SystemExit(1)

    discovered_adapter = Path(args.adapter_dir).expanduser() if args.adapter_dir else discover_default_adapter()
    default_adapter_path = str(discovered_adapter or (RUN_DIR / "adapters"))
    adapter_dir = prompt_for_existing_adapter(stdscr, default_adapter_path)

    base_model = resolve_base_model(adapter_dir)
    model_name = ask_text(
        stdscr,
        "Model name",
        "OpenAI model id to expose from /v1/models:",
        args.model_name or default_model_name(adapter_dir),
    )

    host = ask_text(
        stdscr,
        "Bind address",
        "Which host should the OpenAI server bind to?",
        args.host,
    )

    port = int(ask_text(
        stdscr,
        "Port",
        "Which TCP port should the OpenAI server use?",
        args.port,
    ))
    resolve_port_conflict(stdscr, host, port)

    thread_choices = [2, 4, 8]
    default_thread_index = 1 if args.omp_threads == 4 else min(
        range(len(thread_choices)),
        key=lambda index: abs(thread_choices[index] - int(args.omp_threads)),
    )
    thread_choice = ask_choice(
        stdscr,
        "CPU threads",
        "How many CPU threads should inference use?",
        [f"{value} threads" for value in thread_choices],
        default_thread_index,
    )
    omp_threads = thread_choices[thread_choice]

    preset_index = ask_choice(
        stdscr,
        "Response preset",
        "Choose a generation preset:",
        [
            "Stable JSON - 128 tokens, temperature 0.0",
            "Balanced - 256 tokens, temperature 0.2",
            "Longer answers - 512 tokens, temperature 0.4",
        ],
        1,
    )
    if preset_index == 0:
        max_new_tokens = 128
        temperature = 0.0
    elif preset_index == 1:
        max_new_tokens = 256
        temperature = 0.2
    else:
        max_new_tokens = 512
        temperature = 0.4

    system_prompt = ask_text(
        stdscr,
        "System prompt",
        "Default system prompt for chat requests:",
        DEFAULT_SYSTEM_PROMPT,
    )

    confirm = ask_yes_no(
        stdscr,
        "Confirm",
        f"Load adapter from {adapter_dir} and start the OpenAI server on {host}:{port}?",
        True,
    )
    if not confirm:
        raise KeyboardInterrupt

    config = {
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "model_name": model_name,
        "host": host,
        "port": int(port),
        "omp_threads": int(omp_threads),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "max_messages": int(DEFAULT_MAX_MESSAGES),
        "prompt_max_chars": int(DEFAULT_PROMPT_MAX_CHARS),
        "system_prompt": system_prompt,
        "started_at": utc_now_iso(),
    }
    atomic_write_json(CONFIG_FILE, config)
    return config


# ============================================================
# SHARED RUNTIME STATE
# ============================================================

class RuntimeState:
    def __init__(self, config: dict):
        self.lock = threading.Lock()
        self.data = {
            "status": "starting",
            "pid": os.getpid(),
            "started_at": config.get("started_at", utc_now_iso()),
            "elapsed_seconds": 0,
            "host": config.get("host"),
            "port": config.get("port"),
            "base_model": config.get("base_model"),
            "adapter_dir": config.get("adapter_dir"),
            "model_name": config.get("model_name"),
            "requests_total": 0,
            "requests_ok": 0,
            "requests_error": 0,
            "last_request_at": None,
            "last_path": None,
            "last_client": None,
            "last_latency_ms": None,
            "last_prompt_chars": None,
            "last_prompt_tokens": None,
            "last_completion_tokens": None,
            "last_response_chars": None,
            "last_error": None,
            "message": "Starting server...",
        }
        self.write()

    def write(self):
        atomic_write_json(STATE_FILE, self.data)

    def update(self, **fields):
        with self.lock:
            self.data.update(fields)
            self.write()

    def heartbeat(self):
        started_at = self.data.get("started_at")
        try:
            started_ts = datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()
        except Exception:
            started_ts = time.time()

        while True:
            with self.lock:
                if self.data.get("status") in ("stopped", "error"):
                    self.write()
                    return
                self.data["elapsed_seconds"] = int(time.time() - started_ts)
                self.write()
            time.sleep(1)

    def record_request(
        self,
        *,
        path: str,
        client: str,
        ok: bool,
        latency_ms: float,
        prompt_chars: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        response_chars: int | None,
        error: str | None = None,
    ):
        with self.lock:
            self.data["requests_total"] += 1
            if ok:
                self.data["requests_ok"] += 1
            else:
                self.data["requests_error"] += 1
            self.data["last_request_at"] = utc_now_iso()
            self.data["last_path"] = path
            self.data["last_client"] = client
            self.data["last_latency_ms"] = round(latency_ms, 1)
            self.data["last_prompt_chars"] = prompt_chars
            self.data["last_prompt_tokens"] = prompt_tokens
            self.data["last_completion_tokens"] = completion_tokens
            self.data["last_response_chars"] = response_chars
            self.data["last_error"] = error
            self.write()


# ============================================================
# INFERENCE WORKER
# ============================================================

def server_worker(config):
    harden_signals()
    server_box = {"server": None}

    try:
        try:
            os.setsid()
        except Exception:
            pass

        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
        LOG_FILE.write_text("", encoding="utf-8")

        state = RuntimeState(config)
        heartbeat_thread = threading.Thread(target=state.heartbeat, daemon=True)
        heartbeat_thread.start()

        adapter_dir = Path(config["adapter_dir"]).expanduser().resolve()
        if not is_adapter_dir(adapter_dir):
            raise FileNotFoundError(f"Adapter directory not found or incomplete: {adapter_dir}")

        append_log("Server worker started.")
        append_log(f"Python: {sys.executable}")
        append_log(f"Adapter: {adapter_dir}")
        append_log(f"Base model: {config['base_model']}")

        state.update(status="loading_imports", message="Loading FastAPI and transformers...")

        import torch
        import uvicorn
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(int(config["omp_threads"]))
        if torch.cuda.is_available():
            raise RuntimeError("CUDA is visible. Refusing to run CPU-only OpenAI server.")

        state.update(status="loading_tokenizer", message="Loading tokenizer from adapter/base model...")
        tokenizer_source = str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else config["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        state.update(status="loading_model", message="Loading base model on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=False,
        )
        model.to("cpu")
        model.eval()

        state.update(status="loading_adapter", message="Applying trained QLoRA adapter...")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        model.to("cpu")
        model.eval()

        infer_lock = threading.Lock()

        def extract_text(content: Any) -> str:
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(str(item.get("text", "")))
                        elif "text" in item:
                            parts.append(str(item["text"]))
                return "\n".join(part for part in parts if part)
            if isinstance(content, dict):
                if "text" in content:
                    return str(content["text"])
            return str(content)

        def normalize_stop(stop_value):
            if stop_value is None:
                return []
            if isinstance(stop_value, str):
                return [stop_value]
            if isinstance(stop_value, list):
                return [str(item) for item in stop_value if item]
            return [str(stop_value)]

        def apply_stop_sequences(text: str, stop_sequences: list[str]) -> str:
            clipped = text
            for stop_sequence in stop_sequences:
                if stop_sequence and stop_sequence in clipped:
                    clipped = clipped.split(stop_sequence, 1)[0]
            return clipped.strip()

        def build_chat_prompt(messages: list[dict]) -> str:
            if not messages:
                raise HTTPException(status_code=400, detail="messages must not be empty")

            if len(messages) > int(config["max_messages"]):
                messages = messages[-int(config["max_messages"]):]

            system_prompt = config["system_prompt"]
            transcript = []
            for message in messages:
                role = str(message.get("role", "")).strip().lower()
                content = extract_text(message.get("content")).strip()
                if not content:
                    continue
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    transcript.append(f"User:\n{content}")
                elif role == "assistant":
                    transcript.append(f"Assistant:\n{content}")
                else:
                    transcript.append(f"{role.title() or 'User'}:\n{content}")

            if not transcript:
                raise HTTPException(status_code=400, detail="messages must include text content")

            instruction = system_prompt.strip()
            body = "\n\n".join(transcript)
            prompt = f"<s>[INST]\n{instruction}\n\n{body}\n[/INST]\n"
            if len(prompt) > int(config["prompt_max_chars"]):
                prompt = prompt[-int(config["prompt_max_chars"]):]
            return prompt

        def build_completion_prompt(prompt_value: Any) -> str:
            if isinstance(prompt_value, list):
                if not prompt_value:
                    raise HTTPException(status_code=400, detail="prompt list must not be empty")
                prompt_value = prompt_value[0]
            text = extract_text(prompt_value).strip()
            if not text:
                raise HTTPException(status_code=400, detail="prompt must not be empty")
            if len(text) > int(config["prompt_max_chars"]):
                text = text[-int(config["prompt_max_chars"]):]
            return text

        def generate_text(prompt: str, max_new_tokens: int, temperature: float, stop_sequences: list[str]):
            with infer_lock, torch.inference_mode():
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to("cpu")
                attention_mask = inputs["attention_mask"].to("cpu")
                prompt_tokens = int(input_ids.shape[-1])
                do_sample = float(temperature) > 0.0
                generate_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": int(max_new_tokens),
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    generate_kwargs["temperature"] = max(float(temperature), 1e-5)

                outputs = model.generate(**generate_kwargs)
                new_tokens = outputs[0][prompt_tokens:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                text = apply_stop_sequences(text, stop_sequences)
                completion_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                return text, prompt_tokens, completion_tokens

        state.update(status="building_api", message="Creating OpenAI-compatible API...")
        app = FastAPI(title="CPU QLoRA OpenAI Server")

        @app.middleware("http")
        async def capture_metrics(request: Request, call_next):
            start_time = time.time()
            client = request.client.host if request.client else "-"
            response = None
            error_message = None
            prompt_chars = None
            prompt_tokens = None
            completion_tokens = None
            response_chars = None

            try:
                response = await call_next(request)
                return response
            except Exception as exc:
                error_message = str(exc)
                append_log(f"Unhandled request error on {request.url.path}: {exc}")
                append_log(traceback.format_exc())
                response = JSONResponse(
                    status_code=500,
                    content={"error": {"message": str(exc), "type": "server_error"}},
                )
                return response
            finally:
                request_state = getattr(request.state, "inference_stats", None)
                if request_state:
                    prompt_chars = request_state.get("prompt_chars")
                    prompt_tokens = request_state.get("prompt_tokens")
                    completion_tokens = request_state.get("completion_tokens")
                    response_chars = request_state.get("response_chars")
                    error_message = request_state.get("error") or error_message

                latency_ms = (time.time() - start_time) * 1000.0
                state.record_request(
                    path=request.url.path,
                    client=client,
                    ok=response is not None and response.status_code < 400,
                    latency_ms=latency_ms,
                    prompt_chars=prompt_chars,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    response_chars=response_chars,
                    error=error_message,
                )

        @app.on_event("startup")
        async def on_startup():
            append_log(f"Serving OpenAI protocol on http://{config['host']}:{config['port']}")
            state.update(
                status="serving",
                message="Server is ready for OpenAI-compatible requests.",
            )

        @app.on_event("shutdown")
        async def on_shutdown():
            append_log("Server shutdown requested.")
            state.update(status="stopped", message="Server stopped.")

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "model": config["model_name"],
                "adapter_dir": config["adapter_dir"],
                "base_model": config["base_model"],
                "device": "cpu",
            }

        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": config["model_name"],
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    }
                ],
            }

        def validate_requested_model(requested_model: Any):
            if requested_model and str(requested_model) != str(config["model_name"]):
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown model '{requested_model}'. Available model: {config['model_name']}",
                )

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            validate_requested_model(body.get("model"))
            if body.get("stream"):
                raise HTTPException(status_code=400, detail="stream=true is not supported by this local server")

            messages = body.get("messages")
            if not isinstance(messages, list):
                raise HTTPException(status_code=400, detail="messages must be a list")

            max_tokens = int(body.get("max_tokens", config["max_new_tokens"]))
            temperature = float(body.get("temperature", config["temperature"]))
            stop_sequences = normalize_stop(body.get("stop"))

            prompt = build_chat_prompt(messages)
            prompt_chars = len(prompt)
            start_time = time.time()
            try:
                text, prompt_tokens, completion_tokens = generate_text(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                )
            except HTTPException:
                raise
            except Exception as exc:
                append_log(f"Generation failed on /v1/chat/completions: {exc}")
                append_log(traceback.format_exc())
                request.state.inference_stats = {
                    "prompt_chars": prompt_chars,
                    "error": str(exc),
                }
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            latency_ms = round((time.time() - start_time) * 1000.0, 1)
            append_log(
                "chat completion "
                f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} latency_ms={latency_ms}"
            )
            request.state.inference_stats = {
                "prompt_chars": prompt_chars,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "response_chars": len(text),
            }
            created = int(time.time())
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": created,
                "model": config["model_name"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

        @app.post("/v1/completions")
        async def completions(request: Request):
            body = await request.json()
            validate_requested_model(body.get("model"))
            if body.get("stream"):
                raise HTTPException(status_code=400, detail="stream=true is not supported by this local server")

            max_tokens = int(body.get("max_tokens", config["max_new_tokens"]))
            temperature = float(body.get("temperature", config["temperature"]))
            stop_sequences = normalize_stop(body.get("stop"))

            prompt = build_completion_prompt(body.get("prompt"))
            prompt_chars = len(prompt)
            start_time = time.time()
            try:
                text, prompt_tokens, completion_tokens = generate_text(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                )
            except HTTPException:
                raise
            except Exception as exc:
                append_log(f"Generation failed on /v1/completions: {exc}")
                append_log(traceback.format_exc())
                request.state.inference_stats = {
                    "prompt_chars": prompt_chars,
                    "error": str(exc),
                }
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            latency_ms = round((time.time() - start_time) * 1000.0, 1)
            append_log(
                "completion "
                f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} latency_ms={latency_ms}"
            )
            request.state.inference_stats = {
                "prompt_chars": prompt_chars,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "response_chars": len(text),
            }
            created = int(time.time())
            return {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": created,
                "model": config["model_name"],
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

        uvicorn_config = uvicorn.Config(
            app,
            host=config["host"],
            port=int(config["port"]),
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(uvicorn_config)
        server_box["server"] = server

        def shutdown_handler(sig, frame):
            append_log(f"Received signal {sig}. Shutting down server.")
            state.update(status="stopping", message="Shutdown requested.")
            if server_box["server"] is not None:
                server_box["server"].should_exit = True

        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)

        state.update(status="starting_server", message="Booting OpenAI-compatible server...")
        server.run()

        final_status = "stopped"
        final_message = "Server stopped."
        if state.data.get("status") == "error":
            final_status = "error"
            final_message = state.data.get("message", "Server failed.")
        state.update(status=final_status, message=final_message)

    except Exception as exc:
        append_log("ERROR:")
        append_log(traceback.format_exc())
        current = read_json(STATE_FILE, {}) or {}
        current.update({
            "status": "error",
            "pid": os.getpid(),
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "finished_at": utc_now_iso(),
        })
        atomic_write_json(STATE_FILE, current)


# ============================================================
# MONITOR / PROCESS CONTROL
# ============================================================

def status_attr(status: str):
    normalized = (status or "").lower()
    if normalized in ("serving",):
        return color_attr(2) | curses.A_BOLD
    if normalized in ("starting", "loading_imports", "loading_tokenizer", "loading_model", "loading_adapter", "building_api", "starting_server", "stopping"):
        return color_attr(3) | curses.A_BOLD
    if normalized in ("error",):
        return color_attr(4) | curses.A_BOLD
    return color_attr(1) | curses.A_BOLD


def stop_current_worker():
    state = read_json(STATE_FILE, {}) or {}
    pid = state.get("pid")
    if not pid:
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except Exception:
            pid = None
    if not pid:
        return False
    if not is_pid_alive(int(pid)):
        return False

    append_log(f"Stop requested for PID {pid}.")
    os.kill(int(pid), signal.SIGTERM)
    return True


def draw_monitor(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    init_colors()

    while True:
        state = read_json(STATE_FILE, {}) or {}
        config = read_json(CONFIG_FILE, {}) or {}

        status = state.get("status", "unknown")
        pid = state.get("pid")
        alive = is_pid_alive(pid) if pid else False

        draw_box(
            stdscr,
            "CPU QLoRA OpenAI Server — Live Dashboard",
            "D = detach | S = stop server | Q = quit monitor | R = refresh",
        )

        safe_addstr(stdscr, 2, 4, "Server", color_attr(1) | curses.A_BOLD)
        safe_addstr(stdscr, 3, 6, f"Status: {status}", status_attr(status))
        safe_addstr(stdscr, 4, 6, f"PID: {pid or '-'}")
        safe_addstr(stdscr, 5, 6, f"Alive: {'yes' if alive else 'no'}")
        safe_addstr(stdscr, 6, 6, f"Endpoint: http://{state.get('host') or config.get('host', DEFAULT_HOST)}:{state.get('port') or config.get('port', DEFAULT_PORT)}")
        safe_addstr(stdscr, 7, 6, f"Model id: {state.get('model_name') or config.get('model_name', '-')}")
        safe_addstr(stdscr, 8, 6, f"Elapsed: {format_elapsed(state.get('elapsed_seconds'))}")

        safe_addstr(stdscr, 10, 4, "Model", color_attr(1) | curses.A_BOLD)
        safe_addstr(stdscr, 11, 6, f"Base: {state.get('base_model') or config.get('base_model', '-')}")
        safe_addstr(stdscr, 12, 6, f"Adapter: {state.get('adapter_dir') or config.get('adapter_dir', '-')}")
        safe_addstr(stdscr, 13, 6, f"Threads: {config.get('omp_threads', '-')}")
        safe_addstr(stdscr, 14, 6, f"Default max tokens: {config.get('max_new_tokens', '-')}")
        safe_addstr(stdscr, 15, 6, f"Default temperature: {config.get('temperature', '-')}")

        safe_addstr(stdscr, 17, 4, "Traffic", color_attr(1) | curses.A_BOLD)
        safe_addstr(stdscr, 18, 6, f"Requests total: {state.get('requests_total', 0)}")
        safe_addstr(stdscr, 19, 6, f"Successful: {state.get('requests_ok', 0)}")
        safe_addstr(stdscr, 20, 6, f"Errors: {state.get('requests_error', 0)}")
        safe_addstr(stdscr, 21, 6, f"Last path: {state.get('last_path') or '-'}")
        safe_addstr(stdscr, 22, 6, f"Last client: {state.get('last_client') or '-'}")
        safe_addstr(stdscr, 23, 6, f"Last latency: {state.get('last_latency_ms') or '-'} ms")
        safe_addstr(stdscr, 24, 6, f"Prompt tokens: {state.get('last_prompt_tokens') or '-'}")
        safe_addstr(stdscr, 25, 6, f"Completion tokens: {state.get('last_completion_tokens') or '-'}")

        safe_addstr(stdscr, 27, 4, "Status message", color_attr(1) | curses.A_BOLD)
        safe_addstr(stdscr, 28, 6, state.get("message", "-"))
        safe_addstr(stdscr, 29, 6, f"Last error: {state.get('last_error') or '-'}", color_attr(4) if state.get("last_error") else 0)

        safe_addstr(stdscr, 31, 4, "Recent log lines", color_attr(1) | curses.A_BOLD)
        for index, line in enumerate(tail_lines(LOG_FILE, limit=6)):
            safe_addstr(stdscr, 32 + index, 6, line)

        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord("d"), ord("D")):
            curses.endwin()
            print("\nDetached dashboard. Server should continue running if it is alive.")
            print(f"Reattach with: {SCRIPT_PATH} --attach")
            print(f"Stop with: {SCRIPT_PATH} --stop")
            return

        if key in (ord("s"), ord("S")):
            stop_current_worker()

        if key in (ord("q"), ord("Q")):
            return

        time.sleep(1)


def launch_worker(config):
    if current_worker_alive():
        return False

    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
    process = ctx.Process(target=server_worker, args=(config,))
    process.daemon = False
    process.start()
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    return True


def run_new(stdscr, args):
    config = wizard(stdscr, args)
    launched = launch_worker(config)
    if not launched:
        draw_box(stdscr, "Already running", "Press any key to attach")
        safe_addstr(stdscr, 3, 4, "An OpenAI server worker is already running.")
        stdscr.getch()
    draw_monitor(stdscr)


def run_attach(stdscr):
    if not STATE_FILE.exists():
        draw_box(stdscr, "Attach", "Press any key to exit")
        safe_addstr(stdscr, 3, 4, "No existing OpenAI server state found.")
        safe_addstr(stdscr, 5, 4, "Start a new server first.")
        stdscr.getch()
        return
    draw_monitor(stdscr)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attach", action="store_true", help="Attach to the current server dashboard.")
    parser.add_argument("--status", action="store_true", help="Print current server status and exit.")
    parser.add_argument("--stop", action="store_true", help="Stop the current server and exit.")
    parser.add_argument("--adapter-dir", default="", help="Pre-fill the adapter directory in the wizard.")
    parser.add_argument("--model-name", default="", help="Pre-fill the exposed OpenAI model id.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Default bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Default bind port.")
    parser.add_argument("--omp-threads", type=int, default=4, help="Default CPU thread count shown in the wizard.")
    return parser.parse_args()


def print_status():
    state = read_json(STATE_FILE, {}) or {}
    if not state:
        print("No current OpenAI server state found.")
        return

    pid = state.get("pid")
    alive = is_pid_alive(pid) if pid else False
    payload = dict(state)
    payload["alive"] = alive
    print(json.dumps(payload, indent=2))


def main():
    args = parse_args()

    if args.status:
        print_status()
        return

    if args.stop:
        stopped = stop_current_worker()
        if stopped:
            print("Stop signal sent.")
        else:
            print("No running OpenAI server found.")
        return

    if args.attach:
        curses.wrapper(run_attach)
        return

    curses.wrapper(lambda stdscr: run_new(stdscr, args))


if __name__ == "__main__":
    main()
