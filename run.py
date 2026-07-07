#!/bin/bash ./runVenv-qlora.sh
# -*- coding: utf-8 -*-
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

import sys
import json
import time
import curses
import signal
import argparse
import traceback
import importlib.util
import multiprocessing as mp
from pathlib import Path
from datetime import datetime


# ============================================================
# SIGHUP HARDENING
# ============================================================

def harden_signals():
    # Ignore terminal hangups. This helps training survive SSH disconnects.
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    # Graceful Ctrl+C / termination.
    signal.signal(signal.SIGINT, signal.default_int_handler)


harden_signals()


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parent

RUN_DIR = PROJECT_DIR / "runs_cpu"
RUN_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER_DIR = RUN_DIR / "adapters"
DEFAULT_OUTPUT_DIR = RUN_DIR / "outputs"

STATE_FILE = RUN_DIR / "current_state.json"
LOG_FILE = RUN_DIR / "current_log.txt"
PID_FILE = RUN_DIR / "current_worker.pid"
CONFIG_FILE = RUN_DIR / "current_config.json"


# ============================================================
# DEPENDENCY CHECK
# ============================================================

REQUIRED_IMPORTS = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "peft": "peft",
    "accelerate": "accelerate",
    "sentencepiece": "sentencepiece",
    "huggingface_hub": "huggingface_hub",
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
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


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
        pid = int(PID_FILE.read_text().strip())
        return is_pid_alive(pid)
    except Exception:
        return False


# ============================================================
# TRAINING DATA
# ============================================================

SYSTEM_POLICY = (
    "You are a recruiting decision-support assistant. "
    "Compare the CV to the job description using only job-relevant skills, tools, responsibilities, and experience. "
    "Do not use protected personal attributes such as age, gender, race, religion, nationality, disability, family status, or appearance. "
    "Do not make a final hiring decision. "
    "Return valid JSON with these keys: match, score, reasoning, strengths, gaps, recommendation, review_required. "
    "The recommendation must be one of: advance_to_interview, consider_if_trainable, needs_human_review, do_not_advance_for_this_role. "
    "review_required must always be true."
)


BUILTIN_EXAMPLES = [
    {
        "cv": "Candidate has 5 years of Python backend experience, FastAPI, PostgreSQL, Docker, AWS, REST API design, and production API ownership.",
        "job": "Python Backend Engineer needed for FastAPI services, PostgreSQL, Docker, AWS deployment, and API design.",
        "answer": {
            "match": "strong_match",
            "score": 92,
            "reasoning": "The candidate matches the main backend engineering requirements.",
            "strengths": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
            "gaps": [],
            "recommendation": "advance_to_interview",
            "review_required": True
        }
    },
    {
        "cv": "Candidate is a graphic designer with Figma, branding, visual design, typography, and marketing assets.",
        "job": "Backend Engineer role requiring Python, REST APIs, SQL databases, Docker, and cloud deployment.",
        "answer": {
            "match": "not_match",
            "score": 18,
            "reasoning": "The candidate's experience is design-focused and does not match the backend engineering requirements.",
            "strengths": ["Figma", "Branding", "Visual design"],
            "gaps": ["Python", "REST APIs", "SQL", "Docker", "Cloud deployment"],
            "recommendation": "do_not_advance_for_this_role",
            "review_required": True
        }
    },
    {
        "cv": "Candidate has 3 years of JavaScript, React, Node.js, REST APIs, MongoDB, Git, and some Python scripting.",
        "job": "Full-stack developer needed for React frontend, Node.js services, REST APIs, and database work.",
        "answer": {
            "match": "strong_match",
            "score": 86,
            "reasoning": "The candidate fits the full-stack role well.",
            "strengths": ["React", "Node.js", "REST APIs", "MongoDB"],
            "gaps": [],
            "recommendation": "advance_to_interview",
            "review_required": True
        }
    },
    {
        "cv": "Candidate has customer support experience, CRM tools, ticket handling, communication, and basic Excel.",
        "job": "Junior Data Analyst requiring Excel, SQL basics, dashboards, reporting, and business communication.",
        "answer": {
            "match": "possible_match",
            "score": 54,
            "reasoning": "The candidate has communication and Excel exposure, but lacks SQL and dashboard experience.",
            "strengths": ["Communication", "CRM", "Excel basics"],
            "gaps": ["SQL", "Dashboards", "Data reporting"],
            "recommendation": "consider_if_trainable",
            "review_required": True
        }
    },
    {
        "cv": "Candidate has 6 years in machine learning, Python, PyTorch, scikit-learn, NLP, model evaluation, and MLOps pipelines.",
        "job": "AI Engineer needed for Python, PyTorch, NLP models, evaluation, deployment, and MLOps.",
        "answer": {
            "match": "strong_match",
            "score": 95,
            "reasoning": "The candidate strongly matches the AI engineering requirements.",
            "strengths": ["Python", "PyTorch", "NLP", "Model evaluation", "MLOps"],
            "gaps": [],
            "recommendation": "advance_to_interview",
            "review_required": True
        }
    },
    {
        "cv": "Candidate is a project manager with Agile, stakeholder communication, timelines, risk management, sprint planning, and Jira.",
        "job": "DevOps Engineer requiring Linux, CI/CD, Docker, Kubernetes, Terraform, and cloud infrastructure.",
        "answer": {
            "match": "not_match",
            "score": 24,
            "reasoning": "The candidate has project coordination strengths but lacks required DevOps technical skills.",
            "strengths": ["Agile", "Stakeholder communication", "Jira"],
            "gaps": ["Linux", "CI/CD", "Docker", "Kubernetes", "Terraform"],
            "recommendation": "do_not_advance_for_this_role",
            "review_required": True
        }
    },
]


def build_prompt(cv: str, job: str) -> str:
    return (
        "<s>[INST]\n"
        f"{SYSTEM_POLICY}\n\n"
        f"CV:\n{cv.strip()}\n\n"
        f"JOB:\n{job.strip()}\n"
        "[/INST]\n"
    )


def build_training_text(item: dict) -> str:
    return build_prompt(item["cv"], item["job"]) + json.dumps(item["answer"], ensure_ascii=False) + "</s>"


def load_external_jsonl(path: str):
    p = Path(path).expanduser()
    rows = []
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not all(k in obj for k in ("cv", "job", "answer")):
                raise ValueError(f"Bad JSONL row {line_number}. Required keys: cv, job, answer")
            rows.append(obj)

    if not rows:
        raise ValueError("Dataset is empty.")

    return rows


# ============================================================
# TUI INPUT HELPERS
# ============================================================

def draw_box(stdscr, title: str):
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    stdscr.border()
    stdscr.addstr(0, 2, f" {title} ")
    stdscr.addstr(h - 2, 2, "Enter = continue | Q = quit | D = detach while training")
    stdscr.refresh()


def safe_addstr(stdscr, y, x, text, attr=0):
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h:
        return
    available = max(0, w - x - 1)
    stdscr.addstr(y, x, str(text)[:available], attr)


def ask_text(stdscr, title, question, default):
    curses.echo()
    draw_box(stdscr, title)
    safe_addstr(stdscr, 3, 4, question)
    safe_addstr(stdscr, 5, 4, f"Default: {default}")
    safe_addstr(stdscr, 7, 4, "> ")
    stdscr.refresh()

    value = stdscr.getstr(7, 6, 300).decode("utf-8").strip()
    curses.noecho()

    if value.lower() == "q":
        raise KeyboardInterrupt

    return value or default


def ask_choice(stdscr, title, question, choices, default_index=0):
    selected = default_index

    while True:
        draw_box(stdscr, title)
        safe_addstr(stdscr, 3, 4, question)

        for i, choice in enumerate(choices):
            marker = ">" if i == selected else " "
            attr = curses.A_REVERSE if i == selected else 0
            safe_addstr(stdscr, 5 + i, 6, f"{marker} {choice}", attr)

        key = stdscr.getch()

        if key in (ord("q"), ord("Q")):
            raise KeyboardInterrupt
        if key in (curses.KEY_UP, ord("k")):
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = min(len(choices) - 1, selected + 1)
        elif key in (10, 13):
            return selected


def ask_yes_no(stdscr, title, question, default=True):
    yes_index = 0 if default else 1
    choice = ask_choice(stdscr, title, question, ["Yes", "No"], yes_index)
    return choice == 0


def wizard(stdscr):
    curses.curs_set(0)

    missing = missing_dependencies()
    if missing:
        draw_box(stdscr, "Missing dependencies")
        safe_addstr(stdscr, 3, 4, "The venv is missing required packages:")
        for i, package in enumerate(missing):
            safe_addstr(stdscr, 5 + i, 6, f"- {package}")
        safe_addstr(stdscr, 8 + len(missing), 4, "Install with:")
        safe_addstr(stdscr, 10 + len(missing), 6, f"{sys.executable} -m pip install {' '.join(missing)}")
        safe_addstr(stdscr, 13 + len(missing), 4, "Press any key to exit.")
        stdscr.getch()
        raise SystemExit(1)

    model_id = ask_text(
        stdscr,
        "Model",
        "Which base model should be trained?",
        DEFAULT_MODEL_ID
    )

    source_choice = ask_choice(
        stdscr,
        "Training data",
        "What do you want to train on?",
        [
            "Built-in safe recruiting examples",
            "My own JSONL file with cv/job/answer rows",
            "Built-in examples + my JSONL file",
        ],
        0
    )

    dataset_path = ""
    if source_choice in (1, 2):
        dataset_path = ask_text(
            stdscr,
            "Training data file",
            "Path to your JSONL dataset:",
            str(PROJECT_DIR / "recruiting_data.jsonl")
        )

    size_choice = ask_choice(
        stdscr,
        "Training strength",
        "How hard should CPU training run?",
        [
            "Tiny test run — fastest sanity check",
            "Normal CPU run — safer default",
            "Stronger CPU run — slower, more training",
        ],
        1
    )

    if size_choice == 0:
        epochs = 1
        max_length = 256
        grad_accum = 1
        learning_rate = 2e-4
    elif size_choice == 1:
        epochs = 3
        max_length = 384
        grad_accum = 2
        learning_rate = 2e-4
    else:
        epochs = 5
        max_length = 512
        grad_accum = 4
        learning_rate = 1.5e-4

    adapter_dir = ask_text(
        stdscr,
        "Output",
        "Where should the trained LoRA adapter be saved?",
        str(DEFAULT_ADAPTER_DIR)
    )

    confirm = ask_yes_no(
        stdscr,
        "Confirm",
        "Start CPU training now?",
        True
    )

    if not confirm:
        raise KeyboardInterrupt

    config = {
        "model_id": model_id,
        "source_choice": source_choice,
        "dataset_path": dataset_path,
        "epochs": epochs,
        "max_length": max_length,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "adapter_dir": adapter_dir,
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    atomic_write_json(CONFIG_FILE, config)
    return config


# ============================================================
# TRAINING CALLBACK
# ============================================================

def training_worker(config):
    harden_signals()

    try:
        # Detach worker from terminal session.
        # This helps it survive terminal/SSH hangups.
        try:
            os.setsid()
        except Exception:
            pass

        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
        LOG_FILE.write_text("", encoding="utf-8")

        atomic_write_json(STATE_FILE, {
            "status": "starting",
            "pid": os.getpid(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "step": 0,
            "epoch": 0,
            "loss": None,
            "learning_rate": None,
            "message": "Loading imports...",
        })

        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            TrainerCallback,
        )

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

        if torch.cuda.is_available():
            raise RuntimeError("CUDA is visible. Refusing to run CPU-only trainer.")

        append_log("Worker started.")
        append_log(f"Python: {sys.executable}")
        append_log(f"PyTorch: {torch.__version__}")
        append_log(f"Model: {config['model_id']}")

        atomic_write_json(STATE_FILE, {
            "status": "loading_model",
            "pid": os.getpid(),
            "started_at": config["started_at"],
            "step": 0,
            "epoch": 0,
            "loss": None,
            "learning_rate": None,
            "message": "Loading tokenizer and model on CPU...",
        })

        tokenizer = AutoTokenizer.from_pretrained(config["model_id"], use_fast=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=False,
        )

        model.to("cpu")
        model.config.use_cache = False

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )

        model = get_peft_model(model, lora_config)

        data = []
        if config["source_choice"] in (0, 2):
            data.extend(BUILTIN_EXAMPLES)

        if config["source_choice"] in (1, 2):
            data.extend(load_external_jsonl(config["dataset_path"]))

        texts = [{"text": build_training_text(item)} for item in data]
        dataset = Dataset.from_list(texts)

        def tokenize(example):
            tokens = tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=int(config["max_length"]),
            )

            labels = tokens["input_ids"].copy()
            labels = [
                token if token != tokenizer.pad_token_id else -100
                for token in labels
            ]

            tokens["labels"] = labels
            return tokens

        dataset = dataset.map(tokenize, remove_columns=["text"])
        dataset.set_format(type="torch")

        total_examples = len(dataset)
        append_log(f"Training examples: {total_examples}")

        class StatsCallback(TrainerCallback):
            def __init__(self):
                self.start = time.time()
                self.last_loss = None
                self.last_lr = None

            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = logs or {}

                if "loss" in logs:
                    self.last_loss = float(logs["loss"])

                if "learning_rate" in logs:
                    self.last_lr = float(logs["learning_rate"])

                elapsed = int(time.time() - self.start)

                atomic_write_json(STATE_FILE, {
                    "status": "training",
                    "pid": os.getpid(),
                    "started_at": config["started_at"],
                    "elapsed_seconds": elapsed,
                    "step": int(state.global_step),
                    "max_steps": int(state.max_steps),
                    "epoch": float(state.epoch or 0),
                    "epochs": int(config["epochs"]),
                    "loss": self.last_loss,
                    "learning_rate": self.last_lr,
                    "examples": total_examples,
                    "model_id": config["model_id"],
                    "adapter_dir": config["adapter_dir"],
                    "message": "Training on CPU...",
                })

            def on_train_end(self, args, state, control, **kwargs):
                elapsed = int(time.time() - self.start)
                atomic_write_json(STATE_FILE, {
                    "status": "saving",
                    "pid": os.getpid(),
                    "started_at": config["started_at"],
                    "elapsed_seconds": elapsed,
                    "step": int(state.global_step),
                    "max_steps": int(state.max_steps),
                    "epoch": float(state.epoch or 0),
                    "epochs": int(config["epochs"]),
                    "loss": self.last_loss,
                    "learning_rate": self.last_lr,
                    "examples": total_examples,
                    "model_id": config["model_id"],
                    "adapter_dir": config["adapter_dir"],
                    "message": "Saving LoRA adapter...",
                })

        training_args = TrainingArguments(
            output_dir=str(config["output_dir"]),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            num_train_epochs=int(config["epochs"]),
            learning_rate=float(config["learning_rate"]),
            lr_scheduler_type="linear",
            warmup_ratio=0.05,
            logging_steps=1,
            save_strategy="epoch",
            fp16=False,
            bf16=False,
            optim="adamw_torch",
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            weight_decay=0.01,
            report_to=[],
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[StatsCallback()],
        )

        append_log("Training started.")
        trainer.train()

        adapter_dir = Path(config["adapter_dir"]).expanduser()
        adapter_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        append_log(f"Saved adapter to: {adapter_dir}")

        final_state = read_json(STATE_FILE, {})
        final_state.update({
            "status": "done",
            "message": "Training complete.",
            "adapter_dir": str(adapter_dir),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        })
        atomic_write_json(STATE_FILE, final_state)

    except Exception as e:
        append_log("ERROR:")
        append_log(traceback.format_exc())

        state = read_json(STATE_FILE, {})
        state.update({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        })
        atomic_write_json(STATE_FILE, state)


# ============================================================
# TUI MONITOR
# ============================================================

def format_elapsed(seconds):
    if seconds is None:
        return "-"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def draw_monitor(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)

    while True:
        state = read_json(STATE_FILE, {})
        config = read_json(CONFIG_FILE, {})

        status = state.get("status", "unknown")
        pid = state.get("pid")
        alive = is_pid_alive(pid) if pid else False

        draw_box(stdscr, "CPU Recruiting AI Trainer — Live Monitor")

        safe_addstr(stdscr, 2, 4, f"Status: {status}")
        safe_addstr(stdscr, 3, 4, f"Worker PID: {pid or '-'}")
        safe_addstr(stdscr, 4, 4, f"Worker alive: {'yes' if alive else 'no'}")
        safe_addstr(stdscr, 5, 4, f"Model: {state.get('model_id') or config.get('model_id', '-')}")
        safe_addstr(stdscr, 6, 4, f"Examples: {state.get('examples', '-')}")
        safe_addstr(stdscr, 7, 4, f"Epoch: {state.get('epoch', '-')} / {state.get('epochs') or config.get('epochs', '-')}")
        safe_addstr(stdscr, 8, 4, f"Step: {state.get('step', '-')} / {state.get('max_steps', '-')}")
        safe_addstr(stdscr, 9, 4, f"Loss: {state.get('loss', '-')}")
        safe_addstr(stdscr, 10, 4, f"Learning rate: {state.get('learning_rate', '-')}")
        safe_addstr(stdscr, 11, 4, f"Elapsed: {format_elapsed(state.get('elapsed_seconds'))}")
        safe_addstr(stdscr, 12, 4, f"Adapter: {state.get('adapter_dir') or config.get('adapter_dir', '-')}")

        msg = state.get("message", "")
        safe_addstr(stdscr, 14, 4, f"Message: {msg}")

        safe_addstr(stdscr, 16, 4, "Controls:")
        safe_addstr(stdscr, 17, 6, "D = detach TUI, keep worker running")
        safe_addstr(stdscr, 18, 6, "R = refresh")
        safe_addstr(stdscr, 19, 6, "Q = quit monitor only")
        safe_addstr(stdscr, 20, 6, "Ctrl+C = quit monitor only")

        safe_addstr(stdscr, 22, 4, f"State file: {STATE_FILE}")
        safe_addstr(stdscr, 23, 4, f"Log file: {LOG_FILE}")

        stdscr.refresh()

        key = stdscr.getch()

        if key in (ord("d"), ord("D")):
            curses.endwin()
            print("\nDetached TUI. Training worker should continue if it is running.")
            print(f"Reattach with: {SCRIPT_PATH} --attach")
            print(f"Watch logs with: tail -f {LOG_FILE}")
            return

        if key in (ord("q"), ord("Q")):
            return

        if status in ("done", "error") and not alive:
            # Keep the final screen visible instead of instantly closing.
            pass

        time.sleep(1)


# ============================================================
# LAUNCH
# ============================================================

def launch_worker(config):
    if current_worker_alive():
        return False

    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
    process = ctx.Process(target=training_worker, args=(config,))
    process.daemon = False
    process.start()

    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    return True


def run_new(stdscr):
    config = wizard(stdscr)
    launch_worker(config)
    draw_monitor(stdscr)


def run_attach(stdscr):
    if not STATE_FILE.exists():
        draw_box(stdscr, "Attach")
        safe_addstr(stdscr, 3, 4, "No existing training state found.")
        safe_addstr(stdscr, 5, 4, "Start a new training run first.")
        safe_addstr(stdscr, 7, 4, "Press any key to exit.")
        stdscr.getch()
        return

    draw_monitor(stdscr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attach", action="store_true", help="Attach to the current training monitor.")
    parser.add_argument("--status", action="store_true", help="Print current status and exit.")
    return parser.parse_args()


def print_status():
    state = read_json(STATE_FILE, {})
    if not state:
        print("No current training state found.")
        return

    pid = state.get("pid")
    alive = is_pid_alive(pid) if pid else False

    print(json.dumps({
        "status": state.get("status"),
        "pid": pid,
        "alive": alive,
        "step": state.get("step"),
        "max_steps": state.get("max_steps"),
        "epoch": state.get("epoch"),
        "loss": state.get("loss"),
        "elapsed_seconds": state.get("elapsed_seconds"),
        "adapter_dir": state.get("adapter_dir"),
        "message": state.get("message"),
    }, indent=2))


def main():
    args = parse_args()

    if args.status:
        print_status()
        return

    if args.attach:
        curses.wrapper(run_attach)
    else:
        curses.wrapper(run_new)


if __name__ == "__main__":
    main()