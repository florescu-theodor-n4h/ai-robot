# Copilot Instructions

## Test commands

- Run the full test suite with `pytest tests/ -v`.
- Run one test module with `pytest tests/test_openai_ai_tester.py -v`.
- Run one test class with `pytest tests/test_text_utils.py::TestExtractText -v`.
- Run one specific test with `pytest tests/test_openai_ai_tester.py::test_chat_session_sends_plain_message -v`.
- Run coverage when needed with `pytest tests/ -v --cov=. --cov-report=html`.

## High-level architecture

- `runLLMAgentForAgenticDevs.py` is the main inference server and the source of truth for config dataclasses, environment setup, text utilities, model loading, and FastAPI route registration.
- `training_src/config.py`, `training_src/api.py`, `training_src/model.py`, `training_src/text_utils.py`, `training_src/logging_setup.py`, and `training_src/__main__.py` are mostly compatibility wrappers that re-export behavior from `runLLMAgentForAgenticDevs.py`.
- `training_src/tui.py` and `training_src/status_monitor.py` add the Rich-based TUI and live status monitor for the inference server; `demo_tui.py` exercises those UI pieces without starting the server.
- `run.py` is a separate CPU QLoRA training workflow with a curses wizard plus a monitor. It writes machine-readable state into `runs_cpu/current_state.json`, `runs_cpu/current_config.json`, `runs_cpu/current_worker.pid`, and `runs_cpu/current_log.txt`.
- `run_openai_cpu.py` is another separate workflow: a curses-managed OpenAI-compatible server for trained adapters. It keeps its own dashboard state in `runs_cpu/openai_server_state.json`, `runs_cpu/openai_server_config.json`, `runs_cpu/openai_server.pid`, and `runs_cpu/openai_server_log.txt`, while also reading the training artifacts in `runs_cpu/`.
- `openai_ai_tester.py` is the local curses client for the OpenAI-compatible `/v1` API and supports `/test path/to/file.py` to submit Python source for review through the local server.
- The tests are intentionally lightweight: they verify config, TUI, text-processing, and client/helper behavior without downloading models or booting the real ML servers.

## Key conventions

- Treat `runLLMAgentForAgenticDevs.py`, `run.py`, and `run_openai_cpu.py` as the source-of-truth scripts. If behavior changes, update the wrapper modules in `training_src/` only to keep them aligned, not to introduce parallel implementations.
- Preserve the file-based state protocol under `runs_cpu/`. Writers use `atomic_write_json(...)`, and the TUIs/monitors poll those JSON, log, and PID files, so stable filenames and payload keys matter.
- In `run.py` and `run_openai_cpu.py`, environment hardening happens before ML imports: CUDA/HIP visibility is cleared, tokenizer parallelism and telemetry are disabled, and CPU thread env vars are set early. Keep that ordering when editing startup code.
- The helper launchers (`runVenv-qlora.sh`, `runVenv-inference.sh`, and the `enter_venv` symlink) assume local virtualenvs such as `.venv-cpu` and `AI_weight_lora/venv`. Those environments are not committed, so missing venvs are usually setup issues, not reasons to rewrite the scripts.
- Use `./install_everything.sh` to restore executable bits, recreate both expected venvs, and install the repository's system and Python dependencies.
- Reuse the existing text/message normalization helpers instead of inventing new parsers. This repo already handles OpenAI-style content that may arrive as strings, dicts, or lists of typed content.
- `hard_clip(...)` intentionally keeps the tail of oversized prompts/text so the most recent context survives truncation.
- Tests commonly import repo-root scripts directly via `sys.path.insert(...)` and use fake clients/fixtures instead of end-to-end server startup. Follow that lightweight testing style unless a real integration path is necessary.
