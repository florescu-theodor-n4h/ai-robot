#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_VENV_DIR="${CPU_VENV_DIR:-$SCRIPT_DIR/.venv-cpu}"
INFERENCE_VENV_DIR="${INFERENCE_VENV_DIR:-$SCRIPT_DIR/AI_weight_lora/venv}"
INSTALL_SYSTEM=1
INSTALL_CPU_VENV=1
INSTALL_INFERENCE_VENV=1

log() {
  printf '[setup] %s\n' "$*"
}

die() {
  printf '[setup] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./install_everything.sh [options]

Options:
  --skip-system         Do not install OS packages.
  --skip-cpu-venv       Do not create/install the .venv-cpu environment.
  --skip-inference-venv Do not create/install the AI_weight_lora/venv environment.
  --python PATH         Python interpreter to use for creating venvs.
  -h, --help            Show this help message.

Environment overrides:
  PYTHON_BIN            Python interpreter used for venv creation.
  CPU_VENV_DIR          Path for the CPU training venv.
  INFERENCE_VENV_DIR    Path for the inference venv.

Notes:
  - The CPU venv installs a CPU-only PyTorch wheel.
  - The inference venv builds llama-cpp-python, so a C/C++ toolchain is required.
  - pip uses --no-cache-dir to reduce disk usage during setup.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-system)
      INSTALL_SYSTEM=0
      shift
      ;;
    --skip-cpu-venv)
      INSTALL_CPU_VENV=0
      shift
      ;;
    --skip-inference-venv)
      INSTALL_INFERENCE_VENV=0
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a path"
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run_pkg_install() {
  if have_cmd sudo && [[ "$(id -u)" -ne 0 ]]; then
    sudo "$@"
  else
    "$@"
  fi
}

install_system_packages() {
  if have_cmd dnf; then
    log "Installing system packages with dnf"
    run_pkg_install dnf install -y \
      gcc gcc-c++ make cmake git curl \
      python3 python3-pip python3-devel
    return
  fi

  if have_cmd yum; then
    log "Installing system packages with yum"
    run_pkg_install yum install -y \
      gcc gcc-c++ make cmake git curl \
      python3 python3-pip python3-devel
    return
  fi

  if have_cmd apt-get; then
    log "Installing system packages with apt-get"
    run_pkg_install apt-get update
    run_pkg_install apt-get install -y \
      build-essential cmake git curl \
      python3 python3-pip python3-venv python3-dev
    return
  fi

  die "No supported package manager found (dnf, yum, apt-get)"
}

fix_exec_bits() {
  log "Restoring executable bits for launcher and helper scripts"
  chmod +x \
    runVenv-qlora.sh \
    runVenv-inference.sh \
    run.py \
    run_openai_cpu.py \
    runLLMAgentForAgenticDevs.py \
    demo_tui.py \
    openai_ai_tester.py \
    build_bitsandbytesROCM.sh \
    firewall.sh \
    remove_trash.sh \
    view-tomcat-log-bugs.sh
}

ensure_base_dirs() {
  mkdir -p "$SCRIPT_DIR/AI_weight_lora"
}

create_venv() {
  local venv_dir="$1"
  log "Creating venv at $venv_dir"
  "$PYTHON_BIN" -m venv --clear "$venv_dir"
  "$venv_dir/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel
}

install_cpu_venv() {
  local py="$CPU_VENV_DIR/bin/python"
  [[ -x "$py" ]] || die "CPU venv python not found: $py"

  log "Installing CPU-only torch into $CPU_VENV_DIR"
  "$py" -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

  log "Installing CPU workflow dependencies into $CPU_VENV_DIR"
  "$py" -m pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    transformers \
    peft \
    datasets \
    accelerate \
    sentencepiece \
    huggingface-hub \
    einops \
    safetensors \
    click \
    rich \
    pytest \
    pytest-cov
}

install_inference_venv() {
  local py="$INFERENCE_VENV_DIR/bin/python"
  [[ -x "$py" ]] || die "Inference venv python not found: $py"

  log "Installing CPU-only torch into $INFERENCE_VENV_DIR"
  "$py" -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

  log "Installing inference server dependencies into $INFERENCE_VENV_DIR"
  "$py" -m pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    huggingface-hub \
    transformers \
    peft \
    rich \
    pytest \
    pytest-cov

  log "Building llama-cpp-python into $INFERENCE_VENV_DIR"
  CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}" \
    "$py" -m pip install --no-cache-dir llama-cpp-python
}

write_enter_venv_link() {
  log "Refreshing enter_venv symlink"
  ln -sfn "AI_weight_lora/venv/bin/activate" "$SCRIPT_DIR/enter_venv"
}

verify_common_paths() {
  [[ -x "$PYTHON_BIN" || "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ]] || die "Python interpreter not found: $PYTHON_BIN"
  [[ -d "$SCRIPT_DIR/tests" ]] || die "Run this script from the repository root"
}

main() {
  verify_common_paths
  ensure_base_dirs
  fix_exec_bits

  if [[ "$INSTALL_SYSTEM" -eq 1 ]]; then
    install_system_packages
  fi

  if [[ "$INSTALL_CPU_VENV" -eq 1 ]]; then
    create_venv "$CPU_VENV_DIR"
    install_cpu_venv
  fi

  if [[ "$INSTALL_INFERENCE_VENV" -eq 1 ]]; then
    create_venv "$INFERENCE_VENV_DIR"
    install_inference_venv
  fi

  write_enter_venv_link

  log "Setup complete"
  log "CPU venv:        $CPU_VENV_DIR"
  log "Inference venv:  $INFERENCE_VENV_DIR"
  log "Activate infer:  source $SCRIPT_DIR/enter_venv"
  log "Activate CPU:    source $CPU_VENV_DIR/bin/activate"
}

main "$@"
