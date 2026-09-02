#!/usr/bin/env bash
# GPU box environment (Ubuntu + NVIDIA driver present). One GPU per run.
# Usage: bash setup/setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null 2>&1 || { echo "No NVIDIA driver found."; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 .venv
# Install torch first so vllm/trl resolve against the CUDA build.
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[ml,gpu,dev]"

# TRL supports a bounded range of vLLM versions. Verify before wasting a GPU-hour.
uv run python - << 'PY'
import importlib.metadata as m
for p in ["torch", "transformers", "peft", "trl", "vllm"]:
    try:
        print(f"{p:14s} {m.version(p)}")
    except m.PackageNotFoundError:
        print(f"{p:14s} MISSING")
print("Check TRL's documented vLLM version range: https://huggingface.co/docs/trl/vllm_integration")
PY
uv run python -m rlordata.env_check

# Model cache on the fast local disk; set HF_HOME so every tool agrees.
sudo mkdir -p /data/hf && sudo chown "$USER" /data/hf
grep -q HF_HOME ~/.bashrc || echo 'export HF_HOME=/data/hf' >> ~/.bashrc
export HF_HOME=/data/hf
uv run hf download Qwen/Qwen3-4B-Base --quiet || echo "Model download failed; check HF_TOKEN in .env"

# Idle shutdown: stops the instance after 30 min with no GPU activity. Do not disable.
bash setup/idle_shutdown.sh install
echo "OK. Next: 'make test', then the task in tasks/."
