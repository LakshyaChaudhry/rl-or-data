#!/usr/bin/env bash
# Local dev environment (Apple Silicon). Creates .venv with uv and installs the CPU/MPS stack.
# Usage: bash setup/setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null 2>&1 || { echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

uv venv --python 3.11 .venv
uv pip install -e ".[ml,dev]"

uv run python - << 'PY'
import torch, transformers, peft
print("torch", torch.__version__, "| mps available:", torch.backends.mps.is_available())
print("transformers", transformers.__version__, "| peft", peft.__version__)
PY
uv run python -m rlordata.env_check
echo "OK. Next: copy .env.example to .env, then 'make test'."
