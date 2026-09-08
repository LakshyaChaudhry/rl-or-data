#!/usr/bin/env bash
# GPU box environment (Lambda Cloud 1x H100, Ubuntu + NVIDIA driver present). One GPU per run.
# Usage: bash setup/setup_gpu.sh          (see setup/LAMBDA_LAUNCH.md for the steps before this)
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null 2>&1 || { echo "No NVIDIA driver found."; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv

[ -f .env ] || { echo "No .env — copy .env.example to .env and fill HF_TOKEN, LAMBDA_API_KEY, RLORDATA_ARTIFACTS, RLORDATA_GPU_RATE_USD_PER_HOUR first."; exit 1; }
set -a; . ./.env; set +a

# ---- durable storage: Lambda persistent filesystem (mounted at /lambda/nfs/<name> at launch) ----
# Local disk is ephemeral: it disappears when the instance is terminated. HF cache and artifacts go on the
# filesystem when one is mounted; otherwise warn — everything must then be synced to S3 before terminate.
NFS_DIR=$(ls -d /lambda/nfs/*/ 2>/dev/null | head -n 1 || true)
if [ -n "$NFS_DIR" ]; then
  NFS_DIR="${NFS_DIR%/}"
  echo "persistent filesystem: $NFS_DIR"
  export HF_HOME="$NFS_DIR/hf"
  mkdir -p "$HF_HOME"
  if [ -z "${RLORDATA_ARTIFACTS:-}" ]; then
    export RLORDATA_ARTIFACTS="$NFS_DIR/rlordata-artifacts"
    echo "RLORDATA_ARTIFACTS=$RLORDATA_ARTIFACTS" >> .env
    echo "RLORDATA_ARTIFACTS was unset; defaulted to $RLORDATA_ARTIFACTS (written to .env)"
  fi
  mkdir -p "$RLORDATA_ARTIFACTS"
else
  echo "WARNING: no /lambda/nfs/* filesystem mounted. Local disk is EPHEMERAL." >&2
  echo "         Model cache goes to /data/hf and is lost at terminate; RLORDATA_ARTIFACTS must be s3://... ." >&2
  [ -n "${RLORDATA_ARTIFACTS:-}" ] || { echo "RLORDATA_ARTIFACTS is unset and there is no persistent filesystem. Refusing: results would be lost."; exit 1; }
  sudo mkdir -p /data/hf && sudo chown "$USER" /data/hf
  export HF_HOME=/data/hf
fi
grep -q '^export HF_HOME=' ~/.bashrc && sed -i '/^export HF_HOME=/d' ~/.bashrc
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
echo "HF_HOME=$HF_HOME"

# ---- python env ----
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.11 .venv
# Let vLLM pin the torch build it was compiled against (it ships CUDA wheels); installing torch first
# from a different index produced ABI mismatches in the past.
uv pip install -e ".[ml,gpu,dev]"
uv pip install -e ".[transfer]" || echo "reasoning-gym install failed; transfer-pick will skip RG tasks"

# TRL supports a bounded range of vLLM versions. Verify before wasting a GPU-hour.
uv run python - << 'PY'
import importlib.metadata as m
for p in ["torch", "transformers", "peft", "trl", "vllm", "boto3", "reasoning-gym"]:
    try:
        print(f"{p:14s} {m.version(p)}")
    except m.PackageNotFoundError:
        print(f"{p:14s} MISSING")
print("Check TRL's documented vLLM version range: https://huggingface.co/docs/trl/vllm_integration")
PY
uv run python -m rlordata.env_check
make test

# ---- model weights into HF_HOME (persistent when on the filesystem) ----
uv run hf download Qwen/Qwen3-4B-Base --quiet || echo "Qwen3-4B-Base download failed; check network/HF_TOKEN"

# ---- idle guard LAST (so setup itself does not count as idle time). Terminates via the Lambda API. Do not disable. ----
bash setup/idle_shutdown.sh install
bash setup/idle_shutdown.sh test-api || echo "WARNING: could not verify Lambda terminate wiring — fix before leaving the box unattended"
echo "OK. Next: the tasks/02 GPU sequence — make cap-run && make cap && git add configs/locked/cap.yaml && git commit && make tier && make eval-base"
