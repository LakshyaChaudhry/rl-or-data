#!/usr/bin/env bash
# The tasks/06b box in one command (iterated RFT + exploratory larger-cap re-eval + base gsm8k mean@8).
# Fresh Lambda 1× H100 instance with the persistent filesystem attached:
#
#   git clone https://github.com/LakshyaChaudhry/rl-or-data.git && cd rl-or-data \
#     && git checkout tasks06b-iterated-rft && bash setup/tasks06b_box.sh
#
# What it does, in order (same recipe as setup/grpo_box.sh):
#   1. .env from <filesystem>/bootstrap/.env.
#   2. uv env.
#   3. Restore from the store ONLY what the queue needs: base evals (sanity gate), the RFT draw, the
#      six H3-pair run dirs without checkpoints or epoch adapters, and any iterated-RFT /
#      exploratory work a previous box already did (so the queue resumes instead of repeating).
#   4. Preflight: pre-registration ancestry, adapters present, draw intact; queue dry run; estimate.
#   5. setup/setup_gpu.sh (tests, weights, idle guard LAST), then launch within seconds.
#   6. Queue log and GPU-idle evidence logger, both on the store.
#
# Flags: --dry-run (steps 1-4 only; nothing launched, guard not armed)   --no-setup (skip step 5)
set -euo pipefail
cd "$(dirname "$0")/.."
SETUP=1; DRY=0
for a in "$@"; do case "$a" in --no-setup) SETUP=0;; --dry-run) DRY=1;; *) echo "unknown arg: $a"; exit 2;; esac; done
say() { echo "[tasks06b_box $(date -u +%H:%M:%S)] $*"; }
QUEUE=queue_tasks06b.yaml

NFS_DIR=$(ls -d /lambda/nfs/*/ 2>/dev/null | head -n 1 || true); NFS_DIR=${NFS_DIR%/}
[ -n "$NFS_DIR" ] || { echo "no /lambda/nfs/* mounted — launch the instance with the persistent filesystem attached"; exit 1; }
BOOT="$NFS_DIR/bootstrap"

# ---- 1. keys ----
if [ ! -f .env ]; then
  [ -f "$BOOT/.env" ] || { echo "no .env here and none at $BOOT/.env"; exit 1; }
  cp "$BOOT/.env" .env && chmod 600 .env && say ".env restored from $BOOT/.env"
fi
set -a; . ./.env; set +a
: "${RLORDATA_ARTIFACTS:?RLORDATA_ARTIFACTS must be set in .env}"
STORE=$RLORDATA_ARTIFACTS
[ -d "$STORE/runs" ] || { echo "$STORE/runs not found — wrong filesystem?"; exit 1; }

# ---- 2. python env ----
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export UV_NO_SYNC=1
uv sync --locked --all-extras --python 3.11

# ---- 3. restore ----
restore() { uv run python -m rlordata.artifacts restore "$@"; }
say "restoring base-model evals, the RFT draw and the splits' provenance"
restore runs/eval/Qwen__Qwen3-4B-Base data/samples/base_train_mixed_100_k192_seed1.jsonl
for s in 1 2 3; do
  r="runs/rft/rft_curated/seed${s}_lr1e-05_ep4"
  say "$r: final adapter + provenance (no epoch adapters)"
  restore "$r/budgets.json" "$r/config.yaml" "$r/config_hash.txt" "$r/meta.json" "$r/adapter/final" \
          "$r/eval/final/test_300/greedy/config.yaml"
  g="runs/grpo/grpo_curated_s${s}"
  say "$g: evaluated adapter (step_300) + provenance (no checkpoints)"
  restore "$g/budgets.json" "$g/config.yaml" "$g/config_hash.txt" "$g/meta.json" "$g/adapter/step_300" \
          "$g/adapter/final" "$g/eval/final/test_300/greedy/config.yaml"
done
say "restoring any earlier tasks/06b work (resume)"
restore runs/rft/iter_rft_curated runs/exploratory_cap8704 runs/eval/Qwen__Qwen3-4B-Base/gsm8k_500 2>/dev/null || true

# ---- 4. preflight ----
PREREG=$(uv run python -c "import yaml; print(yaml.safe_load(open('configs/rft/iter_curated.yaml'))['preregistration_commit'])")
git merge-base --is-ancestor "$PREREG" HEAD || { echo "HEAD does not descend from the pre-registration commit $PREREG — refusing"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tracked files are modified — result-bearing runs need a clean tree"; git status --short; exit 1; }
uv run python - <<'PY'
import hashlib, json, sys
from pathlib import Path
from rlordata.sampling.draw import read_draw
from rlordata.data.generator import read_jsonl
draw = read_draw("data/samples/base_train_mixed_100_k192_seed1.jsonl", expect_n=192)
curated = read_jsonl("data/splits/train_curated.jsonl")
missing = [p.problem_id[:12] for p in curated if p.problem_id not in draw]
assert not missing, f"curated prompts without a draw: {missing}"
print(f"[preflight] draw ok: {len(draw)} prompts x 192; curated {len(curated)} -> {len(curated) * 64 * 3:,} rollouts over 3 rounds")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()
for s in (1, 2, 3):
    g = Path(f"runs/grpo/grpo_curated_s{s}/adapter")
    a, b = g / "step_300" / "adapter_model.safetensors", g / "final" / "adapter_model.safetensors"
    assert a.exists(), f"{a} missing"
    same = b.exists() and sha(a) == sha(b)
    print(f"[preflight] grpo_curated_s{s}: step_300 {sha(a)[:16]}…  byte-identical to adapter/final: {same}")
    r = Path(f"runs/rft/rft_curated/seed{s}_lr1e-05_ep4/adapter/final/adapter_model.safetensors")
    assert r.exists(), f"{r} missing"
PY
uv run python scripts/run_queue.py --queue "$QUEUE" --dry-run | tee /tmp/tasks06b_queue_dryrun.txt
rate=${RLORDATA_GPU_RATE_USD_PER_HOUR:-3.29}
say "estimate for a full queue: 13–16 GPU-h ≈ \$$(awk -v r="$rate" 'BEGIN{printf "%.0f–%.0f", 13*r, 16*r}') at \$$rate/h (iterated RFT ≈ 10–12, exploratory ≈ 2, gsm8k ≈ 0.3, overhead ≈ 1)"
if [ "$DRY" = 1 ]; then say "dry run: nothing launched, guard not armed"; exit 0; fi

# ---- 5. box setup; the idle guard is its last step ----
if [ "$SETUP" = 1 ]; then bash setup/setup_gpu.sh; fi

# ---- 6. launch (seconds after the guard armed) ----
TS=$(date -u +%Y%m%dT%H%MZ)
mkdir -p "$STORE/logs" logs
QLOG="$STORE/logs/tasks06b_queue_$TS.log"
ln -sf "$QLOG" "logs/tasks06b_queue_$TS.log"
nohup uv run python scripts/run_queue.py --queue "$QUEUE" >> "$QLOG" 2>&1 &
QPID=$!
sleep 5
nohup bash scripts/gpu_watch.sh "$STORE/logs/tasks06b_gpuwatch_$TS.log" >/dev/null 2>&1 &
say "queue pid $QPID — log: $QLOG"
say "watch: tail -f $QLOG   |   nvidia-smi"
say "when the log prints '[queue] done', everything is on the store; the idle guard then terminates the box."
