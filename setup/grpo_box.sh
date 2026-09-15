#!/usr/bin/env bash
# The GRPO box in one command. Fresh Lambda instance with the persistent filesystem attached:
#
#   git clone https://github.com/LakshyaChaudhry/rl-or-data.git && cd rl-or-data && bash setup/grpo_box.sh
#
# What it does, in order:
#   1. .env from <filesystem>/bootstrap/.env (put it there once from a working box); Claude Code
#      credentials from <filesystem>/bootstrap/claude/ if you chose to keep them there.
#   2. uv env.
#   3. Restore from the store: base-model evals (the adapter eval's sanity gate needs them), every
#      finished GRPO run without its checkpoints, every unfinished GRPO run whole (so run_queue.py
#      appends --resume). Never _failed / _diag, never runs/rft.
#   4. Print the queue (dry run) and a cost estimate for what remains.
#   5. setup/setup_gpu.sh (tests, weights, idle guard LAST) — then launch the queue within seconds, so
#      the guard's 30-minute idle window never opens during setup or restore.
#   6. Queue log, checkpoint sync loop, and GPU-idle evidence logger, all writing to the store's logs/.
#
# Flags: --dry-run (steps 1-4 only; nothing launched, guard not armed)   --no-setup (skip step 5)
set -euo pipefail
cd "$(dirname "$0")/.."
SETUP=1; DRY=0
for a in "$@"; do case "$a" in --no-setup) SETUP=0;; --dry-run) DRY=1;; *) echo "unknown arg: $a"; exit 2;; esac; done
say() { echo "[grpo_box $(date -u +%H:%M:%S)] $*"; }

NFS_DIR=$(ls -d /lambda/nfs/*/ 2>/dev/null | head -n 1 || true); NFS_DIR=${NFS_DIR%/}
[ -n "$NFS_DIR" ] || { echo "no /lambda/nfs/* mounted — launch the instance with the persistent filesystem attached"; exit 1; }
BOOT="$NFS_DIR/bootstrap"

# ---- 1. keys ----
if [ ! -f .env ]; then
  [ -f "$BOOT/.env" ] || { echo "no .env here and none at $BOOT/.env — from a working box: mkdir -p $BOOT && cp .env $BOOT/.env && chmod 600 $BOOT/.env"; exit 1; }
  cp "$BOOT/.env" .env && chmod 600 .env && say ".env restored from $BOOT/.env"
fi
set -a; . ./.env; set +a
: "${RLORDATA_ARTIFACTS:?RLORDATA_ARTIFACTS must be set in .env}"
STORE=$RLORDATA_ARTIFACTS
[ -d "$STORE/runs" ] || { echo "$STORE/runs not found — wrong filesystem?"; exit 1; }
if [ -f "$BOOT/claude/.credentials.json" ] && [ ! -f "$HOME/.claude/.credentials.json" ]; then
  mkdir -p "$HOME/.claude" && cp "$BOOT/claude/.credentials.json" "$HOME/.claude/" && chmod 600 "$HOME/.claude/.credentials.json"
  say "Claude Code credentials restored from $BOOT/claude/"
fi

# ---- 2. python env (restore needs it; setup_gpu.sh repeats this step idempotently) ----
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export UV_NO_SYNC=1
uv sync --locked --all-extras --python 3.11

# ---- 3. restore ----
restore() { uv run python -m rlordata.artifacts restore "$@"; }
say "restoring base-model evals (adapter-eval sanity gate)"
restore runs/eval/Qwen__Qwen3-4B-Base
shopt -s nullglob
finished=(); unfinished=()
for d in "$STORE"/runs/grpo/grpo_*/; do
  d=${d%/}; name=$(basename "$d"); rel="runs/grpo/$name"
  if [ -f "$d/budgets.json" ]; then
    paths=()
    for f in "$d"/*; do [ "$(basename "$f")" = checkpoints ] && continue; paths+=("$rel/$(basename "$f")"); done
    say "$rel: finished — restoring without checkpoints"
    restore "${paths[@]}"
    n=$(wc -l < "$rel/reward_records.jsonl")
    want=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['completions_consumed'])" "$rel/budgets.json")
    [ "$n" = "$want" ] || { echo "$rel: reward_records.jsonl has $n lines, budgets.json says $want — refusing"; exit 1; }
    for s in step_100 step_200 step_300 final; do [ -d "$rel/adapter/$s" ] || { echo "$rel: adapter/$s missing — refusing"; exit 1; }; done
    finished+=("$name")
  else
    say "$rel: unfinished — restoring whole; the queue will --resume from its latest checkpoint"
    restore "$rel"
    unfinished+=("$name")
  fi
done
shopt -u nullglob
say "finished runs: ${finished[*]:-none}; unfinished: ${unfinished[*]:-none}"

# ---- 4. queue preview + cost ----
uv run python scripts/run_queue.py --queue queue.yaml --dry-run | tee /tmp/grpo_queue_dryrun.txt
n_train=$(grep -cE '^\[queue\] \([0-9]+/[0-9]+\) train_' /tmp/grpo_queue_dryrun.txt || true)
n_eval=$(grep -cE '^\[queue\] \([0-9]+/[0-9]+\) eval_' /tmp/grpo_queue_dryrun.txt || true)
rate=${RLORDATA_GPU_RATE_USD_PER_HOUR:-4.29}
say "remaining: $n_train training jobs (~8.2 GPU-h each, grpo_mixed_s1 measured 8.17) + $n_eval evals"
say "estimate: $(awk -v t="$n_train" -v e="$n_eval" -v r="$rate" 'BEGIN{h=t*8.2+e*1.0; printf "%.1f GPU-h ≈ $%.0f at $%s/h (evals guessed at 1 GPU-h)", h, h*r, r}')"
if [ "$DRY" = 1 ]; then say "dry run: nothing launched, guard not armed"; exit 0; fi

# ---- 5. box setup; the idle guard is its last step ----
if [ "$SETUP" = 1 ]; then bash setup/setup_gpu.sh; fi

# ---- 6. launch (seconds after the guard armed) ----
TS=$(date -u +%Y%m%dT%H%MZ)
mkdir -p "$STORE/logs" logs
QLOG="$STORE/logs/grpo_queue_$TS.log"
ln -sf "$QLOG" "logs/grpo_queue_$TS.log"
nohup uv run python scripts/run_queue.py --queue queue.yaml >> "$QLOG" 2>&1 &
QPID=$!
sleep 5
nohup bash scripts/grpo_ckpt_sync.sh >> "$STORE/logs/grpo_ckptsync_$TS.log" 2>&1 &
nohup bash scripts/gpu_watch.sh "$STORE/logs/grpo_gpuwatch_$TS.log" >/dev/null 2>&1 &
say "queue pid $QPID — log: $QLOG"
say "watch: tail -f $QLOG   |   nvidia-smi   |   $STORE/logs/grpo_gpuwatch_$TS.log"
