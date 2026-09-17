#!/usr/bin/env bash
# The RFT finals box, unattended, as ONE process (the idle guard terminates the box after 30 idle
# minutes, so there can be no gap waiting on a human). Notebook 2026-09-17.
#
#   Phase A  re-select: scripts/rft_sweep.py per arm. All 27 sweep trainings are finished, so every
#            train stage skips (budgets.json) and only the val eval runs, through the fixed native
#            vLLM LoRA path (the old val evals merged the adapter into bf16 weights: notebook
#            2026-09-14; they are kept as eval/val_bf16merge_stale). Writes sweep.json + chosen.json.
#   Gate     three chosen.json, 9 finished rows per arm, and per arm the 9 new val accuracies are
#            not all equal to the stale ones (equal everywhere = the fixed path is not in use).
#   Phase B  scripts/rft_finals.py per arm: train seeds 2 and 3 with the chosen (lr, epochs), final
#            eval of seeds 1-3 (once), per-arm report. A failed arm is logged and the queue moves
#            on (an idle GPU helps nobody); the script then exits non-zero.
#
# Idempotent: finished trainings skip via budgets.json, finished evals via samples.jsonl. After a dead
# box: same setup, restore runs/rft/<arm>/seed{2,3}_* as well, launch this script again.
# Syncs only runs/rft/<arm> (never sync-all: the store's runs/ also holds the live GRPO queue's).
#
# Launch (after setup/setup_gpu.sh, within a minute or two of the guard arming):
#   TS=$(date -u +%Y%m%dT%H%MZ); QLOG=$RLORDATA_ARTIFACTS/logs/rft_finals_queue_$TS.log
#   nohup bash scripts/rft_finals_queue.sh >> "$QLOG" 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_SYNC=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
set -a; . ./.env; set +a
: "${RLORDATA_ARTIFACTS:?RLORDATA_ARTIFACTS must be set in .env}"
STORE=$RLORDATA_ARTIFACTS
ARMS=(mixed curated easy)
TS=$(date -u +%Y%m%dT%H%MZ)
mkdir -p "$STORE/logs"
say() { echo "[rft_queue $(date -u +%FT%TZ)] $*"; }
sync_arm() { uv run python -m rlordata.artifacts sync "runs/rft/rft_$1" || say "WARNING: sync of rft_$1 failed"; }
trap 'rc=$?; say "exit rc=$rc"; for a in "${ARMS[@]}"; do [ -d "runs/rft/rft_$a" ] && sync_arm "$a"; done' EXIT

say "git $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) store=$STORE"

# ---- Phase A: re-select (eval only) ----
for ARM in "${ARMS[@]}"; do
  ALOG="$STORE/logs/rft_reselect_${ARM}_$TS.log"
  say "phase A: $ARM -> $ALOG"
  uv run python scripts/rft_sweep.py --config "configs/rft/$ARM.yaml" 2>&1 | tee "$ALOG"
  skipped=$(grep -c "already finished" "$ALOG" || true)
  [ "$skipped" = 9 ] || { say "STOP: $ARM: $skipped/9 train stages skipped — a sweep run would have been (re)trained"; exit 1; }
  [ -f "runs/rft/rft_$ARM/chosen.json" ] || { say "STOP: $ARM: no chosen.json"; exit 1; }
  sync_arm "$ARM"
done

say "gate before phase B"
uv run python - "${ARMS[@]}" << 'PY'
import json, sys
from pathlib import Path

ok = True
total = 0.0
for arm in sys.argv[1:]:
    d = Path("runs/rft") / f"rft_{arm}"
    new = json.loads((d / "sweep.json").read_text())["results"]
    stale = json.loads((d / "sweep_bf16merge_stale.json").read_text())["results"]
    chosen = json.loads((d / "chosen.json").read_text())
    old = json.loads((d / "chosen_bf16merge_stale.json").read_text())
    key = lambda r: (r["learning_rate"], r["epochs"])  # noqa: E731
    finished = [r for r in new if r.get("status") == "finished"]
    if len(finished) != 9:
        print(f"GATE FAIL {arm}: {len(finished)}/9 sweep rows finished (selection needs the full grid)")
        ok = False
        continue
    s = {key(r): r["val_accuracy"] for r in stale}
    changed = sum(1 for r in finished if s.get(key(r)) != r["val_accuracy"])
    if changed == 0:
        print(f"GATE FAIL {arm}: all 9 new val accuracies equal the bf16-merge ones — fixed eval path not in use")
        ok = False
    wall_h = json.loads((Path(chosen["sweep_run_dir"]) / "meta.json").read_text())["wall_clock_s"] / 3600
    total += 2 * wall_h
    print(
        f"{arm}: chosen lr={chosen['learning_rate']:g} ep={chosen['epochs']} acc={chosen['val_accuracy']:.3f} "
        f"(stale: lr={old['learning_rate']:g} ep={old['epochs']} acc={old['val_accuracy']:.3f}); "
        f"{changed}/9 accuracies changed; seed-1 training took {wall_h:.2f} GPU-h -> 2 seeds ≈ {2 * wall_h:.1f} GPU-h"
    )
rate = float(__import__("os").environ.get("RLORDATA_GPU_RATE_USD_PER_HOUR", 4.29))
evals = 9 * 1.4  # GRPO final eval measured 1.41 GPU-h per adapter (notebook 2026-09-16)
print(f"phase B estimate: {total:.1f} GPU-h training + ~{evals:.1f} GPU-h final evals = {total + evals:.1f} GPU-h ≈ ${(total + evals) * rate:.0f}")
sys.exit(0 if ok else 1)
PY

# ---- Phase B: seeds 2 and 3 + final evals ----
failed=()
for ARM in "${ARMS[@]}"; do
  BLOG="$STORE/logs/rft_finals_${ARM}_$TS.log"
  say "phase B: $ARM -> $BLOG"
  if uv run python scripts/rft_finals.py --config "configs/rft/$ARM.yaml" 2>&1 | tee "$BLOG"; then
    say "phase B: $ARM done"
  else
    say "FAILED: finals for $ARM (see $BLOG) — continuing with the next arm"
    failed+=("$ARM")
  fi
  sync_arm "$ARM"
done

if [ "${#failed[@]}" -gt 0 ]; then say "finished with failures: ${failed[*]}"; exit 1; fi
say "all arms finished"
