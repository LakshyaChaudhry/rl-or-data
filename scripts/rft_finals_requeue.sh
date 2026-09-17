#!/usr/bin/env bash
# Re-run scripts/rft_finals_queue.sh until it exits 0 (at most MAX_PASSES), starting each pass
# within seconds of the previous one so the idle guard's 30-minute window never opens. The queue
# is idempotent (finished trainings and evals skip), so every pass only does what is still missing.
# Optional first argument: pid of a queue pass that is already running; waited for first.
# Usage: nohup bash scripts/rft_finals_requeue.sh [running_queue_pid] >> <store>/logs/rft_finals_requeue_<ts>.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
MAX_PASSES=${MAX_PASSES:-4}
say() { echo "[rft_requeue $(date -u +%FT%TZ)] $*"; }
if [ -n "${1:-}" ]; then
  say "waiting for running queue pid $1"
  while kill -0 "$1" 2>/dev/null; do sleep 10; done
fi
for pass in $(seq 1 "$MAX_PASSES"); do
  QLOG="$RLORDATA_ARTIFACTS/logs/rft_finals_queue_$(date -u +%Y%m%dT%H%MZ)_requeue$pass.log"
  say "pass $pass/$MAX_PASSES -> $QLOG (git $(git rev-parse --short HEAD))"
  if bash scripts/rft_finals_queue.sh >> "$QLOG" 2>&1; then say "pass $pass: queue exited 0 — done"; exit 0; fi
  say "pass $pass: queue exited non-zero"
done
say "giving up after $MAX_PASSES passes"
exit 1
