#!/usr/bin/env bash
# Copies GRPO checkpoints and small run-dir files to the store while the queue runs (a training job
# itself syncs only at its end). Exact paths under runs/grpo/ only — never sync-all, never runs/rft.
# Exits when scripts/run_queue.py exits. Started by setup/grpo_box.sh.
set -u
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
S="${RLORDATA_ARTIFACTS:?}"
MIN_AGE=60   # seconds a checkpoint's trainer_state.json must be untouched before copying

ts() { date -u +%H:%M:%S; }
sync_paths() { nice -n 19 ionice -c3 uv run python -m rlordata.artifacts sync "$@" 2>&1 | sed "s/^/$(ts) /"; }
complete_lines() { [ ! -s "$1" ] || [ "$(tail -c1 "$1" | od -An -c | tr -d ' ')" = '\n' ]; }

echo "$(ts) checkpoint sync loop started (store $S)"
while pgrep -f "scripts/run_queue.py" >/dev/null; do
  for rd in runs/grpo/grpo_*/; do
    rd=${rd%/}
    for ck in "$rd"/checkpoints/checkpoint-*; do
      st="$ck/trainer_state.json"
      [ -f "$st" ] || continue
      age=$(( $(date +%s) - $(stat -c %Y "$st") ))
      [ "$age" -ge "$MIN_AGE" ] || continue
      if [ -f "$S/$st" ] && cmp -s "$st" "$S/$st" && \
         [ "$(du -sb "$ck" | cut -f1)" = "$(du -sb "$S/$ck" 2>/dev/null | cut -f1)" ]; then
        continue   # already in the store, same size
      fi
      echo "$(ts) syncing $ck"
      sync_paths "$ck"
      small=()
      for f in "$rd"/*; do
        [ -f "$f" ] || continue
        case "$f" in *.jsonl) complete_lines "$f" || { sleep 2; complete_lines "$f"; } || { echo "$(ts) skip partial $f"; continue; } ;; esac
        small+=("$f")
      done
      [ -d "$rd/checkpoints/completions" ] && small+=("$rd/checkpoints/completions")
      sync_paths "${small[@]}"
    done
  done
  sleep 60
done
echo "$(ts) queue exited; loop done (the queue does its own final sync)"
