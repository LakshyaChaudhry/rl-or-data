#!/usr/bin/env bash
# GPU-idle evidence logger for an unattended queue. Every minute appends utilisation and the alive
# rlordata processes to LOG (put LOG on the store so it survives termination). After IDLE_DUMP_MIN
# consecutive minutes under 5 % utilisation while the queue is alive it takes one py-spy dump of every
# rlordata process into the same directory — the post-mortem for a hung job, taken BEFORE the idle
# guard (30 min) terminates the box. Never kills anything. Exits when scripts/run_queue.py exits.
# Usage: bash scripts/gpu_watch.sh /path/to/store/logs/grpo_gpuwatch_<ts>.log
set -u
LOG=${1:?log path}
IDLE_DUMP_MIN=${IDLE_DUMP_MIN:-10}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$(dirname "$LOG")"
idle=0; dumped=0
echo "$(date -u +%FT%TZ) gpu_watch started (dump after ${IDLE_DUMP_MIN} idle min)" >> "$LOG"
while pgrep -f "scripts/run_queue.py" >/dev/null; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  procs=$(pgrep -af "rlordata|run_queue" | grep -v -E "pgrep|gpu_watch|grpo_ckpt_sync" | cut -c1-120 | tr '\n' '|')
  if [ "${util:-0}" -lt 5 ]; then idle=$((idle + 1)); else idle=0; dumped=0; fi
  echo "$(date -u +%FT%TZ) util=${util:-?}% mem=${mem:-?}MiB idle_min=$idle procs=${procs}" >> "$LOG"
  if [ "$idle" -ge "$IDLE_DUMP_MIN" ] && [ "$dumped" = 0 ]; then
    dumped=1
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    echo "$stamp GPU idle ${idle} min with the queue alive — dumping stacks" >> "$LOG"
    nvidia-smi >> "$(dirname "$LOG")/grpo_hang_${stamp}_nvidia-smi.txt" 2>&1
    for pid in $(pgrep -f "python.*rlordata|python.*run_queue" ); do
      out="$(dirname "$LOG")/grpo_hang_${stamp}_pid${pid}.txt"
      { ps -o pid,etimes,pcpu,rss,args -p "$pid"; echo; sudo -n "$(command -v uvx)" py-spy dump --pid "$pid" 2>&1 \
          || uvx py-spy dump --pid "$pid" 2>&1; } > "$out"
      echo "$stamp dumped pid $pid -> $out" >> "$LOG"
    done
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) queue exited; gpu_watch done" >> "$LOG"
