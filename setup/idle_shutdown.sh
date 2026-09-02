#!/usr/bin/env bash
# Installs a cron job that shuts the instance down after 30 consecutive minutes of <5% GPU util.
# Credits burn on idle boxes; this is the single most important script in setup/.
# Usage: bash setup/idle_shutdown.sh install | status
set -euo pipefail
STATE=/var/tmp/gpu_idle_minutes
case "${1:-install}" in
  install)
    sudo tee /usr/local/bin/gpu_idle_check.sh >/dev/null << 'INNER'
#!/usr/bin/env bash
STATE=/var/tmp/gpu_idle_minutes
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1)
[ -f "$STATE" ] || echo 0 > "$STATE"
if [ "${UTIL:-0}" -lt 5 ]; then
  N=$(( $(cat "$STATE") + 1 )); echo "$N" > "$STATE"
  if [ "$N" -ge 30 ]; then logger "gpu idle 30 min: shutting down"; sudo shutdown -h now; fi
else
  echo 0 > "$STATE"
fi
INNER
    sudo chmod +x /usr/local/bin/gpu_idle_check.sh
    ( crontab -l 2>/dev/null | grep -v gpu_idle_check ; echo "* * * * * /usr/local/bin/gpu_idle_check.sh" ) | crontab -
    echo "idle shutdown installed (30 min at <5% util)";;
  status) cat "$STATE" 2>/dev/null || echo 0;;
  *) echo "unknown command"; exit 1;;
esac
