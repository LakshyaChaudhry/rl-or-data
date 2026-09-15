#!/usr/bin/env bash
# Idle guard: after 30 consecutive minutes of <5 % GPU utilisation, sync artifacts and TERMINATE the instance.
#
# Lambda Cloud keeps billing an instance that was only shut down (`shutdown -h` puts it in "Alert" status
# and the meter keeps running — docs.lambda.ai, "Creating and managing instances"). So with
# LAMBDA_API_KEY set this script finds its own instance id (public IP matched against
# GET https://cloud.lambda.ai/api/v1/instances) and calls
# POST https://cloud.lambda.ai/api/v1/instance-operations/terminate  {"instance_ids": ["<id>"]}
# with `Authorization: Bearer $LAMBDA_API_KEY` (endpoints and auth verified against the published
# OpenAPI spec at cloud.lambda.ai/api/v1/openapi.json on 2026-09-07). Without the key it falls back to
# `shutdown -h now` and warns loudly that Lambda keeps billing.
#
# Credits burn on idle boxes; this is the single most important script in setup/. There is deliberately
# no `uninstall` and no hold file. Do not add one.
#
# Usage: bash setup/idle_shutdown.sh install | status | test-api | check-now
#   install    write /usr/local/bin/gpu_idle_check.sh + /etc/rlordata/idle.env (from .env) + a cron entry
#   status     idle minutes so far
#   test-api   resolve this instance's id via the Lambda API and print it (no terminate) — run once after install
#   check-now  run one check iteration in the foreground (may terminate if already idle ≥ 30 min)
set -euo pipefail

STATE=/var/tmp/gpu_idle_minutes
IDLE_MINUTES=30
UTIL_THRESHOLD=5
REPO="$(cd "$(dirname "$0")/.." && pwd)"

case "${1:-install}" in
  install)
    # Secrets for the cron job: root-only copy of the keys it needs, taken from the repo .env.
    sudo mkdir -p /etc/rlordata
    {
      echo "RLORDATA_REPO=$REPO"
      echo "RLORDATA_USER=$USER"
      if [ -f "$REPO/.env" ]; then
        grep -E '^(LAMBDA_API_KEY|RLORDATA_ARTIFACTS|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_REGION)=' "$REPO/.env" || true
      fi
    } | sudo tee /etc/rlordata/idle.env >/dev/null
    sudo chmod 600 /etc/rlordata/idle.env
    if ! sudo grep -q '^LAMBDA_API_KEY=.\+' /etc/rlordata/idle.env; then
      echo "WARNING: LAMBDA_API_KEY is empty in .env. Idle guard will only SHUT DOWN — Lambda keeps billing a shut-down instance." >&2
      echo "         Put the key in .env and re-run: bash setup/idle_shutdown.sh install" >&2
    fi

    sudo tee /usr/local/bin/gpu_idle_check.sh >/dev/null << 'INNER'
#!/usr/bin/env bash
# Installed by setup/idle_shutdown.sh. Runs every minute from cron as root.
STATE=/var/tmp/gpu_idle_minutes
IDLE_MINUTES=30
UTIL_THRESHOLD=5
LAMBDA_API=https://cloud.lambda.ai/api/v1
[ -f /etc/rlordata/idle.env ] && set -a && . /etc/rlordata/idle.env && set +a

log() {
  logger -t gpu_idle "$*"; echo "[gpu_idle] $*"
  # Also append to the persistent store: local syslog dies with the instance, so without this a
  # guard termination is indistinguishable from the box vanishing (two GRPO boxes, 2026-09-15).
  case "${RLORDATA_ARTIFACTS:-}" in
    ""|s3://*) ;;
    *) sudo -u "${RLORDATA_USER:-ubuntu}" -H mkdir -p "$RLORDATA_ARTIFACTS/logs" 2>/dev/null
       printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$(hostname)" "$*" \
         | sudo -u "${RLORDATA_USER:-ubuntu}" -H tee -a "$RLORDATA_ARTIFACTS/logs/idle_guard_$(hostname).log" >/dev/null 2>&1 ;;
  esac
}

evidence() {
  # What was (not) running when the guard fired — the only post-mortem a terminated box leaves.
  log "evidence gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  ps -eo pid,etimes,pcpu,rss,args --sort=-etimes 2>/dev/null | grep -E 'rlordata|run_queue|vllm|rft_|grpo' | grep -v grep | head -20 \
    | while read -r l; do log "evidence proc: $l"; done
}

lambda_instance_id() {
  # Match this box's public IP (and, as a fallback, any local address) against the account's instances.
  local pub ids
  pub=$(curl -s --max-time 10 https://checkip.amazonaws.com || curl -s --max-time 10 https://api.ipify.org || true)
  pub=$(echo "$pub" | tr -d '[:space:]')
  local locals; locals=$(hostname -I 2>/dev/null || true)
  curl -s --max-time 20 -H "Authorization: Bearer $LAMBDA_API_KEY" "$LAMBDA_API/instances" \
    | python3 -c '
import json, sys
pub = sys.argv[1]; locals_ = set(sys.argv[2].split())
data = json.load(sys.stdin).get("data", [])
hits = [i["id"] for i in data if pub and i.get("ip") == pub]
if not hits:
    hits = [i["id"] for i in data if i.get("ip") in locals_ or i.get("private_ip") in locals_]
print(hits[0] if len(hits) == 1 else "")
' "$pub" "$locals"
}

lambda_terminate() {
  local id="$1"
  curl -s --max-time 30 -X POST -H "Authorization: Bearer $LAMBDA_API_KEY" -H "Content-Type: application/json" \
    -d "{\"instance_ids\": [\"$id\"]}" "$LAMBDA_API/instance-operations/terminate"
}

sync_artifacts() {
  # Best effort, bounded: the sync must not keep an idle box alive for long.
  if [ -n "${RLORDATA_REPO:-}" ] && [ -d "$RLORDATA_REPO" ]; then
    log "syncing artifacts to ${RLORDATA_ARTIFACTS:-<unset>}"
    # -H: cron runs this as root and `-E` alone kept HOME=/root, so uv (as ubuntu) failed on
    # /root/.cache/uv and NOTHING was synced before terminate (found 2026-09-15).
    ( cd "$RLORDATA_REPO" && timeout 1200 sudo -u "${RLORDATA_USER:-ubuntu}" -E -H env PATH="/home/${RLORDATA_USER:-ubuntu}/.local/bin:$PATH" UV_NO_SYNC=1 \
        uv run python -m rlordata.artifacts sync-all ) 2>&1 | tail -n 20 | while read -r l; do log "$l"; done || log "artifact sync failed (rc=$?)"
  fi
}

terminate_or_shutdown() {
  evidence
  sync_artifacts
  if [ -n "${LAMBDA_API_KEY:-}" ]; then
    local id; id=$(lambda_instance_id)
    if [ -n "$id" ]; then
      log "terminating Lambda instance $id"
      local resp; resp=$(lambda_terminate "$id")
      log "terminate response: $resp"
      if echo "$resp" | grep -q '"terminated_instances"'; then
        exit 0
      fi
      log "TERMINATE FAILED — falling back to shutdown. LAMBDA KEEPS BILLING; terminate from the console!"
    else
      log "could not resolve this instance's id via the Lambda API — falling back to shutdown. LAMBDA KEEPS BILLING!"
    fi
  else
    log "LAMBDA_API_KEY unset — shutting down only. LAMBDA KEEPS BILLING a shut-down instance; terminate from the console!"
  fi
  wall "gpu idle ${IDLE_MINUTES} min: shutting down (NOT terminated — Lambda keeps billing; terminate from the console)" 2>/dev/null || true
  shutdown -h now
}

case "${1:-check}" in
  test-api)
    [ -n "${LAMBDA_API_KEY:-}" ] || { echo "LAMBDA_API_KEY unset in /etc/rlordata/idle.env"; exit 1; }
    id=$(lambda_instance_id)
    if [ -n "$id" ]; then echo "this instance: $id (terminate wiring OK)"; else echo "could not resolve instance id"; exit 1; fi
    exit 0;;
esac

UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
[ -f "$STATE" ] || echo 0 > "$STATE"
if [ "${UTIL:-0}" -lt "$UTIL_THRESHOLD" ]; then
  N=$(( $(cat "$STATE") + 1 )); echo "$N" > "$STATE"
  if [ "$N" -ge "$IDLE_MINUTES" ]; then
    log "gpu idle ${N} min (<${UTIL_THRESHOLD}% util)"
    terminate_or_shutdown
  fi
else
  echo 0 > "$STATE"
fi
INNER
    sudo chmod 755 /usr/local/bin/gpu_idle_check.sh
    # `grep -v` exits 1 on an empty/absent crontab; under `set -e` that used to abort the subshell before the
    # echo, installing an EMPTY root crontab (no guard at all). Hence `|| true`.
    ( { sudo crontab -l 2>/dev/null | grep -v gpu_idle_check || true; } ; echo "* * * * * /usr/local/bin/gpu_idle_check.sh" ) | sudo crontab -
    echo "idle guard installed (${IDLE_MINUTES} min at <${UTIL_THRESHOLD}% util; terminate via Lambda API when LAMBDA_API_KEY is set)"
    echo "verify the wiring now:  bash setup/idle_shutdown.sh test-api";;
  status) cat "$STATE" 2>/dev/null || echo 0;;
  test-api) sudo /usr/local/bin/gpu_idle_check.sh test-api;;
  check-now) sudo /usr/local/bin/gpu_idle_check.sh check;;
  *) echo "usage: $0 install | status | test-api | check-now"; exit 1;;
esac
