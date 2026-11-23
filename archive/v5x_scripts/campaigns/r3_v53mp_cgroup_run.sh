#!/usr/bin/env bash
# V5.3-MP: run a campaign command inside a transient systemd scope with a HARD
# memory cap, sampling the cgroup's memory.current / anon / file to a CSV that
# lives on disk (not tmpfs, so it survives a VM restart).
#
# Why this exists
# ---------------
# 2026-10-22 (twice): concurrent 20k drivers exhausted memory and the VM died.
# The second time the guest cgroup was at a 50 GiB cap with oom_kill=0 while
# guest anon was 37.0 GiB — i.e. the HOST ran out, which no guest cgroup can
# prevent. Defaults are therefore 34 GiB / 32 GiB: one 20k run peaks at
# 22.6 GiB, and the guest must stay well under the ~37 GiB host cliff. Two things were missing:
#   1. a hard limit, so the damage is "OOM-kill inside the scope", never the VM;
#   2. the real per-run peak — this kernel (5.15) has no cgroup v2
#      `memory.peak`, so we sample `memory.current` ourselves.
#
# Usage
# -----
#   r3_v53mp_cgroup_run.sh --max 46G --high 42G --csv <path.csv> --unit <name> \
#       -- <command ...>
#
#   --max     MemoryMax  (hard cap; OOM-kill only inside the scope)
#   --high    MemoryHigh (soft cap; kernel reclaims cache above this)
#   --csv     sampler output (csv + .peak + .events siblings)
#   --unit    transient scope name (default v53mp-campaign)
#   --interval sampler period in seconds (default 2)
#
# Exit code is the wrapped command's exit code; 137 usually means cgroup OOM.
set -uo pipefail

MAX="34G"
HIGH="32G"
CSV=""
UNIT="v53mp-campaign"
INTERVAL=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max) MAX="$2"; shift 2 ;;
    --high) HIGH="$2"; shift 2 ;;
    --csv) CSV="$2"; shift 2 ;;
    --unit) UNIT="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "error: no command given (use -- <cmd ...>)" >&2
  exit 2
fi
if [[ -z "$CSV" ]]; then
  echo "error: --csv is required (peak evidence must land on disk)" >&2
  exit 2
fi

CG="/sys/fs/cgroup/system.slice/${UNIT}.scope"
mkdir -p "$(dirname "$CSV")"
printf 'ts,memory_current_bytes,anon_bytes,file_bytes\n' > "$CSV"
: > "${CSV}.peak"
: > "${CSV}.events"

# a stale failed unit with the same name would make systemd-run refuse
systemctl reset-failed "${UNIT}.scope" >/dev/null 2>&1 || true

sampler() {
  local peak=0
  while true; do
    if [[ -r "$CG/memory.current" ]]; then
      local cur anon file
      cur=$(cat "$CG/memory.current" 2>/dev/null || echo 0)
      anon=$(awk '/^anon /{print $2}' "$CG/memory.stat" 2>/dev/null || echo 0)
      file=$(awk '/^file /{print $2}' "$CG/memory.stat" 2>/dev/null || echo 0)
      printf '%s,%s,%s,%s\n' "$(date -u +%H:%M:%S)" "$cur" "$anon" "$file" >> "$CSV"
      if (( cur > peak )); then
        peak="$cur"
        printf '%s,%s,%s\n' "$(date -u +%H:%M:%S)" "$peak" "$anon" > "${CSV}.peak"
      fi
      # keep the last-seen oom counters: the cgroup disappears when the scope
      # exits, so a copy after the fact would be empty
      cat "$CG/memory.events" > "${CSV}.events" 2>/dev/null || true
    fi
    sleep "$INTERVAL"
  done
}

sampler &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null' EXIT

echo "[cgroup] max=$MAX high=$HIGH unit=$UNIT csv=$CSV" >&2
systemd-run --scope --quiet \
  -p MemoryMax="$MAX" \
  -p MemoryHigh="$HIGH" \
  -p MemorySwapMax=0 \
  --unit="$UNIT" -- "$@"
RC=$?

# final accounting (memory.events: oom / oom_kill counters)
if [[ -r "$CG/memory.events" ]]; then
  cp "$CG/memory.events" "${CSV}.events" 2>/dev/null || true
fi
printf '\n[cgroup] exit=%s peak=%s\n' "$RC" "$(cat "${CSV}.peak" 2>/dev/null)" >&2
exit "$RC"
