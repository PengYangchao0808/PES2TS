#!/usr/bin/env bash
# S4 infra: run a command inside a transient cgroup-v2 scope with a HARD
# memory cap + a 2 s memory sampler (plan §5.1).  The systemd path mirrors
# scripts/campaigns/r3_v53mp_cgroup_run.sh EXACTLY (same scope location,
# same systemd-run properties incl. MemorySwapMax=0, same sampler CSV
# format) so bench and campaign runs are directly comparable.
#
# Usage
# -----
#   cgroup_scope.sh [--fallback] <MemoryMax> <MemoryHigh> <outdir> <cmd ...>
#
#   --fallback     skip systemd-run; create a manual cgroup-v2 dir under
#                  $CGROUP_V2_PARENT (default /sys/fs/cgroup) and move the
#                  command into it.  Use when systemd is absent.
#   MemoryMax / MemoryHigh   systemd-style sizes ("34G", "32G", "6G")
#   outdir                   sampler output dir
#   cmd ...                  command to run inside the scope
#
# Environment knobs (optional)
# ----------------------------
#   CGROUP_SCOPE_UNIT        scope/unit name (default: infra-scope-$$)
#   CGROUP_SCOPE_INTERVAL    sample period, s (default: 2)
#   CGROUP_V2_PARENT         --fallback parent cgroup dir (default /sys/fs/cgroup)
#
# Sampler output (r3_v53mp_cgroup_run.sh-compatible; the first FOUR columns
# are byte-identical to that format; column 5 memory_swap_current_bytes is
# an additive extension for the plan §5.1 swap gate — 4-column r3 CSVs
# remain parseable by scripts/infra/infra_bench.py):
#   <outdir>/cgroup_sample.csv          ts,memory_current_bytes,anon_bytes,file_bytes,memory_swap_current_bytes
#   <outdir>/cgroup_sample.csv.peak     ts,peak_memory_current,anon_at_peak   (on each new peak)
#   <outdir>/cgroup_sample.csv.events   raw memory.events snapshot (kept current while the scope lives;
#                                       the cgroup vanishes on exit, so the copy is the evidence)
#   <outdir>/cgroup_summary.json        machine-readable gate summary
#
# Exit code (plan §5.1 gates)
# --------------------------
#   the wrapped command's own exit code (137 == in-scope OOM kill; note
#   systemd-run re-raises the child's fatal signal);
#   91  command succeeded but memory.events oom_kill > 0;
#   90  command succeeded but memory.swap.current stayed > 0 for > 60 s
#       (only reachable if MemorySwapMax was bypassed; with the default
#       MemorySwapMax=0 swap is structurally impossible inside the scope
#       and overflow surfaces as oom_kill instead — "靠 swap 跑完" is
#       never accepted, plan §5.1);
#   0   clean.
#
# This kernel (WSL2 5.15) has NO cgroup-v2 memory.peak and NO PSI
# (/proc/pressure/memory absent) — hence the self-sampled peak.
set -euo pipefail

FALLBACK=0
if [[ "${1:-}" == "--fallback" ]]; then
  FALLBACK=1
  shift
fi

if [[ $# -lt 4 ]]; then
  echo "usage: cgroup_scope.sh [--fallback] <MemoryMax> <MemoryHigh> <outdir> <cmd ...>" >&2
  exit 2
fi

MAX="$1"
HIGH="$2"
OUTDIR="$3"
shift 3

UNIT="${CGROUP_SCOPE_UNIT:-infra-scope-$$}"
INTERVAL="${CGROUP_SCOPE_INTERVAL:-2}"
PARENT="${CGROUP_V2_PARENT:-/sys/fs/cgroup}"

mkdir -p "$OUTDIR"
CSV="$OUTDIR/cgroup_sample.csv"
SUMMARY="$OUTDIR/cgroup_summary.json"
printf 'ts,memory_current_bytes,anon_bytes,file_bytes,memory_swap_current_bytes\n' > "$CSV"
: > "${CSV}.peak"
: > "${CSV}.events"

if [[ "$FALLBACK" -eq 1 ]]; then
  if [[ ! -d "$PARENT" || ! -w "$PARENT" ]]; then
    echo "error: NO_CGROUP_PARENT: $PARENT missing/not writable" >&2
    exit 2
  fi
  # root cgroup needs the memory controller in subtree_control (already true
  # on this box); try once, tolerate "Device or resource busy"
  echo "+memory" > "$PARENT/cgroup.subtree_control" 2>/dev/null || true
  CG="$PARENT/${UNIT}"
  mkdir -p "$CG"
  echo "$MAX" > "$CG/memory.max"
  echo "$HIGH" > "$CG/memory.high"
  echo 0 > "$CG/memory.swap.max" 2>/dev/null || true
else
  if ! command -v systemd-run >/dev/null 2>&1; then
    echo "error: SYSTEMD_RUN_MISSING: systemd-run not found (use --fallback for a manual cgroup)" >&2
    exit 2
  fi
  CG="/sys/fs/cgroup/system.slice/${UNIT}.scope"
  # a stale failed unit with the same name would make systemd-run refuse
  systemctl reset-failed "${UNIT}.scope" >/dev/null 2>&1 || true
fi

sampler() {
  # survives a torn read: every field read is `|| echo 0`-guarded, and the
  # scope dir vanishes on exit, so late iterations are cheap no-ops
  while true; do
    if [[ -r "$CG/memory.current" ]]; then
      local cur swap anon file oom_kill
      cur=$(cat "$CG/memory.current" 2>/dev/null || echo 0)
      swap=$(cat "$CG/memory.swap.current" 2>/dev/null || echo 0)
      anon=$(awk '/^anon /{print $2}' "$CG/memory.stat" 2>/dev/null || echo 0)
      file=$(awk '/^file /{print $2}' "$CG/memory.stat" 2>/dev/null || echo 0)
      oom_kill=$(awk '/^oom_kill /{print $2}' "$CG/memory.events" 2>/dev/null || echo 0)
      printf '%s,%s,%s,%s,%s\n' "$(date -u +%H:%M:%S)" "$cur" "$anon" "$file" "$swap" >> "$CSV"
      if (( cur > 0 )); then
        local peak
        peak=$(cut -d, -f2 "${CSV}.peak" 2>/dev/null | tail -1)
        peak="${peak:-0}"
        if (( cur > peak )); then
          printf '%s,%s,%s\n' "$(date -u +%H:%M:%S)" "$cur" "$anon" > "${CSV}.peak"
        fi
      fi
      # keep the last-seen oom counters: the cgroup disappears when the
      # scope exits, so a copy after the fact would be empty
      cat "$CG/memory.events" > "${CSV}.events" 2>/dev/null || true
    fi
    sleep "$INTERVAL"
  done
}

sampler &
SAMPLER_PID=$!
cleanup() {
  kill "$SAMPLER_PID" 2>/dev/null || true
  wait "$SAMPLER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[cgroup_scope] fallback=$FALLBACK unit=$UNIT max=$MAX high=$HIGH outdir=$OUTDIR" >&2

RC=0
if [[ "$FALLBACK" -eq 1 ]]; then
  # move THIS subshell into the manual scope, then exec the command.
  # ($$ in a subshell is the PARENT pid; $BASHPID is the subshell itself,
  #  and exec preserves the pid, so the command lands in the scope)
  ( echo "$BASHPID" > "$CG/cgroup.procs"; exec "$@" ) || RC=$?
else
  systemd-run --scope --quiet \
    -p MemoryMax="$MAX" \
    -p MemoryHigh="$HIGH" \
    -p MemorySwapMax=0 \
    --unit="$UNIT" -- "$@" || RC=$?
fi

# final accounting (scope may already be gone; the CSV+events are our
# evidence).  A very short-lived/OOM-fast scope can leave the sampler with
# almost nothing: every extraction must be pipefail/set-e-safe — the
# trailing `|| true` wraps the WHOLE pipeline (a `|| true` on one member
# does not protect the assignment under pipefail).
OOM_KILL=$(awk '/^oom_kill /{print $2}' "${CSV}.events" 2>/dev/null | tail -1 || true)
OOM_KILL="${OOM_KILL:-0}"
PEAK_ANON=$(cut -d, -f3 "$CSV" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
MAX_SWAP=$(cut -d, -f5 "$CSV" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
PEAK_ANON="${PEAK_ANON:-0}"
MAX_SWAP="${MAX_SWAP:-0}"

# sustained swap: longest consecutive run of swap>0 samples, in seconds
SWAP_SUSTAINED=$(awk -F, -v iv="$INTERVAL" '
  NR > 1 && $5 + 0 > 0 { run++; if (run > maxrun) maxrun = run }
  NR > 1 && $5 + 0 == 0 { run = 0 }
  END { if (maxrun > 1) printf "%d", (maxrun - 1) * iv; else print 0 }' \
  "$CSV" 2>/dev/null || echo 0)
SWAP_SUSTAINED="${SWAP_SUSTAINED:-0}"

GATE_RC=0
if [[ "$RC" -eq 0 && "$OOM_KILL" -gt 0 ]]; then
  GATE_RC=91
elif [[ "$RC" -eq 0 && "$SWAP_SUSTAINED" -gt 60 ]]; then
  GATE_RC=90
fi

# manual-fallback scope: remove the dir (fails only if a process is stuck
# inside; warn instead of masking the exit code)
if [[ "$FALLBACK" -eq 1 && -d "$CG" ]]; then
  rmdir "$CG" 2>/dev/null || echo "[cgroup_scope] warn: could not rmdir $CG" >&2
fi

printf '{"peak_anon_bytes":%s,"max_swap_current_bytes":%s,"swap_sustained_s":%s,"oom_kill":%s,"cmd_exit":%s,"gate_exit":%s,"interval_s":%s,"unit":"%s","fallback":%s}\n' \
  "$PEAK_ANON" "$MAX_SWAP" "$SWAP_SUSTAINED" "$OOM_KILL" "$RC" "$GATE_RC" \
  "$INTERVAL" "$UNIT" "$FALLBACK" > "$SUMMARY"

echo "[cgroup_scope] exit=$RC gate_exit=$GATE_RC peak_anon=$PEAK_ANON max_swap=$MAX_SWAP swap_sustained_s=$SWAP_SUSTAINED oom_kill=$OOM_KILL" >&2
cat "$SUMMARY" >&2

if [[ "$RC" -ne 0 ]]; then
  exit "$RC"
fi
exit "$GATE_RC"
