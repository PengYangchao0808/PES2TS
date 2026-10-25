#!/usr/bin/env python3
"""V5.3-MP tier orchestrator: run one tier's planned runs CONCURRENTLY.

Additive sibling of ``r3_v53mp_orchestrate.py`` (which is strictly
sequential and must stay untouched).  The tier's planned runs are mutually
independent (different arm/seed); the only serialization points are the
orchestrator loop itself and the driver's exclusive GPU flock.  Each of the
N concurrent workers is therefore launched with its own slot lock file
``/tmp/opencode/pes2ts_gpu.slot<k>.lock`` (k = 0..N-1) via the driver's
existing ``--gpu-lock`` flag — no driver change needed.  Ledger appends are
``fcntl.flock``-protected, so concurrent appends are safe.

Semantics (fail-closed, failure-transparent):

* plan file missing/empty            -> exit 2, typed ``PLAN_MISSING`` /
  ``PLAN_EMPTY`` message on stderr;
* ``--concurrency < 1``              -> exit 2, typed ``BAD_CONCURRENCY``;
* run_id already ``done`` in the ledger -> ``skip_done`` event, never relaunched;
* slot lock busy (held by someone else) -> slot skipped this round (the
  driver would refuse anyway);
* worker exit 0 + ledger ``done``    -> ``done`` event, slot freed;
* worker nonzero exit (or exit 0 but ledger not ``done``) -> ``failed``
  event with returncode + output tail, slot freed, next planned run
  launched — never silently retried, never hidden.

One line of JSONL progress per event to
``<output_root>/orchestrate_<tier>_parallel.jsonl``.

Usage::

    python3 scripts/campaigns/r3_v53mp_orchestrate_parallel.py --tier 5k \
        [--output-root outputs/pesn/v53mp] [--device cuda] \
        [--concurrency 3] [--dry-run]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DRIVER = PROJECT_ROOT / "scripts/campaigns/r3_v53mp_run.py"
CGROUP_SCOPE = PROJECT_ROOT / "scripts/infra/cgroup_scope.sh"
PROGRESS_NAME = "orchestrate_{tier}_parallel.jsonl"
SLOT_LOCK_TEMPLATE = "/tmp/opencode/pes2ts_gpu.slot{k}.lock"

#: I3 (plan §5.1/§7-B4): per-run cgroup cap when --per-run-memory-max is
#: set; MemoryHigh is derived as 0.9 x MemoryMax (systemd-style sizes).
MEMORY_HIGH_FACTOR = 0.9


def _derive_memory_high(memory_max: str) -> str:
    """MemoryHigh = 0.9 x MemoryMax, preserving the systemd size suffix.

    ``"6G" -> "5.4G"``, ``"5120M" -> "4608M"``.  Numeric part must parse;
    any junk is a typed refusal.
    """
    s = str(memory_max).strip()
    num, suffix = s[:-1], s[-1:]
    if suffix.upper() not in ("G", "M", "K", "B") or not num:
        raise SystemExit(
            "error: BAD_MEMORY_MAX: {!r} (expected e.g. 6G, 5120M)".format(
                memory_max))
    try:
        value = float(num)
    except ValueError:
        raise SystemExit(
            "error: BAD_MEMORY_MAX: {!r} (expected e.g. 6G, 5120M)".format(
                memory_max)) from None
    high = value * MEMORY_HIGH_FACTOR
    if abs(high - round(high)) < 1e-9:
        high_s = str(int(round(high)))
    else:
        high_s = ("{:.2f}".format(high)).rstrip("0").rstrip(".")
    return high_s + suffix


def _feature_workers_for(mode: str, concurrency: int) -> int | None:
    """Resolve the per-run feature-workers value for a workers mode.

    ``gate`` -> None (the driver's flag-file behaviour, unchanged);
    ``auto`` -> the plan §4.3 formula evaluated at this concurrency.
    """
    if mode == "gate":
        return None
    if mode == "auto":
        from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (
            workers_for_concurrency,
        )
        return workers_for_concurrency(concurrency)
    raise SystemExit(
        "error: BAD_WORKERS_MODE: {!r} (expected gate|auto)".format(mode))


def _planned_run_ids(plan_path: Path) -> list[str]:
    ids = []
    for ln in plan_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        doc = json.loads(ln)
        if doc.get("record_type") == "plan_header":
            continue
        rid = doc.get("run_id")
        if rid:
            ids.append(str(rid))
    return ids


def _ledger_status(ledger_path: Path, plan_run_id: str) -> str | None:
    if not ledger_path.is_file():
        return None
    status = None
    for ln in ledger_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        doc = json.loads(ln)
        if doc.get("plan_run_id") == plan_run_id or doc.get("run_id") == plan_run_id:
            status = doc.get("status")
    return status


def _emit(path: Path, **fields) -> None:
    fields["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = json.dumps(fields, sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _slot_lock_busy(lock_path: Path) -> bool:
    """Non-blocking probe: is another process holding this slot lock?

    True  -> lock busy (skip the slot; the driver would refuse anyway).
    False -> lock currently free (the driver will acquire it itself).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _file_tail(path: Path, n: int = 800) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-n:]


def _slot_command(rid: str, output_root: Path, device: str, slot: int,
                  *, per_run_memory_max: str | None,
                  feature_workers: int | None,
                  eval_vram_gate_mib: int | None = None,
                  eval_vram_gate_timeout_s: int = 600,
                  ) -> tuple[list[str], dict]:
    """Build one slot's launch argv (+ wrapper metadata).

    Base: the driver ``run`` subcommand with a per-slot gpu lock.  When
    ``per_run_memory_max`` is set the whole invocation is wrapped in
    ``cgroup_scope.sh <max> <high> <slot-outdir>`` with a per-slot
    ``CGROUP_SCOPE_UNIT`` (plan §5.1: one runaway run kills only its own
    6G scope).  ``feature_workers`` (from ``--feature-workers-mode``)
    becomes the driver's ``--feature-workers`` when not None.
    ``eval_vram_gate_mib`` (3-way staggered evals) becomes the driver's
    ``--eval-vram-gate-mib`` + ``--eval-vram-gate-timeout-s`` when not
    None; default off (flags absent — argv byte-identical).
    """
    lock_path = Path(SLOT_LOCK_TEMPLATE.format(k=slot))
    argv = [sys.executable, str(RUN_DRIVER), "--output-root",
            str(output_root), "run", "--run-id", rid,
            "--device", device, "--gpu-lock", str(lock_path)]
    if feature_workers is not None:
        argv += ["--feature-workers", str(int(feature_workers))]
    if eval_vram_gate_mib is not None:
        argv += ["--eval-vram-gate-mib", str(int(eval_vram_gate_mib)),
                 "--eval-vram-gate-timeout-s",
                 str(int(eval_vram_gate_timeout_s))]
    meta = {"gpu_lock": str(lock_path), "unit": None, "cgroup_outdir": None,
            "memory_max": None, "memory_high": None,
            "eval_vram_gate_mib": eval_vram_gate_mib}
    if per_run_memory_max:
        high = _derive_memory_high(per_run_memory_max)
        outdir = output_root / "cgroup" / "slot{}".format(slot)
        unit = "v53mp-orch-slot{}-{}".format(slot, rid)
        argv = ["bash", str(CGROUP_SCOPE), str(per_run_memory_max), high,
                str(outdir)] + argv
        meta.update({"unit": unit, "cgroup_outdir": str(outdir),
                     "memory_max": str(per_run_memory_max),
                     "memory_high": high})
    return argv, meta


def _eval_vram_gate_mib_arg(value) -> int:
    """argparse type: whole-card eval VRAM gate threshold in MiB (>= 1)."""
    try:
        mib = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--eval-vram-gate-mib must be an integer >= 1, got {!r}".format(
                value)) from None
    if mib < 1:
        raise argparse.ArgumentTypeError(
            "--eval-vram-gate-mib must be >= 1, got {!r}".format(value))
    return mib


def _eval_vram_gate_timeout_arg(value) -> int:
    """argparse type: eval VRAM gate wait timeout in seconds (>= 0)."""
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--eval-vram-gate-timeout-s must be an integer >= 0, got {!r}"
            .format(value)) from None
    if seconds < 0:
        raise argparse.ArgumentTypeError(
            "--eval-vram-gate-timeout-s must be >= 0, got {!r}".format(value))
    return seconds


def _dry_run(tier: str, output_root: Path, run_ids: list[str],
             concurrency: int, *, per_run_memory_max: str | None = None,
             workers_mode: str = "gate",
             eval_vram_gate_mib: int | None = None,
             eval_vram_gate_timeout_s: int = 600) -> int:
    """Emit the full launch SCHEDULE (plan order, slots round-robin).

    Read-only w.r.t. the ledger and the GPU: no subprocess is launched.
    Validates the cgroup wrapper contract when --per-run-memory-max is set
    (fail-closed on a missing cgroup_scope.sh).
    """
    ledger_path = output_root / "run_records.jsonl"
    progress = output_root / PROGRESS_NAME.format(tier=tier)
    if per_run_memory_max and not CGROUP_SCOPE.is_file():
        sys.stderr.write(
            "error: CGROUP_SCOPE_MISSING: {} (required by "
            "--per-run-memory-max)\n".format(CGROUP_SCOPE))
        return 2
    feature_workers = _feature_workers_for(workers_mode, concurrency)
    pending: list[str] = []
    skipped: list[str] = []
    for rid in run_ids:
        if _ledger_status(ledger_path, rid) == "done":
            skipped.append(rid)
        else:
            pending.append(rid)
    _emit(progress, event="orchestrate_start", tier=tier, n_runs=len(run_ids),
          n_pending=len(pending), concurrency=concurrency, dry_run=True,
          per_run_memory_max=per_run_memory_max,
          feature_workers=feature_workers)
    for rid in skipped:
        _emit(progress, event="skip_done", run_id=rid, dry_run=True)
    for order, rid in enumerate(pending):
        slot = order % concurrency
        _argv, meta = _slot_command(
            rid, output_root, "cuda", slot,
            per_run_memory_max=per_run_memory_max,
            feature_workers=feature_workers,
            eval_vram_gate_mib=eval_vram_gate_mib,
            eval_vram_gate_timeout_s=eval_vram_gate_timeout_s)
        event_fields = {k: v for k, v in meta.items() if v is not None}
        if feature_workers is not None:
            event_fields["feature_workers"] = feature_workers
        _emit(progress, event="launch", run_id=rid, slot=slot, order=order,
              concurrency=concurrency, dry_run=True, **event_fields)
    _emit(progress, event="orchestrate_end", tier=tier, done=len(skipped),
          failed=0, total=len(run_ids), pending=len(pending),
          concurrency=concurrency, dry_run=True)
    return 0


def orchestrate_parallel(tier: str, output_root: Path, device: str,
                         run_ids: list[str], concurrency: int,
                         poll_s: int = 10, *,
                         per_run_memory_max: str | None = None,
                         workers_mode: str = "gate",
                         eval_vram_gate_mib: int | None = None,
                         eval_vram_gate_timeout_s: int = 600) -> int:
    if per_run_memory_max and not CGROUP_SCOPE.is_file():
        sys.stderr.write(
            "error: CGROUP_SCOPE_MISSING: {} (required by "
            "--per-run-memory-max)\n".format(CGROUP_SCOPE))
        return 2
    feature_workers = _feature_workers_for(workers_mode, concurrency)
    ledger_path = output_root / "run_records.jsonl"
    progress = output_root / PROGRESS_NAME.format(tier=tier)
    queue: list[str] = []
    for rid in run_ids:
        if _ledger_status(ledger_path, rid) == "done":
            _emit(progress, event="skip_done", run_id=rid)
        else:
            queue.append(rid)
    _emit(progress, event="orchestrate_start", tier=tier, n_runs=len(run_ids),
          n_pending=len(queue), concurrency=concurrency,
          per_run_memory_max=per_run_memory_max,
          feature_workers=feature_workers)
    # slot -> (proc, run_id, open log handle)
    active: dict[int, tuple[subprocess.Popen, str, object]] = {}
    n_failed = 0
    while queue or active:
        # reap finished workers
        for slot in list(active):
            proc, rid, log_fh = active[slot]
            if proc.poll() is None:
                continue
            log_fh.close()
            tail = _file_tail(output_root /
                              "orchestrate_{}_parallel_{}.log".format(tier, rid))
            status = _ledger_status(ledger_path, rid)
            if proc.returncode == 0 and status == "done":
                _emit(progress, event="done", run_id=rid, slot=slot,
                      concurrency=concurrency)
            else:
                n_failed += 1
                _emit(progress, event="failed", run_id=rid, slot=slot,
                      concurrency=concurrency, returncode=proc.returncode,
                      ledger_status=status, tail=tail)
            del active[slot]
        # launch the next planned runs into free, non-busy slots
        for slot in range(concurrency):
            if not queue:
                break
            if slot in active:
                continue
            lock_path = Path(SLOT_LOCK_TEMPLATE.format(k=slot))
            if _slot_lock_busy(lock_path):
                continue  # slot busy: skip; the driver would refuse anyway
            rid = queue.pop(0)
            log_path = output_root / \
                "orchestrate_{}_parallel_{}.log".format(tier, rid)
            log_fh = log_path.open("w", encoding="utf-8")
            argv, meta = _slot_command(
                rid, output_root, device, slot,
                per_run_memory_max=per_run_memory_max,
                feature_workers=feature_workers,
                eval_vram_gate_mib=eval_vram_gate_mib,
                eval_vram_gate_timeout_s=eval_vram_gate_timeout_s)
            env = dict(os.environ)
            if meta.get("unit"):
                env["CGROUP_SCOPE_UNIT"] = meta["unit"]
            proc = subprocess.Popen(
                argv, stdout=log_fh, stderr=subprocess.STDOUT, text=True,
                env=env)
            event_fields = {k: v for k, v in meta.items() if v is not None}
            if feature_workers is not None:
                event_fields["feature_workers"] = feature_workers
            _emit(progress, event="launch", run_id=rid, slot=slot,
                  concurrency=concurrency, **event_fields)
            active[slot] = (proc, rid, log_fh)
        if queue or active:
            time.sleep(poll_s)
    done = sum(1 for rid in run_ids
               if _ledger_status(ledger_path, rid) == "done")
    _emit(progress, event="orchestrate_end", tier=tier, done=done,
          failed=n_failed, total=len(run_ids), concurrency=concurrency)
    return 0 if done == len(run_ids) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="r3_v53mp_orchestrate_parallel")
    p.add_argument("--tier", required=True, choices=("2k", "5k", "20k"))
    p.add_argument("--output-root", default="outputs/pesn/v53mp")
    p.add_argument("--device", default="cuda")
    p.add_argument("--concurrency", type=int, default=1,
                   help="max concurrent driver workers (default 1)")
    p.add_argument("--poll-s", type=int, default=10)
    p.add_argument("--per-run-memory-max", default=None,
                   help="wrap each slot's driver run in cgroup_scope.sh "
                        "with MemoryMax=<value> / MemoryHigh=0.9x value "
                        "(plan §5.1, e.g. 6G; default off = uncapped)")
    p.add_argument("--feature-workers-mode", default="gate",
                   choices=("gate", "auto"),
                   help="gate = driver flag-file behaviour (default); "
                        "auto = --feature-workers "
                        "workers_for_concurrency(--concurrency) per slot "
                        "(plan §4.3/B4)")
    p.add_argument("--eval-vram-gate-mib", type=_eval_vram_gate_mib_arg,
                   default=None,
                   help="whole-card MiB threshold forwarded to every slot's "
                        "driver argv as '--eval-vram-gate-mib mib "
                        "--eval-vram-gate-timeout-s s' (3-way staggered "
                        "evals; default off — flags absent)")
    p.add_argument("--eval-vram-gate-timeout-s",
                   type=_eval_vram_gate_timeout_arg, default=600,
                   help="max seconds each slot's trainer waits per gated "
                        "eval (fail-open; default 600; forwarded only when "
                        "--eval-vram-gate-mib is set)")
    p.add_argument("--dry-run", action="store_true",
                   help="print/emit the launch schedule without launching")
    args = p.parse_args(argv)

    if args.concurrency < 1:
        sys.stderr.write(
            "error: BAD_CONCURRENCY: --concurrency must be >= 1 (got {})\n"
            .format(args.concurrency))
        return 2

    output_root = Path(args.output_root)
    plan_path = output_root / "run_plan_v53mp_{}.jsonl".format(args.tier)
    if not plan_path.is_file():
        sys.stderr.write("error: PLAN_MISSING: {}\n".format(plan_path))
        return 2
    run_ids = _planned_run_ids(plan_path)
    if not run_ids:
        sys.stderr.write("error: PLAN_EMPTY: {}\n".format(plan_path))
        return 2

    if args.dry_run:
        return _dry_run(args.tier, output_root, run_ids, args.concurrency,
                        per_run_memory_max=args.per_run_memory_max,
                        workers_mode=args.feature_workers_mode,
                        eval_vram_gate_mib=args.eval_vram_gate_mib,
                        eval_vram_gate_timeout_s=args.eval_vram_gate_timeout_s)
    return orchestrate_parallel(args.tier, output_root, args.device, run_ids,
                                args.concurrency, args.poll_s,
                                per_run_memory_max=args.per_run_memory_max,
                                workers_mode=args.feature_workers_mode,
                                eval_vram_gate_mib=args.eval_vram_gate_mib,
                                eval_vram_gate_timeout_s=(
                                    args.eval_vram_gate_timeout_s))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
