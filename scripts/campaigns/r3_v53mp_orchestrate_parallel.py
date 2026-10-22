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
PROGRESS_NAME = "orchestrate_{tier}_parallel.jsonl"
SLOT_LOCK_TEMPLATE = "/tmp/opencode/pes2ts_gpu.slot{k}.lock"


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


def _dry_run(tier: str, output_root: Path, run_ids: list[str],
             concurrency: int) -> int:
    """Emit the full launch SCHEDULE (plan order, slots round-robin).

    Read-only w.r.t. the ledger and the GPU: no subprocess is launched.
    """
    ledger_path = output_root / "run_records.jsonl"
    progress = output_root / PROGRESS_NAME.format(tier=tier)
    pending: list[str] = []
    skipped: list[str] = []
    for rid in run_ids:
        if _ledger_status(ledger_path, rid) == "done":
            skipped.append(rid)
        else:
            pending.append(rid)
    _emit(progress, event="orchestrate_start", tier=tier, n_runs=len(run_ids),
          n_pending=len(pending), concurrency=concurrency, dry_run=True)
    for rid in skipped:
        _emit(progress, event="skip_done", run_id=rid, dry_run=True)
    for order, rid in enumerate(pending):
        slot = order % concurrency
        _emit(progress, event="launch", run_id=rid, slot=slot, order=order,
              concurrency=concurrency, dry_run=True)
    _emit(progress, event="orchestrate_end", tier=tier, done=len(skipped),
          failed=0, total=len(run_ids), pending=len(pending),
          concurrency=concurrency, dry_run=True)
    return 0


def orchestrate_parallel(tier: str, output_root: Path, device: str,
                         run_ids: list[str], concurrency: int,
                         poll_s: int = 10) -> int:
    ledger_path = output_root / "run_records.jsonl"
    progress = output_root / PROGRESS_NAME.format(tier=tier)
    queue: list[str] = []
    for rid in run_ids:
        if _ledger_status(ledger_path, rid) == "done":
            _emit(progress, event="skip_done", run_id=rid)
        else:
            queue.append(rid)
    _emit(progress, event="orchestrate_start", tier=tier, n_runs=len(run_ids),
          n_pending=len(queue), concurrency=concurrency)
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
            proc = subprocess.Popen(
                [sys.executable, str(RUN_DRIVER), "--output-root",
                 str(output_root), "run", "--run-id", rid,
                 "--device", device, "--gpu-lock", str(lock_path)],
                stdout=log_fh, stderr=subprocess.STDOUT, text=True)
            _emit(progress, event="launch", run_id=rid, slot=slot,
                  concurrency=concurrency)
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
        return _dry_run(args.tier, output_root, run_ids, args.concurrency)
    return orchestrate_parallel(args.tier, output_root, args.device, run_ids,
                                args.concurrency, args.poll_s)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
