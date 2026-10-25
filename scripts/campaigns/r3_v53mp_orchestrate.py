#!/usr/bin/env python3
"""V5.3-MP tier orchestrator: run one tier's planned runs sequentially.

Runs are serialized on the single-GPU flock (the run driver refuses with
exit 1 + "GPU lock busy" while another run holds it); this orchestrator
waits and retries in that case, advances on "done", and records genuine
failures without ever hiding them.  One line of JSONL progress per event
to ``<output_root>/orchestrate_<tier>.jsonl``.

Usage::

    python3 scripts/campaigns/r3_v53mp_orchestrate.py --tier 5k \
        [--output-root outputs/pesn/v53mp] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DRIVER = PROJECT_ROOT / "scripts/campaigns/r3_v53mp_run.py"
PROGRESS_NAME = "orchestrate_{tier}.jsonl"


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
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(fields, sort_keys=True) + "\n")


def orchestrate(tier: str, output_root: Path, device: str,
                poll_s: int = 60) -> int:
    plan_path = output_root / f"run_plan_v53mp_{tier}.jsonl"
    ledger_path = output_root / "run_records.jsonl"
    progress = output_root / PROGRESS_NAME.format(tier=tier)
    run_ids = _planned_run_ids(plan_path)
    _emit(progress, event="orchestrate_start", tier=tier, n_runs=len(run_ids))
    for rid in run_ids:
        status = _ledger_status(ledger_path, rid)
        if status == "done":
            _emit(progress, event="skip_done", run_id=rid)
            continue
        while True:
            status = _ledger_status(ledger_path, rid)
            if status == "done":
                break
            proc = subprocess.run(
                [sys.executable, str(RUN_DRIVER), "--output-root",
                 str(output_root), "run", "--run-id", rid,
                 "--device", device],
                capture_output=True, text=True)
            out = (proc.stdout or "") + (proc.stderr or "")
            status = _ledger_status(ledger_path, rid)
            if status == "done":
                _emit(progress, event="done", run_id=rid)
                break
            if "GPU lock busy" in out:
                _emit(progress, event="wait_gpu_lock", run_id=rid)
                time.sleep(poll_s)
                continue
            if proc.returncode == 0 and status == "done":
                break
            # genuine failure (recorded in the ledger by the driver)
            _emit(progress, event="failed", run_id=rid,
                  returncode=proc.returncode, tail=out[-800:])
            break
        if _ledger_status(ledger_path, rid) != "done":
            _emit(progress, event="run_not_done_continuing", run_id=rid)
    done = sum(1 for rid in run_ids
               if _ledger_status(ledger_path, rid) == "done")
    _emit(progress, event="orchestrate_end", tier=tier, done=done,
          total=len(run_ids))
    return 0 if done == len(run_ids) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="r3_v53mp_orchestrate")
    p.add_argument("--tier", required=True,
                   choices=("2k", "5k", "20k", "50k"))
    p.add_argument("--output-root", default="outputs/pesn/v53mp")
    p.add_argument("--device", default="cuda")
    p.add_argument("--poll-s", type=int, default=60)
    args = p.parse_args(argv)
    return orchestrate(args.tier, Path(args.output_root), args.device,
                       args.poll_s)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
