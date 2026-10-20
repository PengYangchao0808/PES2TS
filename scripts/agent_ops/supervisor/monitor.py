"""Supervisor run monitor: launch supervision that NEVER crashes on plan drift.

WP-L3 residual #3 (E0 sign-off report §5): the supervisor/monitor path must
catch :class:`PlanNotFrozenError` and record ``blocked_with_reason`` on the
affected run record instead of crashing -- a drifted/tampered/unfrozen plan
is a *state* of the campaign, not a reason to lose the monitor loop.

Design rules (fail-closed, non-destructive):

* :func:`attempt_launch` is the single supervised launch funnel: it performs
  the full pre-flight (plan pin re-verification + stage-g0 five-sha binding
  when requested) and then appends the ``running`` transition through
  :class:`~scripts.agent_ops.ledger.run_ledger.RunLedger`.  Every
  :class:`PlanNotFrozenError` (pin mismatch, unfrozen plan line, run_id
  mismatch, stage binding failure) is converted into a *blocked* outcome:
  the run record is annotated with ``blocked_with_reason`` via
  :meth:`RunLedger.mark_blocked` and a structured report is returned -- the
  monitor never propagates the exception to its caller loop.
* Other ledger errors (budget cap, retry cap, illegal transition, unseal
  exhaustion, conflicts) are NOT swallowed: they are either caller errors or
  conditions the caller must handle explicitly.  Only plan-freeze conditions
  are monitor-captured (that is the residual's contract).
* :func:`monitor_scan` iterates the ledger's runs and reports the latest
  state plus any block annotations; it never mutates anything except through
  the explicit supervised launch path.

CLI::

    python -m scripts.agent_ops.supervisor.monitor scan --ledger ... --plan ...
    python -m scripts.agent_ops.supervisor.monitor launch --ledger ... --plan ... \
        --record-json '{...}' [--stage g0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.agent_ops.ledger.run_ledger import (
    CAMPAIGN_PIN,
    LedgerError,
    PlanNotFrozenError,
    RunLedger,
    compute_run_id,
    load_campaign_pin,
    verify_plan_pin,
)
from scripts.agent_ops.supervisor.stage_bind import (
    DEFAULT_TASK_STATE_PATH,
    StageBindingError,
    assert_stage_binding,
)

__all__ = [
    "LaunchReport",
    "attempt_launch",
    "main",
    "monitor_scan",
]


class LaunchReport(Dict):  # type: ignore[misc]
    """Structured supervised-launch outcome (a plain dict subclass)."""


def _resolve_pin(plan_pin: Any, task_state: Optional[Any]) -> Any:
    """Resolve the pin argument for a supervised launch (fail-closed)."""
    if plan_pin is None:
        return None
    if plan_pin is CAMPAIGN_PIN:
        return load_campaign_pin(task_state or DEFAULT_TASK_STATE_PATH)
    return plan_pin


def attempt_launch(
    ledger: RunLedger,
    record: Dict[str, Any],
    *,
    stage: Optional[str] = None,
    task_state: Optional[Any] = None,
) -> LaunchReport:
    """Supervised launch of one run: never crashes on :class:`PlanNotFrozenError`.

    Steps, in order (all fail-closed):
      1. if the ledger has a bound plan + pin, re-verify the pin;
      2. if ``stage`` is given (``g0``/``launch``), assert the five identity
         shas of the plan line for this run are real 64-hex digests
         (:class:`StageBindingError`, a ``PlanNotFrozenError`` subclass);
      3. append the ``running`` transition through the ledger (budget caps,
         batch unseals, PD-4 rules all still apply).

    On any ``PlanNotFrozenError`` the run record is annotated with
    ``blocked_with_reason`` (status unchanged) and the report says
    ``state="blocked"``.  Non-plan ledger errors propagate to the caller.
    """
    rec = dict(record)
    rec["status"] = "running"
    rec.setdefault("pid", os.getpid())
    run_id = compute_run_id(rec, ledger.run_id_scheme)
    report = LaunchReport(
        {
            "run_id": run_id,
            "state": "blocked",
            "action": None,
            "blocked_with_reason": None,
            "stage": stage,
        }
    )
    try:
        if ledger.plan_path is not None and ledger.plan_pin is not None:
            verify_plan_pin(ledger.plan_path, ledger.plan_pin)
        if stage is not None and ledger.plan_path is not None:
            # whole-plan binding: any line still carrying PENDING/zero shas
            # means the plan is not launch-ready (per-run membership is then
            # enforced by the ledger's own plan cross-check, which distinguishes
            # NOT_IN_PLAN / RUN_ID_MISMATCH / PLAN_NOT_FROZEN)
            assert_stage_binding(
                ledger.plan_path,
                stage=stage,
                task_state=task_state,
            )
        outcome = ledger.append(rec)
    except PlanNotFrozenError as exc:
        # WP-L3 residual #3: record the block on the run record; never crash.
        try:
            ledger.mark_blocked(run_id, str(exc))
            report["state"] = "blocked"
        except LedgerError:
            # run not yet registered: still a blocked outcome, just un-annotated
            report["state"] = "blocked_unregistered"
        report["blocked_with_reason"] = str(exc)
        return report
    report["action"] = outcome.action
    report["state"] = str(outcome.entry.get("status"))
    report["blocked_with_reason"] = None
    return report


def monitor_scan(
    ledger: RunLedger,
    *,
    plan_pin: Any = CAMPAIGN_PIN,
    task_state: Optional[Any] = None,
) -> Dict[str, Any]:
    """Read-only monitor scan: latest state per run + block annotations.

    Re-verifies the bound plan pin (when pinned) and reports
    ``plan_frozen`` so a drift is visible without mutating anything.
    ``plan_pin=CAMPAIGN_PIN`` (default) resolves the ledger's OWN bound pin
    (falling back to the task_state-derived pin); pass an explicit dict to
    override, or ``None`` to skip the check for fixture ledgers.
    """
    plan_frozen: Optional[bool] = None
    plan_error: Optional[str] = None
    if ledger.plan_path is not None:
        if plan_pin is CAMPAIGN_PIN:
            pin: Any = ledger.plan_pin
            if pin is None:
                pin = _resolve_pin(CAMPAIGN_PIN, task_state)
        else:
            pin = _resolve_pin(plan_pin, task_state)
        if pin is None:
            plan_frozen = None  # explicitly unpinned (fixture)
        else:
            try:
                verify_plan_pin(ledger.plan_path, pin)
                plan_frozen = True
            except PlanNotFrozenError as exc:
                plan_frozen = False
                plan_error = str(exc)
    runs: List[Dict[str, Any]] = []
    for run_id, entry in sorted(ledger.read_all().items()):
        runs.append(
            {
                "run_id": run_id,
                "status": entry.get("status"),
                "phase": entry.get("phase"),
                "arm": entry.get("arm"),
                "blocked_with_reason": entry.get("blocked_with_reason"),
            }
        )
    return {
        "plan_frozen": plan_frozen,
        "plan_error": plan_error,
        "runs": runs,
        "run_count": len(runs),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supervisor_monitor",
        description="Supervised launch + read-only run scan; PlanNotFrozen "
        "conditions become blocked_with_reason annotations, never crashes",
    )
    parser.add_argument("--ledger", required=True, help="path to run_records.jsonl")
    parser.add_argument(
        "--plan", default=None, help="bound run_plan.jsonl (optional but expected)"
    )
    parser.add_argument(
        "--task-state",
        default=DEFAULT_TASK_STATE_PATH,
        help="task_state.json carrying e0_freeze.run_plan_pinned",
    )
    parser.add_argument(
        "--plan-pin-file",
        default=None,
        help="explicit pin JSON ({sha256, campaign_mode}) instead of the "
        "task_state-derived campaign pin (fixture use)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan", help="read-only report of run states + plan freeze")

    launch = sub.add_parser("launch", help="supervised launch (never crashes on plan drift)")
    launch.add_argument("--record", help="path to a run_record JSON file")
    launch.add_argument(
        "--record-json", help="inline run_record JSON (status is forced to running)"
    )
    launch.add_argument(
        "--stage",
        choices=("g0", "launch"),
        default=None,
        help="also require all five identity shas of the plan line to be real "
        "64-hex digests before launching (stage-g0 binding)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    plan_pin: Any = CAMPAIGN_PIN if args.plan is not None else None
    if args.plan_pin_file is not None:
        try:
            plan_pin = json.loads(
                Path(args.plan_pin_file).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            sys.stderr.write(
                "error: PLAN_PIN_SOURCE_CORRUPT: {}: {}\n".format(args.plan_pin_file, exc)
            )
            return 2
    ledger = RunLedger(
        args.ledger,
        plan_path=args.plan,
        plan_pin=plan_pin,
    )
    try:
        if args.cmd == "scan":
            print(
                json.dumps(
                    monitor_scan(
                        ledger,
                        plan_pin=plan_pin,
                        task_state=args.task_state,
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if args.cmd == "launch":
            if bool(args.record) == bool(args.record_json):
                sys.stderr.write("error: pass exactly one of --record/--record-json\n")
                return 2
            raw = (
                Path(args.record).read_text(encoding="utf-8")
                if args.record
                else args.record_json
            )
            record = json.loads(raw)
            report = attempt_launch(
                ledger, record, stage=args.stage, task_state=args.task_state
            )
            print(json.dumps(report, sort_keys=True))
            # blocked is a REPORTED outcome (exit 3), not a crash (exit stays 0/3)
            return 0 if report["state"] in ("running", "done") else 3
        return 2
    except LedgerError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    except (json.JSONDecodeError, OSError) as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
