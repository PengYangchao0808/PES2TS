"""Stage-2 identity-sha binding check: all five SHAs real before any launch.

WP-L3 residual #5 (small): a supervisor-only verify path ``--stage g0`` that
requires the five identity shas (``config/data/teacher/features/split``) of a
planned run line to be REAL 64-hex digests -- no ``PENDING_E1``/``PENDING_E2``
tokens, no all-zeros, no malformed hex -- before any launch is attempted.

This is the *launch-time* projection of the two-stage binding contract already
declared by ``runplan/verify.py --stage g0`` (which verifies the WHOLE plan);
the supervisor needs the per-run check so a single launch can be gated without
re-verifying 99 lines on every attempt.  It reuses the same classification
(:func:`runplan.build_r3_runplan.classify_sha`) so the two can never disagree.

Semantics (fail-closed):

* ``stage="g0"`` / ``stage="launch"``: every one of the five shas on the
  matching plan line (or every line when ``run_id`` is None) must classify as
  ``"real"``.  Anything else -> :class:`StageBindingError` (a
  :class:`~scripts.agent_ops.ledger.run_ledger.PlanNotFrozenError` subclass,
  so the supervisor monitor captures it as ``blocked_with_reason``).
* ``stage="e0"``: NOT a launch stage -- the e0 plan legitimately carries
  ``PENDING_*`` tokens; asking to "bind" at e0 is a caller bug and refuses.
* missing plan / unknown run_id / corrupt header -> always refuse.

Do NOT bind real E1 SHAs here: the supervisor fills them in after E1 lands
(task_state ``e0_freeze.two_stage``); this module only ENFORCES the format
and frozenness of whatever the plan carries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.agent_ops.ledger.run_ledger import PlanNotFrozenError, read_run_plan
from scripts.agent_ops.runplan.build_r3_runplan import SHA_FIELDS, classify_sha

__all__ = [
    "DEFAULT_PLAN_SUBPATH",
    "DEFAULT_TASK_STATE_PATH",
    "StageBindingError",
    "assert_stage_binding",
    "main",
]

DEFAULT_PLAN_SUBPATH = "outputs/pesn/30h_campaign/run_plan.jsonl"
DEFAULT_TASK_STATE_PATH = "docs/agent_runs/state/task_state.json"
LAUNCH_STAGES = ("g0", "launch")


class StageBindingError(PlanNotFrozenError):
    """Stage-g0 five-sha binding not satisfied for launch."""


def assert_stage_binding(
    plan_path: Any,
    *,
    stage: str = "g0",
    run_id: Optional[str] = None,
    task_state: Optional[Any] = None,  # reserved: E1 re-pin source, unused now
) -> Dict[str, Any]:
    """Require all five identity shas to be real 64-hex before launch.

    ``run_id=None`` checks EVERY line of the plan; otherwise only the line
    whose ``run_id`` matches.  Returns a summary dict on success; raises
    :class:`StageBindingError` on any non-real sha or unusable plan.
    """
    if stage not in LAUNCH_STAGES:
        raise StageBindingError(
            "STAGE_NOT_LAUNCHABLE: stage {!r} cannot gate a launch (only {}); "
            "stage e0 legitimately carries PENDING_E1/PENDING_E2 tokens -- "
            "rebuild the plan with real shas after E1/E2 land".format(
                stage, "/".join(LAUNCH_STAGES)
            )
        )
    path = Path(plan_path)
    if not path.is_file():
        raise StageBindingError(
            "STAGE_BINDING_PLAN_MISSING: {} not found".format(path)
        )
    # read_run_plan raises LedgerError on corrupt plans; StageBindingError is
    # the more precise type for this guard -> re-wrap (still fail-closed).
    try:
        plan = read_run_plan(path)
    except PlanNotFrozenError:
        raise
    except Exception as exc:  # LedgerError hierarchy + OSError
        raise StageBindingError(
            "STAGE_BINDING_PLAN_UNUSABLE: {}: {}".format(path, exc)
        ) from exc

    lines: Dict[str, Dict[str, Any]] = plan["by_run_id"]
    targets: List[str]
    if run_id is None:
        targets = sorted(lines)
        if not targets:
            raise StageBindingError(
                "STAGE_BINDING_PLAN_EMPTY: {} has no planned runs".format(path)
            )
    else:
        if run_id not in lines:
            raise StageBindingError(
                "STAGE_BINDING_RUN_NOT_PLANNED: {} is not in {}".format(run_id, path)
            )
        targets = [run_id]

    problems: List[str] = []
    for rid in targets:
        line = lines[rid]
        for field in SHA_FIELDS:
            status = classify_sha(line.get(field))
            if status != "real":
                problems.append(
                    "run {} {}={!r} is {} (need a real 64-hex digest)".format(
                        rid, field, line.get(field), status
                    )
                )
    if problems:
        shown = "; ".join(problems[:5])
        more = "" if len(problems) <= 5 else " (+{} more)".format(len(problems) - 5)
        raise StageBindingError(
            "STAGE_{}_NOT_BOUND: {} plan line(s) in {} still carry non-real "
            "identity shas -- rebuild the plan with all five shas frozen "
            "before launching: {}{}".format(stage.upper(), len(problems), path, shown, more)
        )
    return {
        "stage": stage,
        "plan": str(path),
        "checked_runs": len(targets),
        "sha_fields": list(SHA_FIELDS),
        "bound": True,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stage_bind",
        description="Supervisor stage-g0 binding check: all five identity shas "
        "must be real 64-hex digests before any launch",
    )
    parser.add_argument(
        "--plan",
        default=None,
        help="run_plan.jsonl (default: <root>/" + DEFAULT_PLAN_SUBPATH + ")",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[3]),
        help="campaign root used to resolve the default --plan path",
    )
    parser.add_argument(
        "--stage", choices=LAUNCH_STAGES, default="g0", help="binding stage (default: g0)"
    )
    parser.add_argument(
        "--run-id", default=None, help="check only this planned run (default: all)"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    plan = Path(args.plan) if args.plan else Path(args.root) / DEFAULT_PLAN_SUBPATH
    try:
        summary = assert_stage_binding(plan, stage=args.stage, run_id=args.run_id)
    except StageBindingError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
