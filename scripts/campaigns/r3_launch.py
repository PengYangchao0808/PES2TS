"""R3 supervised-launch entrypoint (WP-L3): campaign root + plan pin + stage-g0.

Thin composition of the WP-L3 hardening pieces into the single command the
E3 lane will use to launch each R3 run:

1. ``assert_campaign_root`` -- output root defaults to
   ``outputs/pesn/30h_campaign/**``; sealed prefixes refuse fail-closed;
2. bound ``run_plan.jsonl`` is pin-verified (sha256 + campaign_mode vs the
   E0-frozen pin) on every append (``RunLedger`` WP-L3 wiring);
3. optional ``--stage g0`` requires all five identity shas of the plan line
   to be real 64-hex digests before the launch is attempted
   (``supervisor.stage_bind``);
4. the launch itself goes through ``supervisor.monitor.attempt_launch`` --
   a ``PlanNotFrozenError`` is reported as ``blocked_with_reason`` (exit 3),
   never a crash.

Usage::

    python scripts/campaigns/r3_launch.py --record run_record.json --plan run_plan.jsonl
    python scripts/campaigns/r3_launch.py --record r.json --plan p.jsonl --stage g0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.ledger.run_ledger import (  # noqa: E402
    CAMPAIGN_PIN,
    LedgerError,
    RunLedger,
)
from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CAMPAIGN_ROOT,
    CampaignRootError,
    assert_campaign_root,
)
from scripts.agent_ops.supervisor.monitor import attempt_launch  # noqa: E402
from scripts.agent_ops.supervisor.stage_bind import (  # noqa: E402
    DEFAULT_PLAN_SUBPATH,
    DEFAULT_TASK_STATE_PATH,
)

DEFAULT_LEDGER = CAMPAIGN_ROOT + "/run_records.jsonl"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r3_launch",
        description="Supervised R3 launch: campaign-root assert + plan pin + "
        "optional stage-g0 five-sha binding; PlanNotFrozen -> blocked report",
    )
    parser.add_argument("--ledger", default=DEFAULT_LEDGER, help="run_records.jsonl")
    parser.add_argument(
        "--plan", default=None, help="bound run_plan.jsonl (default: campaign plan)"
    )
    parser.add_argument(
        "--task-state", default=DEFAULT_TASK_STATE_PATH, help="task_state.json (pin source)"
    )
    parser.add_argument("--unseal", default=None, help="confirm BATCH unseal token path")
    parser.add_argument(
        "--metric-unseal", default=None, help="B folds3/4 metric-reveal token path"
    )
    parser.add_argument("--record", help="path to a run_record JSON file")
    parser.add_argument("--record-json", help="inline run_record JSON")
    parser.add_argument(
        "--stage",
        choices=("g0", "launch"),
        default=None,
        help="require the plan line's five identity shas to be real 64-hex "
        "before launching",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.record) == bool(args.record_json):
        sys.stderr.write("error: pass exactly one of --record/--record-json\n")
        return 2
    # 1. campaign-root assertion FIRST (never open outputs under sealed dirs)
    try:
        ledger_root = assert_campaign_root(str(Path(args.ledger).parent))
    except CampaignRootError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    if args.plan is not None:
        try:
            assert_campaign_root(str(Path(args.plan).parent))
        except CampaignRootError as exc:
            sys.stderr.write("error: plan outside campaign: {}\n".format(exc))
            return 1
    plan = args.plan or str(PROJECT_ROOT / DEFAULT_PLAN_SUBPATH)
    ledger = RunLedger(
        args.ledger,
        plan_path=plan,
        plan_pin=CAMPAIGN_PIN,
        unseal_path=args.unseal,
        metric_unseal_path=args.metric_unseal,
    )
    try:
        raw = (
            Path(args.record).read_text(encoding="utf-8")
            if args.record
            else args.record_json
        )
        record = json.loads(raw)
        report = attempt_launch(
            ledger, record, stage=args.stage, task_state=args.task_state
        )
    except (LedgerError, OSError) as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    except json.JSONDecodeError as exc:
        sys.stderr.write("error: record is not valid JSON: {}\n".format(exc))
        return 2
    report["ledger_root"] = ledger_root
    print(json.dumps(report, sort_keys=True))
    return 0 if report["state"] in ("running", "done") else 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
