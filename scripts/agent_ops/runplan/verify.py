"""Verify ``run_plan.jsonl``: exact 66+33 budget, unique run_ids, reuse links.

Fail-closed verifier for the E0 planning artifact (30h plan §4.2 + §5.5,
PD-3 conditional).  Every check must pass or the process exits 1:

1. header present (exactly once, first line) with the exact ``pd3_note``;
2. every run line has all required fields, ``status="planned"``, the exact
   campaign id, and valid 64-hex identity shas;
3. exact counts: R3=66, V5.2=33, total=99 (no other phases);
4. exact per-substage counts (12/3/3/12/12/12/6/6 + 9/9/9/6);
5. ``run_id`` unique **and** equal to the recomputed
   ``sha256(campaign|phase|arm|fold|seed|variant|shas)[:16]``;
6. no cross-arm duplicate ``(campaign_id, phase, arm, fold, seed)``;
7. legal seeds (17/29/43) and folds (int 0..4);
8. every ``reuses[]`` target resolves to at least one planned line -- either
   an exact ``run_id`` present in the plan or a selector
   (``substage:<name>`` / ``phase:<name>``) matching ≥1 planned run.

Success prints one verification line, e.g.::

    run_plan VERIFY PASS: R3=66 V5.2=33 total=99 unique_run_ids=99
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.agent_ops.ledger.run_ledger import HASH_RE, compute_run_id
from scripts.agent_ops.runplan.build_r3_runplan import (
    CAMPAIGN_ID,
    DEFAULT_OUT_SUBPATH,
    PHASE_COUNTS,
    PD3_NOTE,
    SEEDS,
    SUBSTAGE_COUNTS,
    SHA_FIELDS,
)

__all__ = ["main", "verify_plan"]

REQUIRED_RUN_KEYS = (
    "run_id",
    "campaign_id",
    "phase",
    "substage",
    "arm",
    "fold",
    "seed",
    "variant",
    "reuses",
    "status",
    "decision_note",
)
LEGAL_FOLD_MAX = 4


def _resolve_reuses(
    target: Any, run_ids: set, selector_index: Dict[Tuple[str, str], int]
) -> bool:
    if not isinstance(target, str) or not target:
        return False
    if target.startswith("substage:"):
        return selector_index.get(("substage", target[len("substage:"):]), 0) > 0
    if target.startswith("phase:"):
        return selector_index.get(("phase", target[len("phase:"):]), 0) > 0
    return target in run_ids  # concrete run_id must exist in this plan


def verify_plan(plan_path: Any) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Run every check; returns ``(ok, problems, summary)`` (never raises)."""
    problems: List[str] = []
    path = Path(plan_path)
    summary: Dict[str, Any] = {"plan": str(path)}

    if not path.is_file():
        return False, ["PLAN_MISSING: {} not found".format(path)], summary
    try:
        raw_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError as exc:
        return False, ["PLAN_UNREADABLE: {}: {}".format(path, exc)], summary
    if not raw_lines:
        return False, ["PLAN_CORRUPT: {} is empty".format(path)], summary

    parsed: List[Dict[str, Any]] = []
    for lineno, line in enumerate(raw_lines, start=1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append("PLAN_CORRUPT: line {}: {}".format(lineno, exc))
            continue
        if not isinstance(rec, dict):
            problems.append("PLAN_CORRUPT: line {} is not a JSON object".format(lineno))
            continue
        parsed.append(rec)

    # 1. header (exactly one, first line, exact pd3_note)
    headers = [
        (i, rec) for i, rec in enumerate(parsed) if rec.get("record_type") == "plan_header"
    ]
    if len(headers) != 1:
        problems.append(
            "HEADER: expected exactly 1 plan_header line, found {}".format(len(headers))
        )
    elif headers[0][0] != 0:
        problems.append("HEADER: plan_header must be the first line")
    else:
        if headers[0][1].get("pd3_note") != PD3_NOTE:
            problems.append(
                "PD3_NOTE: header pd3_note missing/changed (PD-3 conditional text required)"
            )
        if headers[0][1].get("campaign_id") != CAMPAIGN_ID:
            problems.append("PD3_NOTE: header campaign_id != {}".format(CAMPAIGN_ID))

    run_lines = [rec for rec in parsed if rec.get("record_type") != "plan_header"]

    # 2. required fields / status / campaign / shas
    for i, rec in enumerate(run_lines, start=1):
        missing = [k for k in REQUIRED_RUN_KEYS if k not in rec]
        if missing:
            problems.append("RUN_LINE {}: missing {}".format(i, ", ".join(missing)))
            continue
        if rec.get("campaign_id") != CAMPAIGN_ID:
            problems.append(
                "RUN_LINE {}: campaign_id {!r} != {}".format(
                    i, rec.get("campaign_id"), CAMPAIGN_ID
                )
            )
        if rec.get("status") != "planned":
            problems.append(
                "RUN_LINE {}: status {!r} != planned".format(i, rec.get("status"))
            )
        if not isinstance(rec.get("reuses"), list):
            problems.append("RUN_LINE {}: reuses must be a list".format(i))
        for field in SHA_FIELDS:
            value = rec.get(field)
            if not (isinstance(value, str) and HASH_RE.match(value)):
                problems.append(
                    "RUN_LINE {}: invalid {}".format(i, field)
                )

    # 3. exact phase counts
    phase_counts: Dict[str, int] = {}
    for rec in run_lines:
        phase_counts[rec.get("phase")] = phase_counts.get(rec.get("phase"), 0) + 1
    for phase, expected in PHASE_COUNTS.items():
        actual = phase_counts.get(phase, 0)
        if actual != expected:
            problems.append(
                "COUNT: phase {} has {} runs, expected {}".format(phase, actual, expected)
            )
    for phase in phase_counts:
        if phase not in PHASE_COUNTS:
            problems.append("COUNT: unexpected phase {!r} in plan".format(phase))
    total = len(run_lines)
    if total != sum(PHASE_COUNTS.values()):
        problems.append(
            "COUNT: total {} runs, expected {}".format(total, sum(PHASE_COUNTS.values()))
        )

    # 4. per-substage counts (exact set + exact counts)
    substage_counts: Dict[str, int] = {}
    for rec in run_lines:
        substage_counts[rec.get("substage")] = substage_counts.get(rec.get("substage"), 0) + 1
    for substage, expected in SUBSTAGE_COUNTS.items():
        actual = substage_counts.get(substage, 0)
        if actual != expected:
            problems.append(
                "SUBSTAGE: {} has {} runs, expected {}".format(substage, actual, expected)
            )
    for substage in substage_counts:
        if substage not in SUBSTAGE_COUNTS:
            problems.append("SUBSTAGE: unexpected substage {!r}".format(substage))

    # 5. run_id uniqueness + recomputability
    seen_ids: Dict[str, int] = {}
    for i, rec in enumerate(run_lines, start=1):
        run_id = rec.get("run_id")
        if not run_id:
            problems.append("RUN_ID: line {} has no run_id".format(i))
            continue
        if run_id in seen_ids:
            problems.append(
                "RUN_ID: duplicate {} (lines {} and {})".format(
                    run_id, seen_ids[run_id], i
                )
            )
        else:
            seen_ids[run_id] = i
        try:
            expected_id = compute_run_id(rec, scheme="hashed")
        except KeyError as exc:
            problems.append("RUN_ID: line {} cannot recompute run_id: {}".format(i, exc))
            continue
        if run_id != expected_id:
            problems.append(
                "RUN_ID: line {} has {} but recomputes to {} (identity drift)".format(
                    i, run_id, expected_id
                )
            )

    # 6. cross-arm duplicate key
    seen_keys: Dict[tuple, int] = {}
    for i, rec in enumerate(run_lines, start=1):
        key = (
            rec.get("campaign_id"),
            rec.get("phase"),
            rec.get("arm"),
            rec.get("fold"),
            rec.get("seed"),
        )
        if key in seen_keys:
            problems.append(
                "CROSS_ARM_DUP: {} first at line {}, repeated at line {}".format(
                    key, seen_keys[key], i
                )
            )
        else:
            seen_keys[key] = i

    # 7. legal seeds / folds
    for i, rec in enumerate(run_lines, start=1):
        seed = rec.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
            problems.append("SEED: line {} has illegal seed {!r}".format(i, seed))
        fold = rec.get("fold")
        if (
            isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold < 0
            or fold > LEGAL_FOLD_MAX
        ):
            problems.append("FOLD: line {} has illegal fold {!r}".format(i, fold))

    # 8. reuse targets must exist and be planned
    run_ids = {rec.get("run_id") for rec in run_lines}
    selector_index: Dict[Tuple[str, str], int] = {}
    for rec in run_lines:
        if rec.get("status") != "planned":
            continue  # already reported above; reuse requires planned targets
        for kind in ("substage", "phase"):
            selector_index[(kind, rec.get(kind))] = selector_index.get(
                (kind, rec.get(kind)), 0
            ) + 1
    for i, rec in enumerate(run_lines, start=1):
        reuses = rec.get("reuses")
        if not isinstance(reuses, list):
            continue  # reported above
        for target in reuses:
            if not _resolve_reuses(target, run_ids, selector_index):
                problems.append(
                    "REUSE: line {} target {!r} does not resolve to a planned run".format(
                        i, target
                    )
                )

    summary.update(
        {
            "R3": phase_counts.get("R3", 0),
            "V5.2": phase_counts.get("V5.2", 0),
            "total": total,
            "unique_run_ids": len(seen_ids),
            "substages": substage_counts,
            "problems": len(problems),
        }
    )
    return (not problems), problems, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runplan_verify",
        description="Verify run_plan.jsonl: 66 R3 + 33 V5.2, unique run_ids, reuse targets",
    )
    parser.add_argument(
        "--plan",
        default=None,
        help="path to run_plan.jsonl (default: <root>/outputs/pesn/30h_campaign/run_plan.jsonl)",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[3]),
        help="campaign root used to resolve the default --plan path",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    plan = Path(args.plan) if args.plan else Path(args.root) / DEFAULT_OUT_SUBPATH
    ok, problems, summary = verify_plan(plan)
    if ok:
        print(
            "run_plan VERIFY PASS: R3={} V5.2={} total={} unique_run_ids={}".format(
                summary["R3"], summary["V5.2"], summary["total"], summary["unique_run_ids"]
            )
        )
        return 0
    for problem in problems:
        sys.stderr.write("{}\n".format(problem))
    sys.stderr.write(
        "run_plan VERIFY FAIL: {} problem(s) in {}".format(len(problems), plan)
    )
    sys.stderr.write("\n")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
