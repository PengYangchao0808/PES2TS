"""Verify ``run_plan.jsonl``: exact 66+33 budget, unique run_ids, reuse links.

Fail-closed verifier for the E0 planning artifact (30h plan §4.2 + §5.5,
PD-3 conditional; PD-4 partial set).  ``--stage`` selects the identity-sha
binding rules (ORA-2 blocker 1):

* ``e0`` (default): ``config_sha256`` must be REAL 64-hex (campaign yaml);
  ``data/teacher/features/split`` must be the literal ``PENDING_E1`` /
  ``PENDING_E2`` tokens; the header must carry the ``two_stage_binding`` note;
* ``g0`` / ``launch``: ALL five shas must be REAL 64-hex digests;
* **any stage**: an all-zeros sha, an unknown token or a malformed digest is
  ``VERIFY FAIL``.

Every check must pass or the process exits 1:

1. header exactly once (first line); full mode carries the exact ``pd3_note``
   and partial mode carries ``pd4_note`` + ``campaign_mode: "partial"``;
2. every run line has all required fields, ``status="planned"``, the exact
   campaign id, explicit ``confirm``/``exploratory_only`` flags;
3. exact counts: full = R3 66 + V5.2 33 = 99; partial = R3 21 (B3/F-bidir3/
   F-ablate3/L12, exploratory_only true everywhere, no V5.2);
4. ``run_id`` unique **and** equal to the recomputed
   ``sha256(campaign|phase|arm|fold|seed|variant|shas)[:16]``;
5. no cross-arm duplicate ``(campaign_id, phase, arm, fold, seed, variant)``;
6. legal seeds (17/29/43) and folds (int 0..4);
7. every ``reuses[]`` target resolves to at least one planned line -- either
   an exact ``run_id`` present in the plan or a selector
   (``substage:<name>`` / ``phase:<name>``) matching ≥1 planned run;
8. ``confirm: true`` only on fold3/4 C / QT-confirm lines, and those lines
   must always carry it (explicit plan config, no heuristics).

Success prints one verification line, e.g.::

    run_plan VERIFY PASS: R3=66 V5.2=33 total=99 unique_run_ids=99 stage=e0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.agent_ops.ledger.run_ledger import compute_run_id
from scripts.agent_ops.runplan.build_r3_runplan import (
    CAMPAIGN_ID,
    DEFAULT_OUT_SUBPATH,
    PARTIAL_PHASE_COUNTS,
    PARTIAL_SUBSTAGE_COUNTS,
    PD3_NOTE,
    PD4_NOTE,
    PHASE_COUNTS,
    PENDING_TOKENS,
    SEEDS,
    SHA_FIELDS,
    SUBSTAGE_COUNTS,
    TWO_STAGE_BINDING_NOTE,
    classify_sha,
)

__all__ = ["STAGES", "main", "verify_plan"]

STAGES = ("e0", "g0", "launch")

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
    "confirm",
    "exploratory_only",
    "decision_note",
)
LEGAL_FOLD_MAX = 4
CONFIRM_SUBSTAGES = ("C", "QT-confirm")


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


def _check_stage_shas(
    rec: Dict[str, Any], lineno: int, stage: str, problems: List[str]
) -> None:
    statuses = {f: classify_sha(rec.get(f)) for f in SHA_FIELDS}
    for field, status in statuses.items():
        if status == "invalid":
            problems.append(
                "SHA: line {} has invalid {}={!r} (need real 64-hex or "
                "{})".format(lineno, field, rec.get(field), "/".join(PENDING_TOKENS))
            )
        elif status == "zero":
            problems.append(
                "SHA_ZERO: line {} has all-zeros {} (zero shas are never valid "
                "at any stage)".format(lineno, field)
            )
    if stage == "e0":
        if statuses["config_sha256"] != "real":
            problems.append(
                "STAGE_E0: line {} config_sha256 must be a real 64-hex digest "
                "(got {!r})".format(lineno, rec.get("config_sha256"))
            )
        for field in ("data_sha256", "teacher_sha256", "features_sha256", "split_sha256"):
            if statuses[field] not in PENDING_TOKENS:
                problems.append(
                    "STAGE_E0: line {} {} must be {} at stage e0 (got {!r})".format(
                        lineno, field, "/".join(PENDING_TOKENS), rec.get(field)
                    )
                )
    else:  # g0 / launch: everything frozen
        for field, status in statuses.items():
            if status != "real":
                problems.append(
                    "STAGE_{}: line {} {} must be a real frozen 64-hex digest "
                    "(got {!r})".format(stage.upper(), lineno, field, rec.get(field))
                )


def verify_plan(plan_path: Any, stage: str = "e0") -> Tuple[bool, List[str], Dict[str, Any]]:
    """Run every check; returns ``(ok, problems, summary)`` (never raises)."""
    problems: List[str] = []
    path = Path(plan_path)
    summary: Dict[str, Any] = {"plan": str(path), "stage": stage}
    if stage not in STAGES:
        return False, [
            "INVALID_STAGE: {!r} not in {}".format(stage, ", ".join(STAGES))
        ], summary

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

    run_lines = [rec for rec in parsed if rec.get("record_type") != "plan_header"]

    # 1. header (exactly one, first line) + mode + notes + two-stage binding
    headers = [
        (i, rec) for i, rec in enumerate(parsed) if rec.get("record_type") == "plan_header"
    ]
    mode = "full"
    if len(headers) != 1:
        problems.append(
            "HEADER: expected exactly 1 plan_header line, found {}".format(len(headers))
        )
    elif headers[0][0] != 0:
        problems.append("HEADER: plan_header must be the first line")
    else:
        header = headers[0][1]
        mode = header.get("campaign_mode")
        if mode not in ("full", "partial"):
            problems.append(
                "HEADER: campaign_mode {!r} not in full/partial".format(mode)
            )
            mode = "full"
        if not (
            isinstance(header.get("two_stage_binding"), str)
            and header.get("two_stage_binding")
        ):
            problems.append(
                "TWO_STAGE_BINDING: header is missing the two_stage_binding note"
            )
        elif TWO_STAGE_BINDING_NOTE not in header.get("two_stage_binding"):
            problems.append(
                "TWO_STAGE_BINDING: header note does not match the canonical text"
            )
        if header.get("campaign_id") != CAMPAIGN_ID:
            problems.append("HEADER: campaign_id != {}".format(CAMPAIGN_ID))
        if mode == "full":
            if header.get("pd3_note") != PD3_NOTE:
                problems.append(
                    "PD3_NOTE: header pd3_note missing/changed (PD-3 conditional "
                    "text required)"
                )
            if header.get("budget") != PHASE_COUNTS:
                problems.append("HEADER: budget must be {}".format(PHASE_COUNTS))
            if header.get("exploratory_only", False) is True:
                problems.append("HEADER: full plan must not be exploratory")
        else:
            note = header.get("pd4_note")
            if not (isinstance(note, str) and "21" in note and "PD-4" in note):
                problems.append(
                    "PD4_NOTE: partial header must carry the 21-run PD-4 note"
                )
            if header.get("budget") != PARTIAL_PHASE_COUNTS:
                problems.append(
                    "HEADER: partial budget must be {}".format(PARTIAL_PHASE_COUNTS)
                )
        if "sha_status" in header and run_lines:
            for field in SHA_FIELDS:
                reported = header["sha_status"].get(field)
                actual = classify_sha(run_lines[0].get(field))
                if reported != actual:
                    problems.append(
                        "HEADER: sha_status[{!r}]={} but plan lines are {}".format(
                            field, reported, actual
                        )
                    )
        elif "sha_status" not in header:
            problems.append("HEADER: missing per-sha sha_status map")

    # 2. required fields / status / campaign / flags / stage shas
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
        if not isinstance(rec.get("confirm"), bool):
            problems.append(
                "RUN_LINE {}: confirm must be an explicit bool".format(i)
            )
        if not isinstance(rec.get("exploratory_only"), bool):
            problems.append(
                "RUN_LINE {}: exploratory_only must be an explicit bool".format(i)
            )
        elif mode == "partial" and rec.get("exploratory_only") is not True:
            problems.append(
                "RUN_LINE {}: partial-plan line must be exploratory_only=true".format(i)
            )
        elif mode == "full" and rec.get("exploratory_only") is True:
            problems.append(
                "RUN_LINE {}: full-plan line must not be exploratory_only".format(i)
            )
        _check_stage_shas(rec, i, stage, problems)

        # explicit confirm flag consistency (blocker 2: no heuristics)
        substage = rec.get("substage")
        fold = rec.get("fold")
        if rec.get("confirm") is True:
            if substage not in CONFIRM_SUBSTAGES or fold not in (3, 4):
                problems.append(
                    "CONFIRM: line {} sets confirm=true outside the fold3/4 "
                    "{}/{} lines".format(i, "/".join(CONFIRM_SUBSTAGES), "3-4")
                )
        elif substage in CONFIRM_SUBSTAGES and fold in (3, 4):
            problems.append(
                "CONFIRM: line {} ({}, fold {}) must carry explicit "
                "confirm=true".format(i, substage, fold)
            )

    # 3. exact counts (per mode)
    phase_counts: Dict[str, int] = {}
    for rec in run_lines:
        phase_counts[rec.get("phase")] = phase_counts.get(rec.get("phase"), 0) + 1
    expected_phases = PARTIAL_PHASE_COUNTS if mode == "partial" else PHASE_COUNTS
    for phase, expected in expected_phases.items():
        actual = phase_counts.get(phase, 0)
        if actual != expected:
            problems.append(
                "COUNT: phase {} has {} runs, expected {} (mode={})".format(
                    phase, actual, expected, mode
                )
            )
    for phase in phase_counts:
        if phase not in expected_phases:
            problems.append(
                "COUNT: unexpected phase {!r} in {} plan".format(phase, mode)
            )
    total = len(run_lines)
    if total != sum(expected_phases.values()):
        problems.append(
            "COUNT: total {} runs, expected {} (mode={})".format(
                total, sum(expected_phases.values()), mode
            )
        )

    # 4. per-substage counts (exact set + exact counts, per mode)
    substage_counts: Dict[str, int] = {}
    for rec in run_lines:
        substage_counts[rec.get("substage")] = substage_counts.get(rec.get("substage"), 0) + 1
    expected_substages = (
        PARTIAL_SUBSTAGE_COUNTS if mode == "partial" else SUBSTAGE_COUNTS
    )
    for substage, expected in expected_substages.items():
        actual = substage_counts.get(substage, 0)
        if actual != expected:
            problems.append(
                "SUBSTAGE: {} has {} runs, expected {}".format(substage, actual, expected)
            )
    for substage in substage_counts:
        if substage not in expected_substages:
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

    # 6. cross-arm duplicate key (incl. variant)
    seen_keys: Dict[tuple, int] = {}
    for i, rec in enumerate(run_lines, start=1):
        key = (
            rec.get("campaign_id"),
            rec.get("phase"),
            rec.get("arm"),
            rec.get("fold"),
            rec.get("seed"),
            rec.get("variant"),
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
            "mode": mode,
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
        description="Verify run_plan.jsonl: 66 R3 + 33 V5.2 (or PD-4 partial 21), "
        "unique run_ids, reuse targets, two-stage sha binding",
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
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="e0",
        help="identity-sha binding stage: e0 = real config + PENDING_E1/E2 "
        "tokens; g0/launch = all five shas frozen real (default: e0)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    plan = Path(args.plan) if args.plan else Path(args.root) / DEFAULT_OUT_SUBPATH
    ok, problems, summary = verify_plan(plan, stage=args.stage)
    if ok:
        if summary.get("mode") == "partial":
            print(
                "run_plan VERIFY PASS: mode=partial R3={} total={} "
                "unique_run_ids={} stage={}".format(
                    summary["R3"], summary["total"], summary["unique_run_ids"], args.stage
                )
            )
        else:
            print(
                "run_plan VERIFY PASS: R3={} V5.2={} total={} unique_run_ids={} "
                "stage={}".format(
                    summary["R3"],
                    summary["V5.2"],
                    summary["total"],
                    summary["unique_run_ids"],
                    args.stage,
                )
            )
        return 0
    for problem in problems:
        sys.stderr.write("{}\n".format(problem))
    sys.stderr.write(
        "run_plan VERIFY FAIL: {} problem(s) in {} (stage={})\n".format(
            len(problems), plan, args.stage
        )
    )
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
