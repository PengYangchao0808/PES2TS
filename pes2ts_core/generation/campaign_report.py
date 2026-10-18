"""First-feedback report builder (G2-AB2 WP-6, ``pes2ts_g2t_feedback_report_v1``).

Reads one campaign root (terminal states, cost bundles, continuation
results, candidate sets, origin evidences) and emits the nine frozen content
items of the batch plan §4 WP-6.  Anti-overclaim rules are structural
(§11.4): the four-class denominators are computed from the terminal states
and can never fold ``rejected``/``censored``/``blocked`` into a success
fraction; no M0–M5 claim is emitted; the minimal ``J_G`` block is an
explicit no-data statement unless validated results exist; every failure
row carries its evidence reference (R5: no ledger, no report row).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pes2ts_core.generation.campaign import (
    AUDIT_TERMINAL_CLASSES,
    GENERATION_TERMINAL_CLASSES,
    TERMINAL_STATE_SCHEMA,
    VALIDATED_TERMINAL_CLASSES,
    load_campaign_manifest,
)
from pes2ts_core.generation.planning.replay_report import (
    _origin_failure_detail_complete,
)

FEEDBACK_REPORT_SCHEMA = "pes2ts_g2t_feedback_report_v1"

#: Capability status vocabulary (constitution §11.5) — honest layer states.
#: Frozen once against the real-campaign evidence (T20-01, 2026-10-05): the
#: T21 final report embeds exactly these values.  ``validation_chain`` stays
#: ``implemented`` because S4 gated on human review, and ``feedback_reporter``
#: stays ``fixture_passed`` until T21 replays it over the real root.
CAPABILITY_STATUS = {
    "campaign_orchestrator": {"status": "smoke_passed",
        "evidence": "S3 outputs/g2t_campaign_s3_20261004 24/24 typed terminals; "
                    ".omo/evidence/T18b-05-collector.log rc=0 + T18b-06 assertions; "
                    "idempotent resume T17b/T18b-04; double-run cache T15c-03"},
    "cost_capture": {"status": "smoke_passed",
        "evidence": "S3 24 costs.json / 443 ledgers, N_gradient_total=26676; "
                    "T18b-06 assertion [c] + RXN_0000155302 zero-attempt ledger; "
                    "T15c-08 cost completeness gate"},
    "origin_preparation_acp_adapter": {"status": "smoke_passed",
        "evidence": ".omo/evidence/task-11-g2t-real-acp/T11-acceptance.json "
                    "(real opt_only + energy trajectory, fixA/fixB PASS); "
                    "S3 23 origin runs (22 prepared, 1 typed ORIGIN_BUDGET_EXHAUSTED)"},
    "branch_policy_calibration": {"status": "smoke_passed",
        "evidence": "three frozen rounds T15/T15b/T15c + T15c-08 GO re-verification; "
                    "S3 real branch_evidence 319/320 biased_only frames, drift max "
                    "0.998 < limit 1.0, curvature-probe budget respected (T17b)"},
    "pilot_replay_judge": {"status": "smoke_passed",
        "evidence": "judge v2 real pilot rounds T15->T15b->T15c = 0/4->3/4->4/4 GO "
                    "(outputs/g2t_campaign_logs/T15c-08-judge.log, "
                    "T15c-09-summary.log; T14b-05 first real pilot)"},
    "continuation_real_engine": {"status": "smoke_passed",
        "evidence": "S3 real 24-case continuation N_gradient_total=26676 "
                    "(T18b-06/T17b-05); pilot GO 4/4 (T15c-09); v3cal 7762 rerun "
                    "24/24 cache_hit (T15c-03)"},
    "validation_chain": {"status": "implemented",
        "evidence": "S4 gated: 24/24 validation.status=blocked_on_human_review "
                    "(T18b-06 assertion [f]); needs an accepted ReviewRecord"},
    "feedback_reporter": {"status": "fixture_passed",
        "evidence": "tests/test_campaign_report.py (synthetic campaign, offline); "
                    "T21 real-root replay pending"},
}

_NO_DATA_J_G = {
    "available": False,
    "note": "no validated results in this campaign; the full J_G/R_oracle "
            "evaluation protocol belongs to the G3.0 precondition batch",
}

_FAILURE_FIX_DIRECTIONS = {
    "IDENTITY_CONFLICT": "audit receipt reuse vs recompute boundaries; keep "
                         "typed conflict, never recompute in place (B1-1 family)",
    "ORIGIN_CONNECTION_LOST": "constrain the free optimization (bond guards) or "
                              "switch method; the typed atom pairs name the bond",
    "ORIGIN_BACKEND_UNAVAILABLE": "provide ACP checkout/interpreter with the "
                                  "BatchOptimize non-TS capability (ACP-G2-04)",
    "BUDGET_EXHAUSTED": "raise per-path budget in a NEW campaign manifest or "
                        "improve step policy; never silently resume past budget",
    "STEP_LIMIT": "step floor reached — branch policy/steps calibration or "
                  "coordinate set review (segment diagnostics)",
    "LOCALITY_CUMULATIVE": "cross-frame drift escaped the branch anchor — "
                           "strengthen kappa/radii in a NEW calibration",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered)//2
    if len(ordered) % 2:
        return ordered[middle]
    return .5*(ordered[middle-1]+ordered[middle])


def _four_class_denominators(terminals: list[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in GENERATION_TERMINAL_CLASSES}
    counts["validated"] = 0
    counts["censored_budget"] = 0
    counts["blocked"] = 0
    for doc in terminals:
        name = doc.get("terminal_class")
        if name in GENERATION_TERMINAL_CLASSES:
            counts[name] += 1
        elif name in VALIDATED_TERMINAL_CLASSES:
            counts["validated"] += 1
        elif name in AUDIT_TERMINAL_CLASSES:
            counts[name] += 1
    return counts


def _funnel(terminals: list[dict[str, Any]]) -> dict[str, Any]:
    stages = ["input", "origin_preparation", "plan", "continuation",
              "candidate_extraction"]
    reached = {stage: 0 for stage in stages}
    for doc in terminals:
        stage = doc.get("stage_reached")
        if stage in reached:
            reached[stage] += 1
    total = len(terminals)
    return {"n_cases": total, "stage_reached_counts": reached,
            "four_class_denominators": _four_class_denominators(terminals),
            "note": "rejected_typed includes censored_budget; blocked is "
                    "reported separately and never enters a success fraction"}


def _failure_histogram(terminals: list[dict[str, Any]],
                       case_costs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for doc in terminals:
        code = doc.get("failure_code")
        if not code:
            continue
        key = (code, doc.get("stage_reached") or "unknown")
        row = buckets.setdefault(key, {"failure_code": code,
                                       "stage": key[1], "cases": [],
                                       "n_gradient": [], "cpu_seconds": []})
        row["cases"].append(doc.get("reaction_id"))
        aggregate = (case_costs.get(doc.get("reaction_id")) or {}).get("aggregate") or {}
        row["n_gradient"].append(int(aggregate.get("N_gradient") or 0))
        cpu = aggregate.get("cpu_seconds")
        if isinstance(cpu, (int, float)):
            row["cpu_seconds"].append(float(cpu))
    histogram = []
    for row in sorted(buckets.values(),
                      key=lambda r: (-len(r["cases"]), r["failure_code"], r["stage"])):
        histogram.append({"failure_code": row["failure_code"], "stage": row["stage"],
                          "n_cases": len(row["cases"]), "cases": sorted(row["cases"]),
                          "median_n_gradient": _median(row["n_gradient"]),
                          "median_cpu_seconds": _median(row["cpu_seconds"])})
    return histogram


def _b1_checks(terminals: list[dict[str, Any]], case_root: dict[str, Path],
               references: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for doc in terminals:
        rid = doc.get("reaction_id")
        origin_path = case_root.get(rid)
        origin_evidence = _read(origin_path / "origin_evidence.json") \
            if origin_path and (origin_path / "origin_evidence.json").is_file() else {}
        reference = references.get(rid) or {}
        reference_lambda = reference.get("last_accepted_lambda")
        last_lambda = doc.get("last_accepted_lambda")
        # G2-AB3 WP-0 audit fix P2-3: a missing reference (or a missing
        # current lambda) is not_comparable — never "progressed beyond".
        progressed: bool | None = None
        progressed_reason: str | None = None
        if reference_lambda is None:
            progressed_reason = "reference_last_accepted_lambda_missing"
        elif not isinstance(last_lambda, (int, float)):
            progressed_reason = "current_last_accepted_lambda_missing"
        else:
            progressed = float(last_lambda) > float(reference_lambda)
        origin_failed = doc.get("origin_status") == "failed"
        rows.append({
            "reaction_id": rid,
            "b1_1_identity_clear": int(doc.get("identity_conflict_count") or 0) == 0,
            "b1_2_progressed_beyond_reference": progressed,
            "b1_2_not_comparable_reason": progressed_reason,
            "reference_last_accepted_lambda": reference_lambda,
            "current_last_accepted_lambda": last_lambda,
            "b1_3_origin_failure_typed": (bool(_origin_failure_detail_complete(origin_evidence))
                                          if origin_failed else None),
            "b1_4_candidates_extracted_not_dropped": bool(
                doc.get("candidates") is not None
                and isinstance(doc.get("validation"), dict)),
        })
    return {
        "b1_1_identity_conflicts_total": sum(
            int(doc.get("identity_conflict_count") or 0) for doc in terminals),
        "b1_2_cases_with_reference": sum(1 for row in rows
                                         if row["reference_last_accepted_lambda"] is not None),
        "b1_2_progressed_count": sum(1 for row in rows
                                     if row["b1_2_progressed_beyond_reference"] is True),
        "b1_2_not_comparable_count": sum(1 for row in rows
                                         if row["b1_2_progressed_beyond_reference"] is None),
        "b1_3_typed_origin_failures": sum(1 for row in rows
                                          if row["b1_3_origin_failure_typed"]),
        "per_case": rows,
        "note": "B1-2 is true only when both reference and current lambdas "
                "are valid numbers and current > reference; a missing "
                "reference/current is not_comparable (null + reason), never "
                "'progressed beyond'. B1-3 is None when the origin succeeded "
                "(nothing to clear); B1-4 is True when the candidate artifact "
                "exists and the validation status is explicit "
                "(blocked_on_human_review counts)",
    }


def _branch_evidence(case_root: dict[str, Path]) -> dict[str, Any]:
    """Branch statistics over the EXACT verified case set.

    G2-AB3 WP-0 audit fix P2-2: iterates only the case directories that
    carry a verified ``terminal_state.json`` (and, when a manifest was
    supplied, a matching ``campaign_sha256``) — historical crash directories
    like ``RXN_*.crashed_v1`` (continuation results without a current
    terminal) must never enter current statistics.
    """
    accepted_frames, biased_only, drifts, probes = [], [], [], 0
    modes: dict[str, int] = {}
    for rid in sorted(case_root):
        result_path = case_root[rid] / "continuation_result.json"
        if not result_path.is_file():
            continue
        result = _read(result_path)
        if result.get("reaction_id") not in (None, rid):
            raise ValueError(f"CONTINUATION_REACTION_ID_MISMATCH:{rid}")
        frames = result.get("frames", [])
        accepted_frames.append(max(0, len(frames)-1))
        for frame in frames[1:]:
            mode = frame.get("convergence_mode") or "unreported"
            modes[mode] = modes.get(mode, 0)+1
            if mode == "biased_only":
                biased_only.append(1)
            drift = (frame.get("quality") or {}).get("cumulative_drift_angstrom")
            if isinstance(drift, (int, float)) and math.isfinite(float(drift)):
                drifts.append(float(drift))
        probes += sum(1 for attempt in result.get("attempts", [])
                      if attempt.get("curvature_probe"))
    n_frames = sum(accepted_frames)
    return {
        "n_cases_with_continuation": len(accepted_frames),
        "accepted_frames_beyond_origin_per_case": accepted_frames,
        "convergence_mode_histogram": dict(sorted(modes.items())),
        "biased_only_rate": (sum(biased_only)/n_frames) if n_frames else None,
        "cumulative_drift_angstrom": {
            "n_frames": len(drifts),
            "median": _median(drifts),
            "max": max(drifts) if drifts else None},
        "curvature_probes_total": probes,
        "first_real_statistics": True,
    }


def _candidates_and_validation(terminals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for doc in terminals:
        candidates = doc.get("candidates") or {}
        validation = doc.get("validation") or {}
        rows.append({"reaction_id": doc.get("reaction_id"),
                     "terminal_class": doc.get("terminal_class"),
                     "n_candidates": candidates.get("n", 0),
                     "evidence_classes": candidates.get("evidence_classes", []),
                     "completed_interval": candidates.get("completed_interval"),
                     "validation_status": validation.get("status"),
                     "validation_blocked_needs": validation.get("needs", [])})
    return rows


def _cost_summary(terminals: list[dict[str, Any]],
                  case_costs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_gradient = 0
    per_case_gradient: dict[str, int] = {}
    cpu_values: list[float] = []
    cache_hits = 0
    ledgers_total = 0
    failed_gradient = 0
    failed_classes = {"rejected_typed", "censored_budget"}
    for doc in terminals:
        rid = doc.get("reaction_id")
        bundle = case_costs.get(rid) or {}
        used = int((bundle.get("aggregate") or {}).get("N_gradient") or 0)
        total_gradient += used
        per_case_gradient[rid] = used
        cpu = (bundle.get("aggregate") or {}).get("cpu_seconds")
        if isinstance(cpu, (int, float)):
            cpu_values.append(float(cpu))
        for ledger in bundle.get("ledgers", []) or []:
            ledgers_total += 1
            if str(ledger.get("cost_source", "")).startswith("cache_hit"):
                cache_hits += 1
        if doc.get("terminal_class") in failed_classes:
            failed_gradient += used
    most_expensive = max(per_case_gradient.items(), key=lambda kv: kv[1]) \
        if per_case_gradient else None
    return {
        "N_gradient_total": total_gradient,
        "N_gradient_median_per_case": _median(list(per_case_gradient.values())),
        "cpu_seconds_total_known": round(sum(cpu_values), 3) if cpu_values else None,
        "cpu_seconds_known_cases": len(cpu_values),
        "most_expensive_case": {"reaction_id": most_expensive[0],
                                "N_gradient": most_expensive[1]}
        if most_expensive else None,
        "cache_hit_rate": (cache_hits/ledgers_total) if ledgers_total else None,
        "failed_path_gradient_share": (failed_gradient/total_gradient)
        if total_gradient else None,
        "r5_rule": "every reported case carries costs.json; missing costs are "
                   "null+reason in the ledger, never zero",
    }


def _minimal_j_g(terminals: list[dict[str, Any]]) -> dict[str, Any]:
    validated = [doc for doc in terminals
                 if doc.get("terminal_class") in VALIDATED_TERMINAL_CLASSES]
    if not validated:
        return dict(_NO_DATA_J_G)
    verified = sum(1 for doc in validated
                   if doc.get("label_state") == "verified_target")
    return {"available": True, "minimal_only": True,
            "n_validated": len(validated), "n_verified_target": verified,
            "note": "minimal counting version only; the full J_G/R_oracle "
                    "protocol belongs to the G3.0 precondition batch"}


def _ab3_improvement_list(histogram: list[dict[str, Any]]) -> list[dict[str, Any]]:
    improvements = []
    for row in histogram[:5]:
        improvements.append({
            "failure_code": row["failure_code"], "n_cases": row["n_cases"],
            "evidence_cases": row["cases"][:8],
            "candidate_fix_direction": _FAILURE_FIX_DIRECTIONS.get(
                row["failure_code"],
                "diagnose with segment diagnostics; keep the typed failure and "
                "its cost ledger"),
        })
    return improvements


def build_feedback_report(campaign_root: Path | str, *,
                          manifest_path: Path | str | None = None) -> dict[str, Any]:
    """Build ``pes2ts_g2t_feedback_report_v1`` from one campaign root."""
    root = Path(campaign_root)
    if not (root / "campaign_identity.json").is_file():
        raise ValueError(f"NOT_A_CAMPAIGN_ROOT:{root}")
    terminals = []
    case_root: dict[str, Path] = {}
    for path in sorted(root.glob("RXN_*/terminal_state.json")):
        if path.parent.name.split(".")[0] != path.parent.name:
            continue
        doc = _read(path)
        if doc.get("schema_version") != TERMINAL_STATE_SCHEMA:
            raise ValueError(f"INVALID_TERMINAL_STATE:{path.parent.name}")
        terminals.append(doc)
        case_root[doc["reaction_id"]] = path.parent
    if not terminals:
        raise ValueError("NO_TERMINAL_STATES")
    case_set_source = "terminals_glob"
    if manifest_path is not None:
        manifest = load_campaign_manifest(manifest_path)
        campaign_sha = manifest["content_sha256"]
        manifest_ids = {case["reaction_id"] for case in manifest["cases"]}
        for doc in terminals:
            if doc.get("campaign_sha256") != campaign_sha:
                raise ValueError(
                    f"TERMINAL_CAMPAIGN_SHA_MISMATCH:{doc.get('reaction_id')}")
        extra = sorted({doc["reaction_id"] for doc in terminals} - manifest_ids)
        if extra:
            raise ValueError(f"MANIFEST_CASE_SET_MISMATCH:extra={extra}")
        for rid, path in case_root.items():
            if path.name != rid:
                raise ValueError(f"CASE_DIR_NAME_MISMATCH:{rid}:{path.name}")
        # G2-AB3 WP-0 audit fix P2-2: every table below shares the exact
        # manifest∩terminal case set; a partial campaign is reportable but
        # its missing cases are explicit, and directories outside the
        # manifest (e.g. RXN_*.crashed_v1) never enter any statistic.
        terminal_ids = {doc["reaction_id"] for doc in terminals}
        missing_cases = sorted(manifest_ids - terminal_ids)
        if missing_cases:
            terminals = [doc for doc in terminals if doc["reaction_id"] in manifest_ids]
            case_root = {rid: path for rid, path in case_root.items() if rid in manifest_ids}
        case_set_source = "manifest_verified"
    else:
        missing_cases = []
    case_costs = {}
    for rid, path in case_root.items():
        costs_path = path / "costs.json"
        if costs_path.is_file():
            case_costs[rid] = _read(costs_path)
    references: dict[str, Any] = {}
    if manifest_path is not None:
        references = manifest.get("reference_failures") or {}
    histogram = _failure_histogram(terminals, case_costs)
    report = {
        "schema_version": FEEDBACK_REPORT_SCHEMA,
        "campaign_root": str(root),
        "case_set_source": case_set_source,
        "n_cases": len(terminals),
        "n_manifest_cases_missing": len(missing_cases),
        "manifest_missing_cases": missing_cases,
        "funnel": _funnel(terminals),
        "typed_failure_histogram": histogram,
        "b1_comparison": _b1_checks(terminals, case_root, references),
        "branch_evidence": _branch_evidence(case_root),
        "candidates_and_validation": _candidates_and_validation(terminals),
        "cost_summary": _cost_summary(terminals, case_costs),
        "minimal_j_g": _minimal_j_g(terminals),
        "ab3_improvement_list": _ab3_improvement_list(histogram),
        "capability_status": CAPABILITY_STATUS,
        "claims": {"m0_through_m5_claimed": False,
                   "success_rate_claimed": False,
                   "note": "first-round value is the failure distribution and "
                           "cost feedback, not a success rate"},
    }
    return report


__all__ = ["CAPABILITY_STATUS", "FEEDBACK_REPORT_SCHEMA", "build_feedback_report"]
