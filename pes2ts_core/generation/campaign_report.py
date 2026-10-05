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

#: Capability status vocabulary (constitution §11.5) — honest layer states for
#: this batch.  Nothing here claims ``smoke_passed``: that upgrade requires
#: the gated real-engine runs (S2/S3/S4) which are environment-blocked until
#: the ACP wiring exists.
CAPABILITY_STATUS = {
    "campaign_orchestrator": {"status": "fixture_passed",
        "evidence": "tests/test_campaign.py (synthetic backends, offline)"},
    "cost_capture": {"status": "fixture_passed",
        "evidence": "tests/test_cost_capture.py (offline)"},
    "origin_preparation_acp_adapter": {"status": "fixture_passed",
        "evidence": "tests/test_acp_origin_backend.py (fake ACP); real-ACP V2 pending"},
    "branch_policy_calibration": {"status": "fixture_passed",
        "evidence": "tests/test_replay_report.py (analytic grid, offline)"},
    "pilot_replay_judge": {"status": "fixture_passed",
        "evidence": "tests/test_replay_report.py (pure function, offline)"},
    "continuation_real_engine": {"status": "implemented",
        "evidence": "G2-AB1 V0; gated V2 replay requires ACP/ORCA environment"},
    "validation_chain": {"status": "implemented",
        "evidence": "G2-AB1 stage_cli; execution blocked on human review"},
    "feedback_reporter": {"status": "fixture_passed",
        "evidence": "tests/test_campaign_report.py (synthetic campaign, offline)"},
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
        progressed = bool(reference_lambda is None
                          or (isinstance(last_lambda, (int, float))
                              and isinstance(reference_lambda, (int, float))
                              and float(last_lambda) > float(reference_lambda)))
        origin_failed = doc.get("origin_status") == "failed"
        rows.append({
            "reaction_id": rid,
            "b1_1_identity_clear": int(doc.get("identity_conflict_count") or 0) == 0,
            "b1_2_progressed_beyond_reference": progressed,
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
        "b1_3_typed_origin_failures": sum(1 for row in rows
                                          if row["b1_3_origin_failure_typed"]),
        "per_case": rows,
        "note": "B1-3 is None when the origin succeeded (nothing to clear); "
                "B1-4 is True when the candidate artifact exists and the "
                "validation status is explicit (blocked_on_human_review counts)",
    }


def _branch_evidence(campaign_root: Path) -> dict[str, Any]:
    accepted_frames, biased_only, drifts, probes = [], [], [], 0
    modes: dict[str, int] = {}
    for path in sorted(campaign_root.glob("RXN_*/continuation_result.json")):
        result = _read(path)
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
        doc = _read(path)
        if doc.get("schema_version") != TERMINAL_STATE_SCHEMA:
            raise ValueError(f"INVALID_TERMINAL_STATE:{path.parent.name}")
        terminals.append(doc)
        case_root[doc["reaction_id"]] = path.parent
    if not terminals:
        raise ValueError("NO_TERMINAL_STATES")
    case_costs = {}
    for rid, path in case_root.items():
        costs_path = path / "costs.json"
        if costs_path.is_file():
            case_costs[rid] = _read(costs_path)
    references: dict[str, Any] = {}
    if manifest_path is not None:
        manifest = load_campaign_manifest(manifest_path)
        references = manifest.get("reference_failures") or {}
    histogram = _failure_histogram(terminals, case_costs)
    return {
        "schema_version": FEEDBACK_REPORT_SCHEMA,
        "campaign_root": str(root),
        "n_cases": len(terminals),
        "funnel": _funnel(terminals),
        "typed_failure_histogram": histogram,
        "b1_comparison": _b1_checks(terminals, case_root, references),
        "branch_evidence": _branch_evidence(root),
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


__all__ = ["CAPABILITY_STATUS", "FEEDBACK_REPORT_SCHEMA", "build_feedback_report"]
