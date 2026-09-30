"""Case-grain generation, ranking, end-to-end, and cost metrics for Demo runs."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from pes2ts_core.contracts import ContractError, dumps_document
from pes2ts_core.utils.hashing import stable_json_dumps


def load_demo_run_bundle(bundle_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a path-referenced demo bundle; all input files must stay under its root."""
    source = Path(bundle_path).resolve(strict=True)
    root = source.parent
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid DemoRunBundle file: {exc}") from exc
    if (not isinstance(manifest, dict)
            or manifest.get("schema_version") != "pes2ts_demo_run_bundle_v1"
            or not isinstance(manifest.get("runs"), list)
            or not isinstance(manifest.get("cohort_cases"), list)):
        raise ContractError("DemoRunBundle requires schema_version, cohort_cases, and runs arrays")

    def read_ref(reference: Any, label: str) -> Any:
        if reference is None:
            return None
        if not isinstance(reference, str) or not reference:
            raise ContractError(f"{label} must be a bundle-relative path or null")
        try:
            path = (root / reference).resolve(strict=True)
            path.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ContractError(f"{label} is missing or escapes the bundle root") from exc
        if not path.is_file():
            raise ContractError(f"{label} must reference a file")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid JSON at {label}: {exc}") from exc

    runs = []
    for index, row in enumerate(manifest["runs"]):
        if not isinstance(row, dict) or not isinstance(row.get("proposals", []), list):
            raise ContractError(f"DemoRunBundle run {index} must be an object with proposals array")
        loaded = {key: read_ref(row.get(key), f"run {index} {key}")
                  for key in ("case", "plan", "execution", "path", "validation")}
        if loaded["case"] is None:
            raise ContractError(f"DemoRunBundle run {index} requires a case reference")
        loaded["proposals"] = [read_ref(ref, f"run {index} proposal {proposal_index}")
                               for proposal_index, ref in enumerate(row.get("proposals", []))]
        runs.append(loaded)
    cohort_cases = [read_ref(ref, f"cohort case {index}")
                    for index, ref in enumerate(manifest["cohort_cases"])]
    if not all(isinstance(case, dict) for case in cohort_cases):
        raise ContractError("cohort_cases entries must reference ReactionCase documents")
    loaded_labels = read_ref(manifest.get("ranking_labels"), "ranking_labels")
    labels = [] if loaded_labels is None else loaded_labels
    if not isinstance(labels, list):
        raise ContractError("ranking_labels file must contain a JSON array")
    return runs, labels, cohort_cases


def _check_document(document: dict[str, Any], kind: str, label: str) -> None:
    try:
        dumps_document(document)
    except ContractError as exc:
        raise ContractError(f"{label} ({kind}): {exc}") from exc
    if document.get("schema_name") != kind:
        raise ContractError(f"{label} must be a {kind}")


def _known_validation_cpu(validation: dict[str, Any]) -> tuple[float, bool]:
    if validation["status"] == "not_run":
        return 0.0, True
    extensions = validation.get("extensions")
    if not isinstance(extensions, dict):
        return 0.0, False
    ledger = extensions.get("pes2ts.validation_costs.v1")
    if not isinstance(ledger, dict) or not isinstance(ledger.get("stages"), dict):
        return 0.0, False
    known = 0.0
    for stage in ledger["stages"].values():
        attempts = stage.get("attempts") if isinstance(stage, dict) else None
        if not isinstance(attempts, list):
            continue
        known += sum(attempt["cpu_seconds"] for attempt in attempts
                     if isinstance(attempt, dict)
                     and isinstance(attempt.get("cpu_seconds"), (int, float))
                     and not isinstance(attempt.get("cpu_seconds"), bool)
                     and math.isfinite(attempt["cpu_seconds"]) and attempt["cpu_seconds"] >= 0)
    complete = (ledger.get("cost_complete") is True
                and isinstance(ledger.get("total_cpu_seconds"), (int, float))
                and not isinstance(ledger.get("total_cpu_seconds"), bool)
                and math.isfinite(ledger["total_cpu_seconds"])
                and ledger["total_cpu_seconds"] >= 0)
    return known, complete


def _case_cost(run: dict[str, Any]) -> tuple[float, bool]:
    known = 0.0
    complete = True
    if run.get("_run_record_missing") is True:
        return known, False
    execution = run.get("execution")
    if execution is not None:
        known += sum(attempt["cpu_seconds"] for attempt in execution["attempts"]
                     if attempt.get("cpu_seconds") is not None)
        complete &= (execution["cost_complete"] is True
                     and execution["total_cpu_seconds"] is not None)
    elif run.get("plan") is not None and run["plan"]["status"] == "ready":
        complete = False
    validation = run.get("validation")
    if validation is not None:
        validation_known, validation_complete = _known_validation_cpu(validation)
        known += validation_known
        complete &= validation_complete
    elif run.get("path") is not None and run["path"]["status"] == "usable":
        complete = False
    return known, complete


def _case_wall_time(run: dict[str, Any]) -> tuple[float, bool]:
    """Return measured CLI wall time; do not infer it from plan budgets."""
    if run.get("_run_record_missing") is True:
        return 0.0, False
    execution = run.get("execution")
    if execution is None:
        plan = run.get("plan")
        return 0.0, not (plan is not None and plan.get("status") == "ready")
    extensions = execution.get("extensions")
    receipts_doc = extensions.get("pes2ts.acp_cli_attempt_receipts.v1") if isinstance(extensions, dict) else None
    receipts = receipts_doc.get("attempts") if isinstance(receipts_doc, dict) else None
    attempts = execution.get("attempts")
    if not isinstance(receipts, list) or not isinstance(attempts, list):
        return 0.0, False
    known = 0.0
    known_ids: set[str] = set()
    complete = len(receipts) == len(attempts)
    for receipt in receipts:
        if not isinstance(receipt, dict):
            complete = False
            continue
        attempt_id = receipt.get("attempt_id")
        wall = receipt.get("wall_seconds")
        if (not isinstance(attempt_id, str) or attempt_id in known_ids
                or not isinstance(wall, (int, float)) or isinstance(wall, bool)
                or not math.isfinite(wall) or wall < 0):
            complete = False
            continue
        known_ids.add(attempt_id)
        known += float(wall)
    if {attempt.get("attempt_id") for attempt in attempts if isinstance(attempt, dict)} != known_ids:
        complete = False
    return known, complete


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_demo_runs(*, runs: Iterable[dict[str, Any]], cohort_cases: Iterable[dict[str, Any]],
                       ranking_labels: Iterable[dict[str, Any]] = (),
                       ranking_rule: str = "highest_scan_energy",
                       evaluation_split: str = "valid",
                       top_ks: tuple[int, ...] = (1, 3)) -> dict[str, Any]:
    """Summarize one frozen run per ReactionCase without candidate-level inflation.

    Ranking labels are supplied separately from production contracts and are
    scored only on ``evaluation_split``. Missing labels or paths are reported,
    never silently treated as misses in the conditional ranking denominator.
    This function does not set targets or infer labels from the generated data.
    """
    if evaluation_split not in {"train", "valid", "test", "unassigned"}:
        raise ContractError("evaluation_split is invalid")
    if (not isinstance(top_ks, tuple) or not top_ks or any(
            not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in top_ks)
            or tuple(sorted(set(top_ks))) != top_ks):
        raise ContractError("top_ks must be a sorted tuple of unique positive integers")
    cohort_by_id: dict[str, dict[str, Any]] = {}
    cohort_reactions: set[str] = set()
    for index, case in enumerate(cohort_cases):
        if not isinstance(case, dict):
            raise ContractError(f"cohort case {index} must be a ReactionCase")
        _check_document(case, "ReactionCase", f"cohort case {index}")
        if case["case_id"] in cohort_by_id:
            raise ContractError(f"duplicate frozen cohort case_id {case['case_id']}")
        if case["reaction_id"] in cohort_reactions:
            raise ContractError(f"frozen Demo cohort must have one ReactionCase per reaction_id: {case['reaction_id']}")
        cohort_by_id[case["case_id"]] = case
        cohort_reactions.add(case["reaction_id"])
    if not cohort_by_id:
        raise ContractError("frozen cohort_cases must not be empty")
    normalized: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(runs):
        if not isinstance(raw, dict) or not isinstance(raw.get("case"), dict):
            raise ContractError(f"run {index} must include a ReactionCase")
        case = raw["case"]
        _check_document(case, "ReactionCase", f"run {index} case")
        case_id = case["case_id"]
        cohort_case = cohort_by_id.get(case_id)
        if cohort_case is None:
            raise ContractError(f"run case {case_id} is outside the frozen cohort")
        if (case.get("content_sha256") != cohort_case.get("content_sha256")
                or case.get("reaction_id") != cohort_case.get("reaction_id")
                or case.get("split") != cohort_case.get("split")):
            raise ContractError(f"run case {case_id} differs from its frozen cohort ReactionCase")
        if case_id in normalized:
            raise ContractError(f"multiple run rows for ReactionCase {case_id}; aggregate candidates before evaluation")
        run = {key: raw.get(key) for key in ("plan", "execution", "path", "validation")}
        run["case"] = case
        plan, execution, path, validation = (run[key] for key in ("plan", "execution", "path", "validation"))
        if plan is not None:
            _check_document(plan, "ScanPlan", f"run {case_id} plan")
            if (plan.get("case_id") != case_id or plan.get("reaction_id") != case["reaction_id"]
                    or plan.get("split") != case["split"]
                    or plan.get("dataset_version") != case.get("dataset_version")
                    or plan.get("source_case_sha256") != case.get("content_sha256")):
                raise ContractError(f"run {case_id} plan identity/split does not match ReactionCase")
            if plan["status"] == "ready" and case["status"] != "ready":
                raise ContractError(f"run {case_id} has a ready plan from a non-ready ReactionCase")
        if execution is not None:
            _check_document(execution, "ExecutionRecord", f"run {case_id} execution")
            if (plan is None or execution.get("plan_id") != plan.get("plan_id")
                    or execution.get("case_id") != case_id
                    or execution.get("reaction_id") != case["reaction_id"]):
                raise ContractError(f"run {case_id} execution is not bound to its case plan")
            candidate_ids = {candidate["candidate_id"] for candidate in plan.get("candidates", [])}
            if execution.get("candidate_id") not in candidate_ids:
                raise ContractError(f"run {case_id} execution candidate is absent from its ScanPlan")
            expected_request_id = next(candidate["request_id"] for candidate in plan["candidates"]
                                       if candidate["candidate_id"] == execution["candidate_id"])
            if execution.get("request_id") != expected_request_id:
                raise ContractError(f"run {case_id} execution request_id does not match its ScanPlan candidate")
        if path is not None:
            _check_document(path, "PathBundle", f"run {case_id} path")
            if (plan is None or execution is None
                    or path.get("plan_id") != plan.get("plan_id")
                    or path.get("execution_id") != execution.get("object_id")
                    or path.get("acp_task_id") != execution.get("acp_task_id")
                    or path.get("candidate_id") != execution.get("candidate_id")
                    or path.get("case_id") != case_id
                    or path.get("reaction_id") != case["reaction_id"]):
                raise ContractError(f"run {case_id} PathBundle is not bound to its ExecutionRecord")
        proposals = raw.get("proposals", [])
        if not isinstance(proposals, list):
            raise ContractError(f"run {case_id} proposals must be an array")
        by_rule = {}
        for proposal in proposals:
            _check_document(proposal, "SeedProposal", f"run {case_id} proposal")
            if (path is None or proposal.get("path_id") != path.get("object_id")
                    or proposal.get("path_content_sha256") != path.get("content_sha256")
                    or proposal.get("path_status") != path.get("status")):
                raise ContractError(f"run {case_id} proposal is stale or bound to a different path")
            if proposal.get("rule") in by_rule:
                raise ContractError(f"run {case_id} has duplicate proposal rule {proposal.get('rule')}")
            by_rule[proposal.get("rule")] = proposal
        run["proposals_by_rule"] = by_rule
        if validation is not None:
            _check_document(validation, "ValidationResult", f"run {case_id} validation")
            source_proposal = next((p for p in proposals
                                    if p["object_id"] == validation.get("proposal_id")), None)
            source_frame = None
            if path is not None:
                source_frame = next((frame for frame in path.get("frames", [])
                                     if frame.get("frame_id") == validation.get("source_frame_id")), None)
            source_geometry_digest = (hashlib.sha256(stable_json_dumps(source_frame["geometry"]).encode()).hexdigest()
                                      if source_frame is not None else None)
            if (path is None or validation.get("path_id") != path.get("object_id")
                    or validation.get("case_id") != case_id
                    or validation.get("reaction_id") != case["reaction_id"]
                    or source_proposal is None
                    or (validation.get("status") == "passed" and (
                        validation.get("path_content_sha256") != path.get("content_sha256")
                        or validation.get("proposal_content_sha256") != source_proposal.get("content_sha256")
                        or validation.get("source_frame_id") != (source_proposal.get("selected_frames") or [{}])[0].get("frame_id")
                        or validation.get("source_geometry_sha256") != source_geometry_digest
                        or (source_proposal.get("selected_frames") or [{}])[0].get("geometry_sha256") != source_geometry_digest))
                    or (validation["status"] == "passed" and
                        (path["status"] != "usable" or source_proposal["status"] != "accepted"))):
                raise ContractError(f"run {case_id} ValidationResult is stale or not bound to the exact path/proposal/frame geometry")
        normalized[case_id] = run

    run_record_count = len(normalized)
    for case_id, case in cohort_by_id.items():
        normalized.setdefault(case_id, {"case": case, "plan": None, "execution": None,
                                        "path": None, "validation": None, "proposals_by_rule": {},
                                        "_run_record_missing": True})

    label_by_case: dict[str, set[str]] = {}
    for label in ranking_labels:
        if not isinstance(label, dict):
            raise ContractError("ranking labels must be objects")
        case_id = label.get("case_id")
        if case_id not in normalized:
            raise ContractError(f"ranking label references unknown case_id {case_id!r}")
        if (not isinstance(label.get("label_source"), str) or not label["label_source"].strip()
                or not isinstance(label.get("method"), str) or not label["method"].strip()
                or not isinstance(label.get("source_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", label["source_sha256"])):
            raise ContractError(f"ranking label for {case_id} requires a method, source, and lowercase source SHA256")
        if normalized[case_id]["case"]["split"] != evaluation_split:
            raise ContractError(f"ranking label for {case_id} is outside evaluation split {evaluation_split}")
        frame_ids = label.get("acceptable_frame_ids")
        if (case_id in label_by_case or not isinstance(frame_ids, list) or not frame_ids
                or any(not isinstance(frame_id, str) or not frame_id for frame_id in frame_ids)
                or len(set(frame_ids)) != len(frame_ids)):
            raise ContractError(f"ranking label for {case_id} must have unique acceptable frame IDs")
        path = normalized[case_id].get("path")
        if path is not None:
            path_frame_ids = {frame["frame_id"] for frame in path["frames"]}
            if not set(frame_ids) <= path_frame_ids:
                raise ContractError(f"ranking label for {case_id} references a frame outside its PathBundle")
        label_by_case[case_id] = set(frame_ids)

    split_rows: dict[str, list[dict[str, Any]]] = {}
    for run in normalized.values():
        split_rows.setdefault(run["case"]["split"], []).append(run)
    generation_by_split: dict[str, Any] = {}
    end_to_end_by_split: dict[str, Any] = {}
    for split in ("train", "valid", "test", "unassigned"):
        rows = split_rows.get(split, [])
        n_cases = len(rows)
        ready_plans = sum(row.get("plan") is not None and row["plan"]["status"] == "ready" for row in rows)
        completed = sum(row.get("execution") is not None and row["execution"]["status"] == "completed" for row in rows)
        usable_paths = sum(row.get("path") is not None and row["path"]["status"] == "usable" for row in rows)
        plan_rejected = sum(row.get("plan") is not None and row["plan"]["status"] == "rejected" for row in rows)
        failed_exec = sum(row.get("execution") is not None and row["execution"]["status"] in {"failed", "cancelled"} for row in rows)
        validation_counts = Counter(row["validation"]["status"] if row.get("validation") is not None
                                    else "missing" for row in rows)
        validation_failure_notes = Counter(
            row["validation"].get("validation_note", "unspecified") for row in rows
            if row.get("validation") is not None and row["validation"]["status"] == "failed")
        rejected_reasons = Counter(reason for row in rows if row.get("plan") is not None
                                   for reason in row["plan"].get("reject_reasons", []))
        execution_failure_codes = Counter(attempt.get("failure_code") for row in rows
            if row.get("execution") is not None for attempt in row["execution"]["attempts"]
            if attempt["status"] == "failed")
        path_status_counts = Counter(row["path"]["status"] if row.get("path") is not None
                                     else "missing" for row in rows)
        passed = validation_counts["passed"]
        generation_by_split[split] = {
            "case_count": n_cases,
            "ready_plan_cases": ready_plans,
            "ready_plan_rate": _rate(ready_plans, n_cases),
            "completed_execution_cases": completed,
            "completed_execution_rate": _rate(completed, n_cases),
            "usable_path_cases": usable_paths,
            "usable_path_coverage": _rate(usable_paths, n_cases),
            "rejected_plan_cases": plan_rejected,
            "failed_or_cancelled_execution_cases": failed_exec,
            "failure_analysis": {
                "plan_rejection_reasons": dict(sorted(rejected_reasons.items())),
                "execution_failure_codes": dict(sorted(execution_failure_codes.items())),
                "path_status_counts": dict(sorted(path_status_counts.items())),
            },
        }
        end_to_end_by_split[split] = {
            "case_count": n_cases,
            "validation_status_counts": dict(sorted(validation_counts.items())),
            "validation_failure_notes": dict(sorted(validation_failure_notes.items())),
            "passed_cases": passed,
            "pass_rate": _rate(passed, n_cases),
        }

    eval_rows = [row for row in normalized.values() if row["case"]["split"] == evaluation_split]
    labeled_ids = {row["case"]["case_id"] for row in eval_rows if row["case"]["case_id"] in label_by_case}
    eligible = [row for row in eval_rows if row["case"]["case_id"] in label_by_case
                and row.get("path") is not None and row["path"]["status"] == "usable"]
    hits = {k: 0 for k in top_ks}
    reciprocal_ranks = []
    proposals_present = 0
    for row in eligible:
        proposal = row["proposals_by_rule"].get(ranking_rule)
        if (proposal is None or proposal.get("selection_source") != "ranking"
                or not proposal.get("selected_frames")):
            reciprocal_ranks.append(0.0)
            continue
        proposals_present += 1
        selected = proposal["selected_frames"]
        ranked_ids = [item["frame_id"] for item in selected]
        acceptable = label_by_case[row["case"]["case_id"]]
        first_relevant = next((index for index, frame_id in enumerate(ranked_ids, start=1)
                               if frame_id in acceptable), None)
        reciprocal_ranks.append(1.0 / first_relevant if first_relevant is not None else 0.0)
        for k in top_ks:
            if any(frame_id in acceptable for frame_id in ranked_ids[:k]):
                hits[k] += 1
    ranking_metrics = {
        "status": "measured" if eligible else "not_measured",
        "evaluation_split": evaluation_split,
        "ranking_rule": ranking_rule,
        "top_ks": list(top_ks),
        "labeled_case_count": len(labeled_ids),
        "labeled_cases_without_usable_path": len(labeled_ids) - len(eligible),
        "usable_labeled_case_count": len(eligible),
        "proposal_cases": proposals_present,
        "proposal_coverage": _rate(proposals_present, len(eligible)),
        "recall_at_k": {str(k): _rate(hits[k], len(eligible)) for k in top_ks},
        "mean_reciprocal_rank": (sum(reciprocal_ranks) / len(eligible) if eligible else None),
    }

    cost_complete_cases = 0
    known_cpu_total = 0.0
    wall_time_complete_cases = 0
    known_wall_total = 0.0
    for row in normalized.values():
        known, complete = _case_cost(row)
        known_cpu_total += known
        cost_complete_cases += int(complete)
        known_wall, wall_complete = _case_wall_time(row)
        known_wall_total += known_wall
        wall_time_complete_cases += int(wall_complete)
    case_count = len(normalized)
    all_costs_complete = cost_complete_cases == case_count
    all_wall_times_complete = wall_time_complete_cases == case_count
    return {
        "schema_version": "pes2ts_demo_metrics_v1",
        "measurement_grain": "reaction_id",
        "case_snapshot_unit": "ReactionCase",
        "ranking_labels_separate_from_production_contracts": True,
        "cohort": {"case_count": case_count,
                   "run_record_cases": run_record_count,
                   "missing_run_record_cases": case_count - run_record_count,
                   "run_record_coverage": _rate(run_record_count, case_count),
                   "split_counts": {split: len(split_rows.get(split, []))
                                    for split in ("train", "valid", "test", "unassigned")}},
        "generation": {"by_split": generation_by_split},
        "ranking": ranking_metrics,
        "end_to_end": {"by_split": end_to_end_by_split},
        "costs": {
            "case_count": case_count,
            "cost_complete_cases": cost_complete_cases,
            "cost_incomplete_cases": case_count - cost_complete_cases,
            "known_cpu_seconds_lower_bound": known_cpu_total,
            "total_cpu_seconds": known_cpu_total if all_costs_complete else None,
            "mean_cpu_seconds_per_case": (known_cpu_total / case_count if all_costs_complete and case_count else None),
            "wall_time_complete_cases": wall_time_complete_cases,
            "wall_time_incomplete_cases": case_count - wall_time_complete_cases,
            "known_wall_seconds_lower_bound": known_wall_total,
            "total_wall_seconds": known_wall_total if all_wall_times_complete else None,
            "mean_wall_seconds_per_case": (known_wall_total / case_count
                if all_wall_times_complete and case_count else None),
        },
    }
