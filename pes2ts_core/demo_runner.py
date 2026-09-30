"""Train-first cohort execution for the frozen PES2TS Demo workflow."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any

from pes2ts_core.contracts import ContractError, dumps_document, seal_document
from pes2ts_core.demo_metrics import evaluate_demo_runs
from pes2ts_core.integration.acp.adapter import scan_plan_to_acp_request
from pes2ts_core.integration.acp.adapter import acp_execution_to_record
from pes2ts_core.integration.acp.cli_backend import (
    ACPCLIBackend, ACPCLIError, cli_result_to_execution_record,
    collect_cli_path_bundle,
)
from pes2ts_core.integration.acp.quality import apply_scan_path_quality
from pes2ts_core.planning import build_scan_plan_with_rejection
from pes2ts_core.ranking import rank_path_bundle
from pes2ts_core.utils.hashing import stable_json_dumps
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.viewer import render_path_viewer


EXECUTION_SCHEMA = "pes2ts_demo_execution_manifest_v1"
BUNDLE_SCHEMA = "pes2ts_demo_run_bundle_v1"


def _read_ref(root: Path, reference: Any, label: str, *, required: bool = True) -> dict[str, Any] | None:
    if reference is None and not required:
        return None
    if not isinstance(reference, str) or not reference:
        raise ContractError(f"{label} must be a bundle-relative JSON path")
    value = reference.replace("\\", "/")
    if value.startswith("/") or ":" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ContractError(f"{label} contains an unsafe path")
    path = (root / Path(*PurePosixPath(value).parts)).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ContractError(f"{label} is missing or escapes the manifest directory")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return result


def _read_array_ref(root: Path, reference: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(reference, str) or not reference:
        raise ContractError(f"{label} must be a bundle-relative JSON path")
    value = reference.replace("\\", "/")
    if value.startswith("/") or ":" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ContractError(f"{label} contains an unsafe path")
    path = (root / Path(*PurePosixPath(value).parts)).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ContractError(f"{label} is missing or escapes the manifest directory")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
        raise ContractError(f"{label} must contain an array of ranking-label objects")
    return result


def _validate_train_index(path: str | Path | None, *, manifest_sha256: str,
                         input_fingerprint: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if path is None:
        raise ContractError("valid evaluation requires --train-index from a completed train-only run")
    index_path = Path(path).expanduser().resolve(strict=True)
    index = _read_ref(index_path.parent, index_path.name, "train_index")
    if (index.get("schema_version") != EXECUTION_SCHEMA
            or index.get("manifest_sha256") != manifest_sha256
            or index.get("input_fingerprint_sha256") != input_fingerprint
            or index.get("selected_splits") != ["train"]
            or index.get("cohort_case_count") != len(cases)
            or not isinstance(index.get("runs"), list)):
        raise ContractError("train index must use the exact frozen input manifest and train-only selection")
    expected = {case["case_id"]: case["split"] for case in cases}
    rows: dict[str, dict[str, Any]] = {}
    for row in index["runs"]:
        if not isinstance(row, dict) or row.get("case_id") not in expected or row["case_id"] in rows:
            raise ContractError("train index runs do not match the unique frozen ReactionCase cohort")
        rows[row["case_id"]] = row
    if set(rows) != set(expected):
        raise ContractError("train index is missing frozen cohort rows")
    for case_id, split in expected.items():
        row = rows[case_id]
        if split == "valid" and row.get("status") != "valid_held_out":
            raise ContractError("train index must show every valid case as held out")
        if split == "train" and row.get("review_status") in (None, "pending"):
            raise ContractError("train index must account for a completed human review decision per train case")
    bundle_ref = index.get("bundle_ref")
    bundle = _read_ref(index_path.parent, bundle_ref, "train_index.bundle_ref")
    bundle_path = index_path.parent / Path(*PurePosixPath(bundle_ref).parts)
    from pes2ts_core.demo_metrics import evaluate_demo_runs, load_demo_run_bundle
    bundle_runs, bundle_labels, cohort = load_demo_run_bundle(bundle_path)
    cohort_by_id = {case["case_id"]:case for case in cohort}
    if set(cohort_by_id) != set(expected) or any(
            cohort_by_id[case_id].get("content_sha256") != case["content_sha256"]
            for case_id, case in ((case["case_id"], case) for case in cases)):
        raise ContractError("train bundle cohort differs from the current frozen case snapshots")
    run_by_id = {run["case"]["case_id"]:run for run in bundle_runs}
    if set(run_by_id) != set(expected):
        raise ContractError("train bundle must contain one auditable run row per frozen case")
    bundle_rows: dict[str, dict[str, Any]] = {}
    for bundle_row in bundle.get("runs", []):
        if not isinstance(bundle_row, dict):
            raise ContractError("train bundle run row must be an object")
        case_doc = _read_ref(index_path.parent, bundle_row.get("case"), "train bundle run case")
        case_id = case_doc["case_id"]
        if case_id in bundle_rows or case_id not in expected:
            raise ContractError("train bundle rows must match unique frozen ReactionCases")
        if bundle_row.get("run_status") != rows[case_id].get("status"):
            raise ContractError("train index status differs from its auditable run bundle")
        bundle_rows[case_id] = bundle_row
    if set(bundle_rows) != set(expected):
        raise ContractError("train bundle is missing frozen cohort rows")
    if bundle.get("ranking_labels") is not None and not isinstance(bundle.get("ranking_labels"), str):
        raise ContractError("train bundle ranking_labels reference is malformed")
    evaluate_demo_runs(runs=bundle_runs, cohort_cases=cohort,
        ranking_labels=bundle_labels, evaluation_split="valid")
    return {"index":index, "index_path":index_path, "root":index_path.parent, "index_rows":rows,
        "runs":run_by_id, "bundle_rows":bundle_rows, "bundle_labels":bundle_labels}


def _copy_run_tree(source_root: Path, case_ref: str, target_root: Path) -> None:
    value = case_ref.replace("\\", "/")
    if not value.startswith("runs/"):
        raise ContractError("train bundle case reference must stay under runs/")
    source_case = (source_root / Path(*PurePosixPath(value).parts)).resolve(strict=True)
    if not source_case.is_relative_to(source_root.resolve()) or not source_case.is_file():
        raise ContractError("train bundle case reference is missing or escapes its root")
    source_dir = source_case.parent
    destination = target_root / Path(*PurePosixPath(value).parts).parent
    if destination.exists():
        raise ContractError("refusing to overwrite copied train artifacts")
    for path in source_dir.rglob("*"):
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(source_root.resolve()):
            raise ContractError("train run artifacts contain a symlink or escaping reference")
    shutil.copytree(source_dir, destination)


def _portable_stem(index: int, case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._-") or "case"
    # Keep nested ACP WORK/RESULT paths below Windows' legacy MAX_PATH limit.
    # The cohort index still makes stems unique when IDs share a prefix.
    return f"{index:03d}_{safe[:32]}"


def _python_tree_sha256(root: Path) -> str:
    """Fingerprint ACP Python sources so train results cannot mask code drift."""
    source_root = (root / "src" / "acp").resolve(strict=True)
    entries = []
    for path in sorted(source_root.rglob("*.py")):
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(source_root):
            raise ContractError("ACP source tree contains a symlink or escaping Python module")
        entries.append((path.relative_to(source_root).as_posix(),
                        hashlib.sha256(path.read_bytes()).hexdigest()))
    if not entries:
        raise ContractError("ACP source tree contains no Python modules")
    return hashlib.sha256(stable_json_dumps(entries).encode()).hexdigest()


def _accepted_review(case: dict[str, Any], review: dict[str, Any]) -> bool:
    source = case.get("source", {})
    extensions = review.get("extensions", {})
    extension = extensions.get("pes2ts.review_output.v1", {}) if isinstance(extensions, dict) else {}
    if not isinstance(source, dict) or not isinstance(extension, dict):
        return False
    return (case.get("status") == "ready"
        and review.get("schema_name") == "ReviewRecord"
        and review.get("status") == "accepted"
        and (review.get("reaction_id"), review.get("case_id"),
             review.get("dataset_version"), review.get("split"))
           == (case.get("reaction_id"), case.get("case_id"),
               case.get("dataset_version"), case.get("split"))
        and source.get("review_record_id") == review.get("object_id")
        and source.get("reviewed_from_case_sha256") == review.get("case_sha256")
        and extension.get("reviewed_case_sha256") == case.get("content_sha256"))


def run_demo_execution_manifest(*, manifest_path: str | Path, output_root: str | Path,
                                acp_root: str | Path,
                                python_executable: str | Path | None = None,
                                config_path: str | Path | None = None,
                                nproc: int | None = None, memory: str | None = None,
                                include_valid: bool = False,
                                train_index_path: str | Path | None = None,
                                ranking_rules: tuple[str, ...] = (
                                    "highest_scan_energy", "internal_scan_peak"),
                                ranking_labels_path: str | Path | None = None,
                                evaluation_split: str = "valid") -> dict[str, Any]:
    """Execute accepted cohort entries sequentially and write an auditable bundle.

    `train` entries run by default. `valid` entries remain in the frozen
    denominator but are not planned or calculated until `include_valid=True`.
    Every path reference in the source manifest is confined to its directory.
    """
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    source_root = manifest_file.parent
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid DemoExecutionManifest: {exc}") from exc
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != EXECUTION_SCHEMA
            or not isinstance(manifest.get("cohort_cases"), list)
            or not isinstance(manifest.get("runs"), list)):
        raise ContractError("DemoExecutionManifest requires the v1 schema, cohort_cases, and runs")
    allowed_manifest_fields = {"schema_version", "cohort_cases", "runs", "experiment_id",
        "method", "n_points", "budget", "attempt_generation", "ranking_labels"}
    if set(manifest) - allowed_manifest_fields:
        raise ContractError(f"DemoExecutionManifest has unsupported fields: {sorted(set(manifest) - allowed_manifest_fields)}")
    if evaluation_split not in {"train", "valid", "test", "unassigned"}:
        raise ContractError("evaluation_split is invalid")
    if not isinstance(include_valid, bool):
        raise ContractError("include_valid must be boolean")
    if not manifest["cohort_cases"]:
        raise ContractError("DemoExecutionManifest cohort_cases must not be empty")
    if (not isinstance(ranking_rules, tuple) or not ranking_rules
            or len(set(ranking_rules)) != len(ranking_rules)
            or any(rule not in {"highest_scan_energy", "internal_scan_peak"} for rule in ranking_rules)):
        raise ContractError("ranking_rules must be a non-empty tuple of unique supported rules")
    if not isinstance(manifest.get("method"), (dict, type(None))):
        raise ContractError("method must be a frozen method object")
    budget = manifest.get("budget")
    if budget is not None and (not isinstance(budget, dict)
            or set(budget) != {"max_attempts", "max_cpu_hours", "max_wall_seconds"}):
        raise ContractError("budget must define max_attempts, max_cpu_hours, and max_wall_seconds")
    if (not isinstance(manifest.get("n_points", 9), int)
            or isinstance(manifest.get("n_points", 9), bool)
            or manifest.get("n_points", 9) < 3):
        raise ContractError("n_points must be an integer of at least 3")
    label_ref = ranking_labels_path or manifest.get("ranking_labels")
    if ranking_labels_path is not None and manifest.get("ranking_labels") is not None:
        raise ContractError("specify ranking labels either in the manifest or as an override, not both")
    ranking_labels = _read_array_ref(source_root, label_ref, "ranking_labels") if label_ref else []
    cases = []
    cohort_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(manifest["cohort_cases"]):
        if isinstance(row, dict) and set(row) - {"case"}:
            raise ContractError(f"cohort_cases[{index}] contains unsupported fields")
        reference = row.get("case") if isinstance(row, dict) else row
        case = _read_ref(source_root, reference, f"cohort_cases[{index}]")
        try:
            dumps_document(case)
        except ContractError as exc:
            raise ContractError(f"cohort_cases[{index}] ReactionCase: {exc}") from exc
        if case.get("schema_name") != "ReactionCase":
            raise ContractError(f"cohort_cases[{index}] must be a ReactionCase")
        if case["case_id"] in cohort_by_id or case["reaction_id"] in {
            item["reaction_id"] for item in cases
        }:
            raise ContractError("Demo cohort requires unique case_id and reaction_id values")
        cases.append(case)
        cohort_by_id[case["case_id"]] = case
    runs = manifest["runs"]
    run_cases: set[str] = set()
    loaded_entries = []
    for index, row in enumerate(runs):
        if not isinstance(row, dict) or set(row) - {"case", "review_record", "candidate_id"}:
            raise ContractError(f"runs[{index}] must be an object")
        case = _read_ref(source_root, row.get("case"), f"runs[{index}].case")
        review = _read_ref(source_root, row.get("review_record"),
                           f"runs[{index}].review_record", required=False)
        if case["case_id"] not in cohort_by_id or case["content_sha256"] != cohort_by_id[case["case_id"]]["content_sha256"]:
            raise ContractError(f"runs[{index}] case does not match the frozen cohort snapshot")
        if case["case_id"] in run_cases:
            raise ContractError(f"multiple run entries for case_id {case['case_id']}")
        run_cases.add(case["case_id"])
        loaded_entries.append((index, row, case, review))
    missing_run_cases = set(cohort_by_id) - run_cases
    if missing_run_cases:
        raise ContractError(f"every frozen cohort case requires a run row; missing {sorted(missing_run_cases)}")

    manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    backend = ACPCLIBackend(acp_root=acp_root, python_executable=python_executable,
                            config_path=config_path)
    config_sha256 = hashlib.sha256(backend.config_path.read_bytes()).hexdigest() if backend.config_path else None
    input_fingerprint_payload = {
        "manifest_sha256":manifest_sha256,
        "cases":[{"case_id":case["case_id"], "content_sha256":case["content_sha256"]}
                 for case in cases],
        "reviews":[{"case_id":case["case_id"],
            "content_sha256":review.get("content_sha256") if review else None}
            for _, _, case, review in loaded_entries],
        "ranking_labels_sha256":hashlib.sha256(stable_json_dumps(ranking_labels).encode()).hexdigest(),
        "execution_environment":{"acp_root":str(backend.acp_root),
            "python_executable":backend.python_executable, "config_sha256":config_sha256,
            "acp_python_tree_sha256":_python_tree_sha256(backend.acp_root),
            "nproc":nproc, "memory":memory},
        "ranking_rules":list(ranking_rules), "evaluation_split":evaluation_split,
    }
    input_fingerprint_sha256 = hashlib.sha256(stable_json_dumps(input_fingerprint_payload).encode()).hexdigest()
    train_snapshot = None
    if include_valid:
        train_snapshot = _validate_train_index(train_index_path, manifest_sha256=manifest_sha256,
            input_fingerprint=input_fingerprint_sha256, cases=cases)
        if train_snapshot["bundle_labels"] != ranking_labels:
            raise ContractError("train run ranking labels differ from this frozen evaluation input")

    output = Path(output_root).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ContractError("output_root must be absent or empty; use a new root to preserve immutable run bundles")
    output.mkdir(parents=True, exist_ok=True)
    cohort_refs = []
    for index, case in enumerate(cases):
        relative = f"cohort/cases/{_portable_stem(index, case['case_id'])}.json"
        write_json(output / relative, case)
        cohort_refs.append(relative)

    bundle_runs: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    metrics_runs: list[dict[str, Any]] = []
    for row_index, source_row, case, review in loaded_entries:
        if include_valid and case["split"] == "train":
            case_id = case["case_id"]
            train_bundle_row = train_snapshot["bundle_rows"][case_id]
            _copy_run_tree(train_snapshot["root"], train_bundle_row["case"], output)
            bundle_runs.append(train_bundle_row)
            metrics_runs.append(train_snapshot["runs"][case_id])
            index_rows.append(train_snapshot["index_rows"][case_id])
            continue
        stem = _portable_stem(row_index, case["case_id"])
        run_dir = output / "runs" / stem
        run_dir.mkdir(parents=True, exist_ok=True)
        case_ref = f"runs/{stem}/ReactionCase.json"
        write_json(output / case_ref, case)
        entry_status = "blocked_review"
        plan = None
        execution = None
        path = None
        proposals: list[dict[str, Any]] = []
        plan_ref = execution_ref = path_ref = None
        proposal_refs: list[str] = []
        note = None
        if case.get("split") == "valid" and not include_valid:
            entry_status = "valid_held_out"
            note = "valid split held out; pass include_valid=True only for the frozen evaluation run"
        elif case.get("split") not in {"train", "valid"}:
            entry_status = "unsupported_split"
            note = f"unsupported split: {case.get('split')}"
        elif review is None or not _accepted_review(case, review):
            entry_status = "blocked_review"
            note = "accepted human ReviewRecord bound to the exact ready ReactionCase is required"
        else:
            try:
                try:
                    dumps_document(review)
                except ContractError as exc:
                    raise ContractError(f"runs[{row_index}].review_record: {exc}") from exc
                plan = build_scan_plan_with_rejection(case,
                    experiment_id=manifest.get("experiment_id", "demo24-v1"),
                    method=manifest.get("method"), n_points=manifest.get("n_points", 9),
                    budget=budget)
                plan_ref = f"runs/{stem}/ScanPlan.json"
                write_json(output / plan_ref, plan)
                if plan.get("status") != "ready":
                    entry_status = "plan_rejected"
                    note = "; ".join(plan.get("reject_reasons", []))
                else:
                    candidate_id = source_row.get("candidate_id")
                    preflight = scan_plan_to_acp_request(case, plan, candidate_id)
                    candidate_id = preflight["metadata"]["candidate_id"]
                    candidate = next(item for item in plan["candidates"]
                                     if item["candidate_id"] == candidate_id)
                    execution_id = "exec-" + hashlib.sha256(stable_json_dumps({
                        "input_fingerprint_sha256": input_fingerprint_sha256,
                        "case": case["content_sha256"], "candidate": candidate_id,
                        "include_valid": include_valid,
                        "attempt_generation": manifest.get("attempt_generation", 1)}).encode()).hexdigest()[:20]
                    attempt_base = "attempt-" + hashlib.sha256(stable_json_dumps({
                        "execution_id": execution_id,
                        "attempt_generation": manifest.get("attempt_generation", 1)}).encode()).hexdigest()[:20]
                    attempt_results = []
                    deadline = time.monotonic() + float(candidate["budget"]["max_wall_seconds"])
                    max_attempts = candidate["budget"]["max_attempts"]
                    for attempt_number in range(1, max_attempts + 1):
                        remaining_wall = deadline - time.monotonic()
                        if remaining_wall <= 0:
                            break
                        attempt_id = f"{attempt_base}-a{attempt_number:03d}"
                        result = backend.run_scan(execution_id=execution_id, attempt_id=attempt_id,
                            scan_request=preflight["scan_request"], output_root=run_dir,
                            timeout_seconds=remaining_wall, nproc=nproc, memory=memory)
                        attempt_results.append(result)
                        if result.status != "failed":
                            break
                    if not attempt_results:
                        raise ACPCLIError("frozen wall-time budget expired before an ACP attempt could start")
                    attempt_records = [cli_result_to_execution_record(case=case, plan=plan,
                        candidate_id=candidate_id, result=item) for item in attempt_results]
                    execution = _aggregate_cli_execution_records(case=case, plan=plan,
                        candidate_id=candidate_id, execution_id=execution_id,
                        attempt_results=attempt_results, attempt_records=attempt_records)
                    result = attempt_results[-1]
                    execution_ref = f"runs/{stem}/ExecutionRecord.json"
                    write_json(output / execution_ref, execution)
                    if result.status == "completed":
                        path = collect_cli_path_bundle(output_dir=result.attempt_dir,
                            case=case, plan=plan, execution_id=execution["object_id"],
                            candidate_id=candidate_id)
                        assessed = apply_scan_path_quality(plan=plan, execution=execution, path=path)
                        path = assessed["path_bundle"]
                        path_ref = f"runs/{stem}/PathBundle.json"
                        write_json(output / path_ref, path)
                        write_json(output / f"runs/{stem}/PathQuality.json", assessed["quality_assessment"])
                        for rule in ranking_rules:
                            proposal = rank_path_bundle(path, rule=rule, top_k=3)
                            proposals.append(proposal)
                            proposal_ref = f"runs/{stem}/SeedProposal-{rule}.json"
                            write_json(output / proposal_ref, proposal)
                            proposal_refs.append(proposal_ref)
                        viewer_ref = f"runs/{stem}/path_viewer.html"
                        render_path_viewer(path, output / viewer_ref, top_k=3)
                        entry_status = "path_collected"
                    else:
                        entry_status = "execution_failed"
                        note = result.error
            except (OSError, ValueError, ContractError, ACPCLIError) as exc:
                entry_status = "preflight_or_collection_failed"
                note = str(exc)
        if entry_status in {"blocked_review", "valid_held_out", "unsupported_split"}:
            review_ref = None
        else:
            review_ref = f"runs/{stem}/ReviewRecord.json"
            if review is not None:
                write_json(output / review_ref, review)
            else:
                review_ref = None
        metrics_runs.append({"case": case, "plan": plan, "execution": execution,
            "path": path, "validation": None, "proposals": proposals})
        bundle_runs.append({"case": case_ref, "plan": plan_ref,
            "execution": execution_ref, "path": path_ref, "validation": None,
            "proposals": proposal_refs, "run_status": entry_status,
            "review_status": review.get("status") if review is not None else None,
            "note": note})
        index_rows.append({"reaction_id": case["reaction_id"], "case_id": case["case_id"],
            "split": case["split"], "status": entry_status, "plan_ref": plan_ref,
            "execution_ref": execution_ref, "path_ref": path_ref,
            "review_status": review.get("status") if review is not None else None,
            "proposal_refs": proposal_refs, "review_record_ref": review_ref, "note": note})

    labels_output_ref = None
    if label_ref is not None:
        labels_output_ref = "RankingLabels.json"
        write_json(output / labels_output_ref, ranking_labels)
    bundle = {"schema_version": BUNDLE_SCHEMA, "cohort_cases": cohort_refs,
        "runs": bundle_runs, "ranking_labels": labels_output_ref}
    write_json(output / "DemoRunBundle.json", bundle)
    metrics = evaluate_demo_runs(runs=metrics_runs, cohort_cases=cases,
        ranking_labels=ranking_labels, evaluation_split=evaluation_split)
    write_json(output / "DemoMetrics.json", metrics)
    index_doc = {"schema_version": EXECUTION_SCHEMA, "manifest_sha256": manifest_sha256,
        "input_fingerprint_sha256":input_fingerprint_sha256,
        "cohort_case_count": len(cases),
        "selected_splits": ["train", "valid"] if include_valid else ["train"],
        "status_counts": {}, "runs": index_rows,
        "bundle_ref": "DemoRunBundle.json", "metrics_ref": "DemoMetrics.json",
        "ranking_labels_source_ref": label_ref,
        "train_index_sha256": (hashlib.sha256(train_snapshot["index_path"].read_bytes()).hexdigest()
            if train_snapshot else None)}
    for row in index_rows:
        index_doc["status_counts"][row["status"]] = index_doc["status_counts"].get(row["status"], 0) + 1
    write_json(output / "DemoExecutionIndex.json", index_doc)
    return index_doc


def _aggregate_cli_execution_records(*, case: dict[str, Any], plan: dict[str, Any],
                                     candidate_id: str, execution_id: str,
                                     attempt_results: list[Any],
                                     attempt_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep every immutable outer CLI attempt on one candidate execution."""
    if not attempt_results or len(attempt_results) != len(attempt_records):
        raise ContractError("CLI attempt results and attempt records must be non-empty and aligned")
    normalized_attempts = [record["attempts"][0] for record in attempt_records]
    final = attempt_results[-1]
    record = acp_execution_to_record(case=case, plan=plan, candidate_id=candidate_id,
        execution_id=execution_id, acp_task_id=None, task_status=final.status,
        attempts=normalized_attempts, work_ref="WORK/pes2ts",
        result_ref="RESULT/result_manifest.json")
    receipts = []
    for result in attempt_results:
        attempt_dir = Path(result.attempt_dir).resolve()
        try:
            attempt_ref = attempt_dir.relative_to(attempt_dir.parents[1]).as_posix()
        except (IndexError, ValueError) as exc:
            raise ContractError("ACP CLI attempt directory has an invalid layout") from exc
        receipts.append({"attempt_id": result.attempt_id, "status": result.status,
            "returncode": result.returncode, "wall_seconds": result.wall_seconds,
            "request_sha256": result.request_sha256,
            "result_manifest_sha256": result.manifest_sha256,
            "log_ref": f"{attempt_ref}/WORK/pes2ts/acp_cli.log",
            "attempt_ref": attempt_ref, "reused": result.reused,
            "timed_out": result.timed_out, "error": result.error})
    extensions = dict(record.get("extensions", {}))
    extensions["pes2ts.acp_cli_attempt_receipts.v1"] = {
        "schema": "pes2ts_acp_cli_attempt_receipts_v1",
        "wall_time_unit": "second", "cpu_time_available": False,
        "attempts": receipts,
    }
    return seal_document({**record, "extensions": extensions})


__all__ = ["EXECUTION_SCHEMA", "run_demo_execution_manifest"]
