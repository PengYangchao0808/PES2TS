"""Freeze reviewed Demo24 snapshots into an ACP CLI execution input bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.contracts import ContractError, dumps_document
from pes2ts_core.utils.jsonio import write_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ContractError(f"{label} must be a JSON object")
    return result


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ContractError(f"{label} must be a relative path")
    portable = relative.replace("\\", "/")
    if portable.startswith("/") or ":" in portable or any(part in {"", ".", ".."} for part in portable.split("/")):
        raise ContractError(f"{label} contains an unsafe path")
    path = (root / Path(*PurePosixPath(portable).parts)).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ContractError(f"{label} is missing or escapes its source root")
    return path


def _review_is_bound(case: dict[str, Any], review: dict[str, Any]) -> bool:
    source = case.get("source", {})
    extensions = review.get("extensions", {})
    review_output = extensions.get("pes2ts.review_output.v1", {}) if isinstance(extensions, dict) else {}
    return (case.get("status") == "ready" and review.get("status") == "accepted"
        and review.get("schema_name") == "ReviewRecord"
        and (review.get("reaction_id"), review.get("case_id"), review.get("dataset_version"), review.get("split"))
           == (case.get("reaction_id"), case.get("case_id"), case.get("dataset_version"), case.get("split"))
        and isinstance(source, dict) and source.get("review_record_id") == review.get("object_id")
        and source.get("reviewed_from_case_sha256") == review.get("case_sha256")
        and isinstance(review_output, dict) and review_output.get("reviewed_case_sha256") == case.get("content_sha256"))


def build_demo24_execution_bundle(*, reviewed_cases_root: str | Path,
        reviewed_manifest_path: str | Path, source_manifest_path: str | Path,
        output_root: str | Path, experiment_id: str = "demo24-v1",
        method: dict[str, Any] | None = None, n_points: int = 9,
        budget: dict[str, Any] | None = None,
        ranking_labels_path: str | Path | None = None) -> dict[str, Any]:
    """Copy a reviewed 16/8 cohort into a path-safe immutable run-input bundle."""
    cases_root = Path(reviewed_cases_root).expanduser().resolve(strict=True)
    reviewed_manifest_file = Path(reviewed_manifest_path).expanduser().resolve(strict=True)
    source_manifest_file = Path(source_manifest_path).expanduser().resolve(strict=True)
    source_manifest = _load_json(source_manifest_file, "source ReactionCase manifest")
    reviewed_manifest = _load_json(reviewed_manifest_file, "reviewed ReactionCase manifest")
    if (source_manifest.get("schema_version") != "demo24_reaction_case_manifest_v1"
            or reviewed_manifest.get("schema_version") != "demo24_reviewed_case_manifest_v1"
            or reviewed_manifest.get("status") != "completed_human_review_import"):
        raise ContractError("expected the canonical Demo24 and completed reviewed-case manifests")
    if source_manifest.get("n_cases") != 24 or reviewed_manifest.get("n_cases") != 24:
        raise ContractError("Demo24 execution bundle requires exactly 24 frozen ReactionCases")
    source_rows = source_manifest.get("records")
    reviewed_rows = reviewed_manifest.get("records")
    if not isinstance(source_rows, list) or not isinstance(reviewed_rows, list):
        raise ContractError("both Demo24 manifests must contain records arrays")
    source_by_id = {row.get("reaction_id"): row for row in source_rows if isinstance(row, dict)}
    if len(source_by_id) != 24 or len(reviewed_rows) != 24:
        raise ContractError("Demo24 manifests must each contain 24 unique reaction records")
    split_counts = {"train": 0, "valid": 0}
    seen_reactions: set[str] = set()
    output = Path(output_root).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ContractError("output_root must be absent or empty; frozen input bundles are immutable")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ContractError("experiment_id must be a non-empty frozen identifier")
    if not isinstance(n_points, int) or isinstance(n_points, bool) or not 3 <= n_points <= 101:
        raise ContractError("n_points must be an integer in 3..101")
    if method is not None and not isinstance(method, dict):
        raise ContractError("method must be a frozen object")
    if budget is not None and (not isinstance(budget, dict)
            or set(budget) != {"max_attempts", "max_cpu_hours", "max_wall_seconds"}):
        raise ContractError("budget must define max_attempts, max_cpu_hours, and max_wall_seconds")

    cohort_refs: list[str] = []
    run_rows: list[dict[str, str]] = []
    payloads: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    for index, row in enumerate(reviewed_rows):
        if not isinstance(row, dict):
            raise ContractError(f"reviewed records[{index}] must be an object")
        reaction_id = row.get("reaction_id")
        original = source_by_id.get(reaction_id)
        if reaction_id in seen_reactions:
            raise ContractError(f"duplicate reviewed reaction row: {reaction_id}")
        seen_reactions.add(reaction_id)
        if original is None or (row.get("case_id"), row.get("split")) != (
                original.get("case_id"), original.get("split")):
            raise ContractError(f"reviewed identity/split does not match the frozen cohort: {reaction_id}")
        split = row.get("split")
        if split not in split_counts:
            raise ContractError(f"unsupported Demo24 split: {split}")
        split_counts[split] += 1
        case_path = _safe_child(cases_root, row.get("case_relative_path"), f"{reaction_id} case path")
        review_path = _safe_child(cases_root, row.get("review_relative_path"), f"{reaction_id} ReviewRecord path")
        case = _load_json(case_path, f"{reaction_id} ReactionCase")
        review = _load_json(review_path, f"{reaction_id} ReviewRecord")
        dumps_document(case); dumps_document(review)
        if (case.get("reaction_id"), case.get("case_id"), case.get("split"), case.get("content_sha256")) != (
                reaction_id, row.get("case_id"), split, row.get("case_sha256")):
            raise ContractError(f"reviewed ReactionCase does not match its manifest row: {reaction_id}")
        case_source = case.get("source", {})
        if (not isinstance(case_source, dict)
                or case_source.get("reviewed_from_case_sha256") != original.get("content_sha256")):
            raise ContractError(f"reviewed ReactionCase does not trace to the canonical endpoint snapshot: {reaction_id}")
        if (review.get("object_id") != row.get("review_object_id")
                or review.get("content_sha256") != row.get("review_sha256")
                or (review.get("reaction_id"), review.get("case_id"), review.get("split"),
                    review.get("dataset_version")) != (case.get("reaction_id"), case.get("case_id"),
                    case.get("split"), case.get("dataset_version"))
                or case_source.get("review_record_id") != review.get("object_id")
                or case_source.get("reviewed_from_case_sha256") != review.get("case_sha256")):
            raise ContractError(f"ReviewRecord does not match its manifest row: {reaction_id}")
        review_extensions = review.get("extensions", {})
        review_output = review_extensions.get("pes2ts.review_output.v1", {}) if isinstance(review_extensions, dict) else {}
        if (not isinstance(review_output, dict)
                or review_output.get("reviewed_case_sha256") != case.get("content_sha256")):
            raise ContractError(f"ReviewRecord output digest does not match the reviewed case: {reaction_id}")
        if review.get("status") == "accepted" and not _review_is_bound(case, review):
            raise ContractError(f"accepted ReviewRecord is not bound to the ready case: {reaction_id}")
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", reaction_id)
        case_ref = f"cohort/cases/{index:03d}_{stem}.json"
        review_ref = f"cohort/review_records/{index:03d}_{stem}.json"
        cohort_refs.append(case_ref)
        run_rows.append({"case": case_ref, "review_record": review_ref})
        payloads.append((case_ref, review_ref, case, review))

    if seen_reactions != set(source_by_id):
        raise ContractError("reviewed cases do not exactly cover the canonical Demo24 reactions")
    if split_counts != {"train": 16, "valid": 8}:
        raise ContractError(f"Demo24 split counts must be train=16, valid=8; received {split_counts}")
    labels_ref = None
    labels_digest = None
    if ranking_labels_path is not None:
        label_file = Path(ranking_labels_path).expanduser().resolve(strict=True)
        try:
            labels = json.loads(label_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ContractError(f"ranking labels are not valid JSON: {exc}") from exc
        if not isinstance(labels, list) or any(not isinstance(item, dict) for item in labels):
            raise ContractError("ranking labels must be a JSON array of objects")
        labels_ref = "evaluation/ranking_labels.json"
        write_json(output / labels_ref, labels)
        labels_digest = _sha256(label_file)
    run_manifest: dict[str, Any] = {"schema_version":"pes2ts_demo_execution_manifest_v1",
        "experiment_id":experiment_id, "n_points":n_points,
        "cohort_cases":cohort_refs, "runs":run_rows}
    if method is not None:
        run_manifest["method"] = method
    if budget is not None:
        run_manifest["budget"] = budget
    if labels_ref:
        run_manifest["ranking_labels"] = labels_ref
    output.mkdir(parents=True, exist_ok=True)
    for case_ref, review_ref, case, review in payloads:
        write_json(output / case_ref, case)
        write_json(output / review_ref, review)
    if labels_ref:
        write_json(output / labels_ref, labels)
    write_json(output / "DemoExecutionManifest.json", run_manifest)
    receipt = {"schema_version":"pes2ts_demo24_input_freeze_v1", "reaction_case_count":24,
        "split_counts":split_counts, "source_manifest_sha256":_sha256(source_manifest_file),
        "reviewed_manifest_sha256":_sha256(reviewed_manifest_file),
        "workbook_sha256":reviewed_manifest.get("workbook_sha256"),
        "ranking_labels_sha256":labels_digest, "execution_manifest_sha256":_sha256(output/"DemoExecutionManifest.json")}
    write_json(output / "Demo24InputFreezeReceipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-cases", type=Path, required=True,
                        help="root written by import_demo24_review_workbook.py")
    parser.add_argument("--reviewed-manifest", type=Path, required=True,
                        help="demo24_review_records_v1.json")
    parser.add_argument("--source-manifest", type=Path,
                        default=REPO_ROOT / "data/manifests/demo24_reaction_case_manifest_v1.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="new/empty immutable execution input bundle directory")
    parser.add_argument("--experiment-id", default="demo24-v1")
    parser.add_argument("--method-json", type=Path, help="frozen MethodSpec JSON object")
    parser.add_argument("--budget-json", type=Path, help="frozen retry/compute budget JSON object")
    parser.add_argument("--n-points", type=int, default=9)
    parser.add_argument("--ranking-labels", type=Path, help="independently sourced ranking labels JSON")
    args = parser.parse_args()
    try:
        method = _load_json(args.method_json, "MethodSpec") if args.method_json else None
        budget = _load_json(args.budget_json, "ScanPlan budget") if args.budget_json else None
        receipt = build_demo24_execution_bundle(reviewed_cases_root=args.reviewed_cases,
            reviewed_manifest_path=args.reviewed_manifest, source_manifest_path=args.source_manifest,
            output_root=args.output, experiment_id=args.experiment_id, method=method,
            budget=budget, n_points=args.n_points, ranking_labels_path=args.ranking_labels)
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
