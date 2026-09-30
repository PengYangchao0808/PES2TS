"""Validate a completed Demo24 workbook and write audited review/case snapshots."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.contracts import ContractError, dumps_document, loads_document
from pes2ts_core.g1.review_import import import_review_rows, read_review_workbook

DEFAULT_CASES = REPO_ROOT / "data/interim/g1_v2/reaction_cases_review_v1"
DEFAULT_MANIFEST = REPO_ROOT / "data/manifests/demo24_reaction_case_manifest_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/interim/g1_v2/reaction_cases_reviewed_v1"
DEFAULT_RECORDS_MANIFEST = REPO_ROOT / "data/manifests/demo24_review_records_v1.json"


def load_cases(case_root: Path, manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = {}
    for item in manifest.get("records", []):
        path = case_root / item["relative_path"]
        case = loads_document(path.read_text(encoding="utf-8"))
        if (case.get("reaction_id") != item.get("reaction_id")
                or case.get("case_id") != item.get("case_id")
                or case.get("content_sha256") != item.get("content_sha256")
                or case.get("split") != item.get("split")):
            raise ContractError(f"ReactionCase/manifest mismatch: {path}")
        if case["status"] != "needs_review":
            raise ContractError(f"expected non-executable needs_review ReactionCase: {path}")
        cases[case["reaction_id"]] = case
    if not cases or len(cases) != manifest.get("n_cases"):
        raise ContractError("ReactionCase manifest count does not match its records")
    return cases


def import_workbook(workbook_path: Path, *, case_root: Path = DEFAULT_CASES,
                    manifest_path: Path = DEFAULT_MANIFEST) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
    sheets, workbook_sha256 = read_review_workbook(workbook_path)
    cases = load_cases(case_root, manifest_path)
    processed, records = import_review_rows(
        cases=cases,
        reviewer_one=sheets["复核者1"],
        reviewer_two=sheets["复核者2"],
        adjudications=sheets["汇总裁定"],
        workbook_sha256=workbook_sha256,
    )
    return processed, records, workbook_sha256


def write_snapshot(processed: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]],
                   workbook_sha256: str, output_root: Path, records_manifest_path: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty reviewed case directory: {output_root}")
    if records_manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing review manifest: {records_manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    records_root = output_root / "review_records"
    records_root.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    manifest_records = []
    for reaction_id, case in processed.items():
        split_dir = output_root / case["split"]
        split_dir.mkdir(parents=True, exist_ok=True)
        case_relative = Path(case["split"]) / f"{reaction_id}.json"
        review_relative = Path("review_records") / f"{reaction_id}.json"
        case_text = dumps_document(case) + "\n"
        review_text = dumps_document(records[reaction_id]) + "\n"
        (output_root / case_relative).write_text(case_text, encoding="utf-8")
        (output_root / review_relative).write_text(review_text, encoding="utf-8")
        counts[case["status"]] += 1
        manifest_records.append({
            "reaction_id": reaction_id, "case_id": case["case_id"], "split": case["split"],
            "case_status": case["status"], "case_sha256": case["content_sha256"],
            "case_relative_path": case_relative.as_posix(),
            "review_object_id": records[reaction_id]["object_id"],
            "review_sha256": records[reaction_id]["content_sha256"],
            "review_relative_path": review_relative.as_posix(),
        })
    summary = {
        "schema_version": "demo24_reviewed_case_manifest_v1",
        "status": "completed_human_review_import",
        "workbook_sha256": workbook_sha256,
        "n_cases": len(manifest_records),
        "case_status_counts": dict(sorted(counts.items())),
        "records": manifest_records,
    }
    records_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records_manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="completed two-reviewer workbook")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--records-manifest", type=Path, default=DEFAULT_RECORDS_MANIFEST)
    args = parser.parse_args()
    try:
        processed, records, workbook_sha256 = import_workbook(args.workbook,
            case_root=args.cases, manifest_path=args.manifest)
        result = write_snapshot(processed, records, workbook_sha256, args.output, args.records_manifest)
    except (ContractError, FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: result[key] for key in ("status", "n_cases", "case_status_counts", "workbook_sha256")},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
