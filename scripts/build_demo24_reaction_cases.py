"""Build endpoint-only, non-executable ReactionCases for the Demo24 review queue."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.g1.reaction_case import reaction_case_from_g1_export, resolve_endpoint_multiplicity
from scripts.audit_demo24_endpoints import EXPORT_ROOT, audit_demo24


DEFAULT_OUTPUT = REPO_ROOT / "data/interim/g1_v2/reaction_cases_review_v1"
MANIFEST_PATH = REPO_ROOT / "data/manifests/demo24_reaction_case_manifest_v1.json"
REVIEW_REASON = "endpoint reaction and atom mapping require manual chemistry review before acceptance"


def build_review_cases(output_root: Path = DEFAULT_OUTPUT, manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Create 24 standard ReactionCases with conservative source-based spin resolution."""
    audit = audit_demo24()
    if audit["audit_status"] != "consistent_pending_human_review":
        raise ValueError("endpoint data audit must pass before ReactionCase generation")
    candidate_path = REPO_ROOT / "data/manifests/pes2ts_demo24_candidates_v1.csv"
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as stream:
        candidate_rows = {row["reaction_id"]: row for row in csv.DictReader(stream)}

    records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for audit_row in audit["records"]:
        reaction_id = audit_row["reaction_id"]
        row = candidate_rows.get(reaction_id)
        if row is None or row["split"] != audit_row["split"]:
            raise ValueError(f"candidate list does not match audited identity for {reaction_id}")
        numeric_id = int(reaction_id.rsplit("_", 1)[1])
        export_path = EXPORT_ROOT / f"{numeric_id // 1000:05d}" / f"{reaction_id}.json"
        export_bytes = export_path.read_bytes()
        export_sha256 = hashlib.sha256(export_bytes).hexdigest()
        export = json.loads(export_bytes)
        spin = {side: resolve_endpoint_multiplicity(export, side)
                for side in ("reactant", "product")}
        case = reaction_case_from_g1_export(
            export,
            split=row["split"],
            reactant_multiplicity=spin["reactant"]["multiplicity"],
            product_multiplicity=spin["product"]["multiplicity"],
            review_reasons=[REVIEW_REASON],
            spin_provenance=spin,
            source_export_sha256=export_sha256,
        )
        if case["status"] != "needs_review":
            raise ValueError(f"unreviewed case unexpectedly executable: {reaction_id}")
        split_dir = output_root / row["split"]
        split_dir.mkdir(parents=True, exist_ok=True)
        relative_path = Path(row["split"]) / f"{reaction_id}.json"
        case_path = output_root / relative_path
        case_path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        status_counts[case["status"]] += 1
        records.append({
            "reaction_id": reaction_id,
            "case_id": case["case_id"],
            "split": row["split"],
            "source_export_sha256": export_sha256,
            "status": case["status"],
            "multiplicities": {"reactant": case["reactant"]["multiplicity"],
                               "product": case["product"]["multiplicity"]},
            "spin_resolution": {side: spin[side]["status"] for side in spin},
            "review_reasons": case["review_reasons"],
            "relative_path": relative_path.as_posix(),
            "content_sha256": case["content_sha256"],
        })

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(
        "# Demo24 ReactionCase review packet\n\n"
        "These are standard endpoint-only ReactionCase documents generated from sanitized G1 exports. "
        "All 24 cases remain `needs_review` pending independent chemistry review. Endpoint multiplicities "
        "are resolved only when the source component multiplicities uniquely determine total spin; "
        "ambiguous spin coupling remains null. The evidence is recorded in `source.spin_provenance`.\n\n"
        "No TS/IRC geometries, energies, labels, or barriers were read. See the sibling manifest in "
        "`data/manifests/demo24_reaction_case_manifest_v1.json` for split counts, identities, paths, and content hashes.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "demo24_reaction_case_manifest_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "awaiting_manual_chemistry_review",
        "information_boundary": "Only sanitized reactant/product endpoint exports and their per-component charge/multiplicity fields were used. No TS/IRC geometries, energies, labels, or barriers were read.",
        "n_cases": len(records),
        "split_counts": dict(sorted(Counter(record["split"] for record in records).items())),
        "case_status_counts": dict(sorted(status_counts.items())),
        "sources": {
            "candidate_csv_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            "endpoint_audit_schema": audit["schema_version"],
            "sanitized_exports": "data/interim/g1_v2/export_contracts_v1",
        },
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    result = build_review_cases(args.output, args.manifest)
    print(json.dumps({key: result[key] for key in ("status", "n_cases", "split_counts", "case_status_counts")},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
