"""Audit whether sanitized source components determine each endpoint's total spin."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.contracts import ContractError
from pes2ts_core.g1.reaction_case import resolve_endpoint_multiplicity
from pes2ts_core.utils.jsonio import write_json
from scripts.audit_demo24_endpoints import EXPORT_ROOT, audit_demo24

DEFAULT_OUTPUT = REPO_ROOT / "data/manifests/demo24_spin_source_audit_v1.json"


def build_spin_audit(output_path: Path | None = None) -> dict[str, Any]:
    audit = audit_demo24()
    if audit["audit_status"] != "consistent_pending_human_review":
        raise ContractError("endpoint audit must pass before spin-source audit")
    records = []
    counts = {side: {"resolved": 0, "unresolved": 0} for side in ("reactant", "product")}
    for audited in audit["records"]:
        reaction_id = audited["reaction_id"]
        numeric_id = int(reaction_id.rsplit("_", 1)[1])
        export_path = EXPORT_ROOT / f"{numeric_id // 1000:05d}" / f"{reaction_id}.json"
        raw = export_path.read_bytes()
        export = json.loads(raw)
        spin = {side: resolve_endpoint_multiplicity(export, side)
                for side in ("reactant", "product")}
        for side in counts:
            counts[side][spin[side]["status"]] += 1
        records.append({
            "reaction_id": reaction_id,
            "split": audited["split"],
            "export_sha256": hashlib.sha256(raw).hexdigest(),
            "endpoint_spin": spin,
        })
    report = {
        "schema_version": "demo24_spin_source_audit_v1",
        "status": "source_spin_audited_manual_chemistry_review_pending",
        "n_reactions": len(records),
        "counts": counts,
        "rule": "Resolve only when at most one component is non-singlet; never use multiplicity_max.",
        "information_boundary": "Reads sanitized R/P endpoint exports and component charge/multiplicity only; does not read TS/IRC geometry, energies, labels, or barriers.",
        "records": records,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_spin_audit(args.output)
    print(json.dumps({"n_reactions": report["n_reactions"], "counts": report["counts"],
                      "output": str(args.output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
