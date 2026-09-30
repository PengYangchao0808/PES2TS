"""G1 v2 gate: scan-ready release decision and the whitelisted G2 export.

The v2 gate replaces the single ``eligible`` number with a three-way
decision per reaction -- ``scan_ready`` / ``needs_review`` / ``excluded`` --
with typed reasons in three separated dimensions.  Records whose IRC
evidence contradicts the mapped bond events, whose symmetry collapse can
change the recorded chemistry, or whose candidate search was truncated land
in the adjudication queue (``g2_needs_review.json``) instead of silently
passing as verified.

For every scan-ready reaction the gate writes a **whitelisted** G2 export
document: the map-sorted reactant/product atomic numbers and coordinates
(assembled through the P1 ``map_to_atoms`` bijection from the inventory
endpoint components), charges/spins, the exclusive edits, the aromatic
regions, the reaction center, and the v2 classification -- with
``mapping_provenance="truth_assisted_p1"`` and **no** TS/IRC-derived fields
(``ts_irc_index`` above all; the export verifier enforces the blacklist).
G2 may choose its own scan method; it may not redefine which two atoms
bond, break, or transfer hydrogen.
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g1.build import shard_name
from pes2ts_core.g1.p1_truth import p1_document_path, p1_settings
from pes2ts_core.g1.truth_schema import G2_ELIGIBLE_STATUSES
from pes2ts_core.g1.v2_build import v2_document_path, v2_settings
from pes2ts_core.g1.v2_classify import v2_class_document_path
from pes2ts_core.g1.v2_schema import (
    AUDIT_EXCLUDED,
    AUDIT_ISSUES,
    DIMENSIONS,
    EXPORT_DIRNAME,
    G2_NEEDS_REVIEW_FILENAME,
    G2_SCAN_READY_FILENAME,
    ISSUE_DIMENSIONS,
    MAPPING_PROVENANCE,
    V2_DIRNAME,
    V2_CLASS_SUMMARY_FILENAME,
    V2_EXPORT_SCHEMA_VERSION,
    V2_GATE_FILENAME,
    V2_GATE_SCHEMA_VERSION,
    V2_SUMMARY_FILENAME,
)
from pes2ts_core.g1.v2_verify import verify_v2
from pes2ts_core.utils.hashing import JSONValue
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import read_parquet

logger = logging.getLogger(__name__)

#: G0 split assignment artifact (report-only cross-tab).
SPLIT_ASSIGNMENT_FILENAME: Final[str] = "split_assignment.parquet"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def v2_export_dir(interim_dir: str | Path) -> Path:
    return Path(interim_dir) / V2_DIRNAME / EXPORT_DIRNAME


def v2_export_path(interim_dir: str | Path, reaction_id: str, shard_size: int) -> Path:
    return v2_export_dir(interim_dir) / shard_name(reaction_id, shard_size) / f"{reaction_id}.json"


def _side_rows(
    map_to_atoms: list[Mapping[str, Any]], side: str
) -> tuple[list[int], list[list[float]]]:
    """Return ``(atomic_numbers, coordinates)`` in map-sorted order."""
    numbers: list[int] = []
    coordinates: list[list[float]] = []
    for entry in sorted(map_to_atoms, key=lambda item: int(item["map"])):
        numbers.append(int(entry[f"_{side}_atomic_number"]))
        coordinates.append([float(v) for v in entry[f"_{side}_coordinate"]])
    return numbers, coordinates


def build_export_document(
    v2_document: Mapping[str, Any],
    p1_document: Mapping[str, Any],
    inventory_row: Mapping[str, Any],
    class_document: Mapping[str, Any] | None,
) -> dict[str, JSONValue]:
    """Return the whitelisted G2 export document of one scan-ready reaction."""
    reaction_id = str(v2_document["reaction_id"])
    map_to_atoms = list(p1_document["mapping"]["map_to_atoms"])
    side_index = {
        str(component["tag"]): component
        for component in inventory_row["components"]
    }
    enriched: list[dict[str, JSONValue]] = []
    for entry in sorted(map_to_atoms, key=lambda item: int(item["map"])):
        r_component = side_index[str(entry["r_component"])]
        p_component = side_index[str(entry["p_component"])]
        enriched.append({
            "map": int(entry["map"]),
            "element": str(entry["element"]),
            "r_component": str(entry["r_component"]),
            "r_local_index": int(entry["r_local_index"]),
            "p_component": str(entry["p_component"]),
            "p_local_index": int(entry["p_local_index"]),
            "_r_atomic_number": int(r_component["atomic_numbers"][int(entry["r_local_index"])]),
            "_r_coordinate": [
                float(v) for v in r_component["coordinates"][int(entry["r_local_index"])]
            ],
            "_p_atomic_number": int(p_component["atomic_numbers"][int(entry["p_local_index"])]),
            "_p_coordinate": [
                float(v) for v in p_component["coordinates"][int(entry["p_local_index"])]
            ],
        })
    r_numbers, r_coordinates = _side_rows(enriched, "r")
    p_numbers, p_coordinates = _side_rows(enriched, "p")
    components_block: dict[str, JSONValue] = {}
    for component in inventory_row["components"]:
        components_block[str(component["tag"])] = {
            "atomic_numbers": [int(z) for z in component["atomic_numbers"]],
            "charge": int(component["charge"]),
            "multiplicity": int(component["multiplicity"]),
        }
    levels: Mapping[str, Any] = (class_document or {}).get("levels") or {}
    document: dict[str, JSONValue] = {
        "schema_version": V2_EXPORT_SCHEMA_VERSION,
        "reaction_id": reaction_id,
        "dataset_version": str(inventory_row.get("dataset_version", "")),
        "mapping_provenance": MAPPING_PROVENANCE,
        "atom_order": "map ascending",
        "maps": [int(entry["map"]) for entry in enriched],
        "elements": [str(entry["element"]) for entry in enriched],
        "r_atomic_numbers": r_numbers,
        "r_coordinates": r_coordinates,
        "p_atomic_numbers": p_numbers,
        "p_coordinates": p_coordinates,
        "atom_rows": [
            {key: value for key, value in entry.items() if not key.startswith("_")}
            for entry in enriched
        ],
        "components": components_block,
        "charge_total_reactants": int(inventory_row.get("charge_total_reactants", 0)),
        "charge_total_products": int(inventory_row.get("charge_total_products", 0)),
        "multiplicity_max": int(inventory_row.get("multiplicity_max", 1)),
        "edits": v2_document.get("edits") or [],
        "hydrogen_partner_changes": v2_document.get("hydrogen_partner_changes") or [],
        "aromatic_regions": v2_document.get("aromatic_regions") or {},
        "reaction_center": v2_document.get("reaction_center") or {},
        "classification": {
            "l0_cluster_id": levels.get("l0_edit_family", {}).get("cluster_id"),
            "l0u_cluster_id": levels.get("l0u_undirected_family", {}).get("cluster_id"),
            "l1_cluster_id": levels.get("l1_center_template", {}).get("cluster_id"),
            "family_labels": list((class_document or {}).get("family_labels") or []),
        },
        # Production inputs must not carry IRC-derived matching/orientation
        # fields. Keep only endpoint-only mapping provenance and structural
        # audit status; evaluation evidence stays behind the truth boundary.
        "status_summary": {
            "p1_status": str(v2_document.get("p1_status", "")),
            "audit_status": str(v2_document.get("audit_status", "")),
        },
        "generated_at": _now(),
    }
    return document


@dataclass(frozen=True, slots=True)
class V2GateResult:
    """Outcome of one v2 gate run."""

    n_denominator: int
    n_scan_ready: int
    n_needs_review: int
    n_excluded: int
    manifest_path: Path
    scan_ready_path: Path
    needs_review_path: Path
    refusals: tuple[str, ...]


def run_v2_gate(config: Mapping[str, Any], *, assume_verified: bool = False) -> V2GateResult:
    """Verify both v2 trees, then write the gate manifests and the export."""
    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
    settings = v2_settings(config)
    shard_size = int(settings["shard_size"])
    p1_shard_size = int(p1_settings(config)["shard_size"])
    refusals: list[str] = []
    if not assume_verified:
        verification = verify_v2(config)
        refusals.extend(verification.problems)
    v2_rows = {
        str(row["reaction_id"]): row
        for row in read_parquet(interim_dir / V2_SUMMARY_FILENAME).to_pylist()
    }
    class_rows = {
        str(row["reaction_id"]): row
        for row in read_parquet(interim_dir / V2_CLASS_SUMMARY_FILENAME).to_pylist()
    }
    inventory_rows = {
        str(row["reaction_id"]): row
        for row in read_parquet(interim_dir / INVENTORY_PARQUET_FILENAME).to_pylist()
    }
    split_of: dict[str, str] = {}
    split_path = interim_dir / SPLIT_ASSIGNMENT_FILENAME
    if split_path.is_file():
        split_of = {
            str(row["reaction_id"]): str(row["split"])
            for row in read_parquet(split_path).to_pylist()
        }
    scan_ready: list[str] = []
    needs_review: list[dict[str, JSONValue]] = []
    excluded: list[str] = []
    by_reason: Counter[str] = Counter()
    by_dimension: Counter[str] = Counter()
    split_histogram: Counter[str] = Counter()
    for reaction_id in sorted(v2_rows):
        row = v2_rows[reaction_id]
        p1_status = str(row["p1_status"])
        audit_status = str(row["audit_status"])
        if p1_status not in G2_ELIGIBLE_STATUSES or audit_status == AUDIT_EXCLUDED:
            excluded.append(reaction_id)
            by_reason[f"excluded:p1:{p1_status}"] += 1
            continue
        issues = [part for part in str(row["issues"]).split("|") if part]
        if audit_status == AUDIT_ISSUES and issues:
            needs_review.append({
                "reaction_id": reaction_id,
                "reasons": issues,
                "dimensions": sorted({ISSUE_DIMENSIONS.get(code, "?") for code in issues}),
            })
            for code in issues:
                by_reason[f"review:{code}"] += 1
                dimension = ISSUE_DIMENSIONS.get(code)
                if dimension:
                    by_dimension[dimension] += 1
            continue
        scan_ready.append(reaction_id)
        split_histogram[split_of.get(reaction_id, "unassigned")] += 1
    export_root = v2_export_dir(interim_dir)
    if export_root.is_dir():
        shutil.rmtree(export_root)
    for reaction_id in scan_ready:
        v2_document = read_json(v2_document_path(interim_dir, reaction_id, shard_size))
        p1_document = read_json(p1_document_path(interim_dir, reaction_id, p1_shard_size))
        class_path = v2_class_document_path(interim_dir, reaction_id, shard_size)
        class_document = read_json(class_path) if class_path.is_file() else None
        export = build_export_document(
            v2_document, p1_document, inventory_rows[reaction_id], class_document,
        )
        write_json(v2_export_path(interim_dir, reaction_id, shard_size), export)
    manifest: dict[str, JSONValue] = {
        "schema_version": V2_GATE_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "n_denominator": len(v2_rows),
        "n_scan_ready": len(scan_ready),
        "n_needs_review": len(needs_review),
        "n_excluded": len(excluded),
        "scan_ready_fraction": round(len(scan_ready) / len(v2_rows), 6) if v2_rows else 0.0,
        "by_reason": dict(sorted(by_reason.items())),
        "by_dimension": dict(sorted(by_dimension.items())),
        "dimensions": list(DIMENSIONS),
        "scan_ready_split_histogram": dict(sorted(split_histogram.items())),
        "refusals": refusals,
        "gate_pass": not refusals,
        "generated_at": _now(),
    }
    manifest_path = manifests_dir / V2_GATE_FILENAME
    write_json(manifest_path, manifest)
    scan_ready_path = interim_dir / G2_SCAN_READY_FILENAME
    write_json(scan_ready_path, {
        "schema_version": V2_GATE_SCHEMA_VERSION,
        "reaction_ids": scan_ready,
        "n_eligible": len(scan_ready),
        "export_dir": str(v2_export_dir(interim_dir)),
        "mapping_provenance": MAPPING_PROVENANCE,
        "note": (
            "scan-ready reactions for endpoint-only G2; export documents are "
            "whitelisted and carry no TS/IRC-derived fields"
        ),
        "generated_at": _now(),
    })
    needs_review_path = interim_dir / G2_NEEDS_REVIEW_FILENAME
    write_json(needs_review_path, {
        "schema_version": V2_GATE_SCHEMA_VERSION,
        "records": needs_review,
        "n_records": len(needs_review),
        "dimensions": list(DIMENSIONS),
        "note": "adjudication queue; conflicts are never silently released",
        "generated_at": _now(),
    })
    logger.info(
        "G1 v2 gate: %d scan-ready, %d needs_review, %d excluded (of %d) -> %s",
        len(scan_ready), len(needs_review), len(excluded), len(v2_rows), manifest_path,
    )
    return V2GateResult(
        n_denominator=len(v2_rows),
        n_scan_ready=len(scan_ready),
        n_needs_review=len(needs_review),
        n_excluded=len(excluded),
        manifest_path=manifest_path,
        scan_ready_path=scan_ready_path,
        needs_review_path=needs_review_path,
        refusals=tuple(refusals),
    )


__all__ = [
    "V2GateResult",
    "build_export_document",
    "run_v2_gate",
    "v2_export_dir",
    "v2_export_path",
]
