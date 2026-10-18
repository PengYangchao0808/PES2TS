"""Source/metadata inventory join with typed qualification (B1).

Joins the frozen G0 inventory, the G1 P1 per-reaction summary, the G1 v2
scan-ready gate, the frozen parent split, and the v2 family classification —
all transition-free metadata, no quarantined truth is read here.  Every
reaction receives exactly one typed qualification code; nothing is dropped
silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import pyarrow.parquet as pq

from pes2ts_core.utils.hashing import JSONValue, sha256_file, stable_json_dumps
from pes2ts_core.utils.jsonio import read_json

QUALIFIED: Final[str] = "qualified"
P1_NOT_RESOLVED: Final[str] = "p1_not_resolved"
NOT_SCAN_READY: Final[str] = "not_scan_ready"
LAYOUT_UNSUPPORTED: Final[str] = "layout_unsupported"
ORIENTATION_UNRESOLVED: Final[str] = "orientation_unresolved"
NOT_IN_PARENT_TRAIN: Final[str] = "not_in_parent_train"

#: Layouts the first-cycle teacher builder can label.
SUPPORTED_LAYOUTS: Final[frozenset[str]] = frozenset(
    {"ts_first_dual_branch", "single_branch"}
)
#: P1 statuses whose atom mapping is usable for teacher labels.
RESOLVED_P1_STATUSES: Final[frozenset[str]] = frozenset(
    {"resolved_unique", "resolved_symmetry_collapsed"}
)
SUPPORTED_ORIENTATIONS: Final[frozenset[str]] = frozenset({"P_first", "R_first"})

INVENTORY_REPORT_SCHEMA: Final[str] = "pesn_inventory_report_v1"


class InventoryDataError(RuntimeError):
    """A required upstream artifact is missing or malformed."""


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise InventoryDataError(f"required upstream artifact missing: {path}")


@dataclass(frozen=True, slots=True)
class SourceInventory:
    """In-memory join of every transition-free metadata source."""

    interim_dir: Path
    inventory_rows: dict[str, dict[str, JSONValue]]
    p1_summary: dict[str, dict[str, JSONValue]]
    scan_ready: frozenset[str]
    split_assignment: dict[str, str]
    family_by_reaction: dict[str, str]
    source_hashes: dict[str, str]

    def qualification(self, reaction_id: str) -> str:
        """Typed teacher-training eligibility code for one reaction."""
        summary = self.p1_summary.get(reaction_id)
        if summary is None or str(summary.get("status")) not in RESOLVED_P1_STATUSES:
            return P1_NOT_RESOLVED
        if reaction_id not in self.scan_ready:
            return NOT_SCAN_READY
        layout = summary.get("layout")
        if layout is None or str(layout) not in SUPPORTED_LAYOUTS:
            return LAYOUT_UNSUPPORTED
        if str(summary.get("orientation")) not in SUPPORTED_ORIENTATIONS:
            return ORIENTATION_UNRESOLVED
        if self.split_assignment.get(reaction_id) != "train":
            return NOT_IN_PARENT_TRAIN
        return QUALIFIED

    def qualified_train_reactions(self) -> list[str]:
        return sorted(
            rid
            for rid in self.p1_summary
            if rid in self.inventory_rows and self.qualification(rid) == QUALIFIED
        )


def load_source_inventory(interim_dir: str | Path) -> SourceInventory:
    """Read and join every metadata source under *interim_dir* (no truth)."""
    interim = Path(interim_dir)
    inventory_path = interim / "inventory.parquet"
    p1_summary_path = interim / "g1_p1_summary.parquet"
    split_path = interim / "split_assignment.parquet"
    class_summary_path = interim / "g1_v2_class_summary.parquet"
    scan_ready_path = interim / "g2_scan_ready.json"
    for path in (inventory_path, p1_summary_path, split_path, class_summary_path, scan_ready_path):
        _require_file(path)

    inventory_rows = {
        str(row["reaction_id"]): row
        for row in pq.read_table(inventory_path).to_pylist()
    }
    p1_summary = {
        str(row["reaction_id"]): row
        for row in pq.read_table(p1_summary_path).to_pylist()
    }
    scan_ready_doc = read_json(scan_ready_path)
    scan_ready_ids = scan_ready_doc.get("reaction_ids")
    if not isinstance(scan_ready_ids, list):
        raise InventoryDataError(f"scan-ready id list malformed: {scan_ready_path}")
    split_assignment = {
        str(row["reaction_id"]): str(row["split"])
        for row in pq.read_table(split_path).to_pylist()
    }
    family_by_reaction: dict[str, str] = {}
    for row in pq.read_table(class_summary_path, columns=["reaction_id", "l0u_cluster_id"]).to_pylist():
        family_id = row.get("l0u_cluster_id")
        if isinstance(family_id, str) and family_id:
            family_by_reaction[str(row["reaction_id"])] = family_id

    return SourceInventory(
        interim_dir=interim,
        inventory_rows=inventory_rows,
        p1_summary=p1_summary,
        scan_ready=frozenset(str(rid) for rid in scan_ready_ids),
        split_assignment=split_assignment,
        family_by_reaction=family_by_reaction,
        source_hashes={
            "inventory_parquet": sha256_file(inventory_path),
            "g1_p1_summary_parquet": sha256_file(p1_summary_path),
            "split_assignment_parquet": sha256_file(split_path),
            "g1_v2_class_summary_parquet": sha256_file(class_summary_path),
            "g2_scan_ready_json": sha256_file(scan_ready_path),
        },
    )


def build_inventory_report(interim_dir: str | Path) -> dict[str, JSONValue]:
    """Qualified/unqualified histogram plus per-source hashes."""
    inventory = load_source_inventory(interim_dir)
    counts: dict[str, int] = {}
    for reaction_id in inventory.p1_summary:
        if reaction_id in inventory.inventory_rows:
            code = inventory.qualification(reaction_id)
            counts[code] = counts.get(code, 0) + 1
    return {
        "schema_version": INVENTORY_REPORT_SCHEMA,
        "n_inventory_rows": len(inventory.inventory_rows),
        "n_p1_summary_rows": len(inventory.p1_summary),
        "n_scan_ready": len(inventory.scan_ready),
        "qualification_counts": dict(sorted(counts.items())),
        "n_qualified_train": len(inventory.qualified_train_reactions()),
        "source_hashes": inventory.source_hashes,
        "interim_dir": str(inventory.interim_dir),
    }


__all__ = [
    "INVENTORY_REPORT_SCHEMA",
    "LAYOUT_UNSUPPORTED",
    "NOT_IN_PARENT_TRAIN",
    "NOT_SCAN_READY",
    "ORIENTATION_UNRESOLVED",
    "P1_NOT_RESOLVED",
    "QUALIFIED",
    "RESOLVED_P1_STATUSES",
    "SUPPORTED_LAYOUTS",
    "SUPPORTED_ORIENTATIONS",
    "InventoryDataError",
    "SourceInventory",
    "build_inventory_report",
    "load_source_inventory",
]
