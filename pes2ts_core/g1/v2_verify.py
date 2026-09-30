"""G1 v2 verification: full recomputation of the v2 trees.

Unlike the v1 verifiers (which mostly reconcile fields, files, and digests),
``verify_v2`` **recomputes every bond event and every classification** from
the persisted v2 documents: the exclusive-edit block must be reproducible
from the embedded graph, the reaction center must match, the audit issue
dimensions must agree with the issue codes, the classification ids must be
reproducible byte-for-byte, and the manifest counts/digests must reconcile
with the trees.  The G2 export tree is checked against its whitelist: no
truth-derived key (``ts_irc_index`` above all) may appear in any exported
document.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pes2ts_core.g1.v2_build import v2_document_path, v2_settings
from pes2ts_core.g1.v2_classify import (
    _class_document,
    v2_class_document_path,
    v2_class_settings,
)
from pes2ts_core.g1.v2_edits import (
    exclusive_edits,
    reaction_center,
)
from pes2ts_core.g1.v2_schema import (
    AUDIT_CLEAN,
    AUDIT_EXCLUDED,
    AUDIT_ISSUES,
    G2_SCAN_READY_FILENAME,
    ISSUE_CODES,
    ISSUE_DIMENSIONS,
    V2_CLASS_SCHEMA_VERSION,
    V2_CLASS_SUMMARY_FILENAME,
    V2_EDITS_SCHEMA_VERSION,
    V2_EXPORT_SCHEMA_VERSION,
    V2_GATE_SCHEMA_VERSION,
    V2_MANIFEST_FILENAME,
    V2_SUMMARY_FILENAME,
)
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json
from pes2ts_core.utils.parquet_io import read_parquet

#: Keys that must never appear in a G2 export document (truth-derived).
FORBIDDEN_EXPORT_KEYS: frozenset[str] = frozenset({
    "ts_irc_index", "irc_frames", "ts_coordinates", "EHG", "ehg", "forces",
    "endpoint_match", "orientation", "irc_evidence", "ts_energy", "validation_label",
})


@dataclass(frozen=True, slots=True)
class V2Verification:
    """Outcome of one v2 verification run."""

    n_total: int
    n_clean: int
    n_issues: int
    n_excluded: int
    problems: tuple[str, ...]


def _edit_problems(
    document: Mapping[str, Any], reaction_id: str
) -> list[str]:
    """Recompute one document's edit block and return every disagreement."""
    problems: list[str] = []
    graph = document.get("graph")
    if document.get("audit_status") == AUDIT_EXCLUDED:
        return problems
    if not isinstance(graph, Mapping):
        return [f"{reaction_id}: non-excluded document without a graph"]
    elements = {int(k): str(v) for k, v in graph["elements"].items()}
    r_bonds = [[float(v) for v in bond] for bond in graph["r_bonds"]]
    p_bonds = [[float(v) for v in bond] for bond in graph["p_bonds"]]
    block = exclusive_edits(elements, r_bonds, p_bonds)
    if document.get("edits") != block["edits"]:
        problems.append(f"{reaction_id}: edits not reproducible from the graph")
    if document.get("hydrogen_partner_changes") != block["hydrogen_partner_changes"]:
        problems.append(f"{reaction_id}: hydrogen changes not reproducible")
    if document.get("aromatic_regions") != block["aromatic_regions"]:
        problems.append(f"{reaction_id}: aromatic regions not reproducible")
    center = reaction_center(
        block["edits"], block["hydrogen_partner_changes"], block["_adjacency"], 1,
    )
    if document.get("reaction_center") != center:
        problems.append(f"{reaction_id}: reaction center not reproducible")
    issues = document.get("issues") or []
    for code in issues:
        if code.startswith("p1:"):
            continue
        if code not in ISSUE_CODES:
            problems.append(f"{reaction_id}: unknown issue code {code!r}")
    expected_dimensions = sorted({
        ISSUE_DIMENSIONS[code] for code in issues if code in ISSUE_DIMENSIONS
    })
    if sorted(document.get("dimensions") or []) != expected_dimensions:
        problems.append(f"{reaction_id}: dimensions disagree with issue codes")
    return problems


def verify_v2(config: Mapping[str, Any], *, with_classes: bool = True) -> V2Verification:
    """Recompute both v2 trees and reconcile them with the manifests."""
    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
    settings = v2_settings(config)
    shard_size = int(settings["shard_size"])
    summary_path = interim_dir / V2_SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing v2 summary {summary_path}; run `g1 v2-build` first")
    manifest_path = manifests_dir / V2_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing v2 manifest {manifest_path}; run `g1 v2-build` first")
    rows = read_parquet(summary_path).to_pylist()
    problems: list[str] = []
    histogram = Counter(str(row["audit_status"]) for row in rows)
    for row in rows:
        reaction_id = str(row["reaction_id"])
        path = v2_document_path(interim_dir, reaction_id, shard_size)
        if not path.is_file():
            problems.append(f"summary row {reaction_id} has no document at {path}")
            continue
        document = read_json(path)
        if not isinstance(document, Mapping):
            problems.append(f"{path}: document is not a JSON object")
            continue
        if document.get("schema_version") != V2_EDITS_SCHEMA_VERSION:
            problems.append(f"{path}: schema_version mismatch")
        if str(document.get("audit_status")) not in (AUDIT_CLEAN, AUDIT_ISSUES, AUDIT_EXCLUDED):
            problems.append(f"{path}: unknown audit_status")
        problems.extend(_edit_problems(document, reaction_id))
    manifest = read_json(manifest_path)
    if isinstance(manifest, Mapping):
        expected = {
            "n_total": len(rows),
            "n_clean": histogram.get(AUDIT_CLEAN, 0),
            "n_issues": histogram.get(AUDIT_ISSUES, 0),
            "n_excluded": histogram.get(AUDIT_EXCLUDED, 0),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                problems.append(f"v2 manifest {key}={manifest.get(key)!r} but the tree says {value!r}")
        if manifest.get("summary_sha256") != sha256_file(summary_path):
            problems.append("v2 manifest summary_sha256 does not match the summary Parquet")
    else:
        problems.append(f"{manifest_path}: manifest is not a JSON object")
    if with_classes:
        problems.extend(_verify_classes(config, interim_dir))
    problems.extend(_verify_export(config, interim_dir))
    return V2Verification(
        len(rows),
        histogram.get(AUDIT_CLEAN, 0),
        histogram.get(AUDIT_ISSUES, 0),
        histogram.get(AUDIT_EXCLUDED, 0),
        tuple(problems),
    )


def _verify_classes(config: Mapping[str, Any], interim_dir: Path) -> list[str]:
    """Recompute every v2 classification and reconcile the class tree."""
    settings = v2_class_settings(config)
    shard_size = int(v2_settings(config)["shard_size"])
    class_summary_path = interim_dir / V2_CLASS_SUMMARY_FILENAME
    if not class_summary_path.is_file():
        return [f"Missing v2 class summary {class_summary_path}; run `g1 v2-classify` first"]
    problems: list[str] = []
    rows = read_parquet(class_summary_path).to_pylist()
    n_classified = 0
    for row in rows:
        reaction_id = str(row["reaction_id"])
        path = v2_class_document_path(interim_dir, reaction_id, shard_size)
        if not path.is_file():
            problems.append(f"class summary row {reaction_id} has no document at {path}")
            continue
        document = read_json(path)
        if not isinstance(document, Mapping):
            problems.append(f"{path}: document is not a JSON object")
            continue
        if document.get("schema_version") != V2_CLASS_SCHEMA_VERSION:
            problems.append(f"{path}: schema_version mismatch")
        if str(row["status"]) == "classified":
            n_classified += 1
        v2_path = v2_document_path(interim_dir, reaction_id, shard_size)
        v2_document = read_json(v2_path)
        if isinstance(v2_document, Mapping):
            recomputed = _class_document(v2_document, settings)
            if recomputed.get("levels", {}).get("l1_center_template", {}).get("cluster_id") != (
                document.get("levels", {}).get("l1_center_template", {}).get("cluster_id")
            ):
                problems.append(f"{path}: L1 cluster id not reproducible")
            if recomputed.get("family_labels") != document.get("family_labels"):
                problems.append(f"{path}: family labels not reproducible")
            if recomputed.get("levels", {}).get("l0_edit_family", {}).get("cluster_id") != (
                document.get("levels", {}).get("l0_edit_family", {}).get("cluster_id")
            ):
                problems.append(f"{path}: L0 cluster id not reproducible")
    return problems


def _verify_export(config: Mapping[str, Any], interim_dir: Path) -> list[str]:
    """Check the export tree against the whitelist and the scan-ready list."""
    ready_path = interim_dir / G2_SCAN_READY_FILENAME
    if not ready_path.is_file():
        return []
    ready = read_json(ready_path)
    problems: list[str] = []
    ids: list[str] = []
    if isinstance(ready, Mapping):
        ids = [str(value) for value in ready.get("reaction_ids") or []]
        if ready.get("schema_version") != V2_GATE_SCHEMA_VERSION:
            problems.append("scan-ready list schema_version mismatch")
        if ready.get("n_eligible") != len(ids):
            problems.append("scan-ready list n_eligible disagrees with the id list")
    else:
        problems.append(f"{ready_path}: not a JSON object")
        return problems
    export_root = interim_dir / "g1_v2" / "export"
    if not export_root.is_dir():
        return problems + ["scan-ready list exists but the export tree is missing"]
    seen: set[str] = set()
    export_stamps = (
        f'"schema_version":"{V2_EXPORT_SCHEMA_VERSION}"',
        f'"schema_version": "{V2_EXPORT_SCHEMA_VERSION}"',
    )
    for path in sorted(export_root.rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        reaction_id = path.stem
        seen.add(reaction_id)
        if not any(stamp in text for stamp in export_stamps):
            problems.append(f"{path}: export document schema_version mismatch")
        for key in FORBIDDEN_EXPORT_KEYS:
            if f'"{key}"' in text:
                problems.append(f"{path}: forbidden truth-derived key {key!r}")
                break
    missing = sorted(set(ids) - seen)
    stray = sorted(seen - set(ids))
    if missing:
        problems.append(f"export tree is missing {len(missing)} scan-ready reaction(s)")
    if stray:
        problems.append(f"export tree carries {len(stray)} non-scan-ready reaction(s)")
    return problems


__all__ = [
    "FORBIDDEN_EXPORT_KEYS",
    "V2Verification",
    "verify_v2",
]
