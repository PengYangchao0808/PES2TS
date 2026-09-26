"""P1 artifact verification: re-read the persisted tree and reconcile it.

:func:`verify_p1` re-reads every written P1 document and checks the plan's
P1 acceptance contracts: schema version and id agreement with the summary,
statuses inside the enumeration, eligibility consistent with the status, a
complete map table (all maps 1..n with TS/IRC rows) on every eligible
document, no persisted trajectory arrays, and manifest counts/digests that
agree with the tree.  Problems are collected, never short-circuited.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pes2ts_core.g1.p1_truth import p1_document_path, p1_settings
from pes2ts_core.g1.truth_schema import (
    G1_TRUTH_DIRNAME,
    G2_ELIGIBLE_STATUSES,
    P1_DOCUMENT_SCHEMA_VERSION,
    P1_MANIFEST_FILENAME,
    P1_MANIFEST_SCHEMA_VERSION,
    P1_MAPPING_DIRNAME,
    P1_STATUSES,
    P1_SUMMARY_FILENAME,
)
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json
from pes2ts_core.utils.parquet_io import read_parquet

#: Keys that must never appear in a persisted P1 document (truth isolation).
FORBIDDEN_KEYS: tuple[str, ...] = ("coordinates", "EHG", "forces")


@dataclass(frozen=True, slots=True)
class P1Verification:
    """Outcome of one P1 verification run."""

    n_total: int
    n_eligible: int
    problems: tuple[str, ...]


def _document_problems(
    document: Mapping[str, Any], reaction_id: str, path: Path
) -> list[str]:
    """Return every contract violation of one persisted P1 document."""
    problems: list[str] = []
    if document.get("schema_version") != P1_DOCUMENT_SCHEMA_VERSION:
        problems.append(f"{path}: schema_version mismatch")
    if str(document.get("reaction_id")) != reaction_id or path.stem != reaction_id:
        problems.append(f"{path}: reaction_id mismatch")
    status = str(document.get("status"))
    if status not in P1_STATUSES:
        problems.append(f"{path}: unknown status {status!r}")
    eligible = status in G2_ELIGIBLE_STATUSES
    if bool(document.get("truth_assisted")) is not True:
        problems.append(f"{path}: truth_assisted flag missing")
    sources = document.get("sources")
    if not isinstance(sources, Mapping) or not sources.get("truth_manifest_digest"):
        problems.append(f"{path}: sources block lacks truth provenance")
    for key in FORBIDDEN_KEYS:
        if key in document:
            problems.append(f"{path}: forbidden truth array {key!r} persisted")
    if eligible:
        mapping = document.get("mapping")
        if not isinstance(mapping, Mapping) or not mapping.get("map_to_atoms"):
            problems.append(f"{path}: eligible document has no mapping table")
        else:
            maps = [int(entry["map"]) for entry in mapping["map_to_atoms"]]
            if maps != list(range(1, len(maps) + 1)):
                problems.append(f"{path}: mapping table does not cover maps 1..n exactly")
            rows = [int(entry["ts_irc_index"]) for entry in mapping["map_to_atoms"]]
            if len(set(rows)) != len(rows):
                problems.append(f"{path}: ts_irc_index rows are not a bijection")
        if not isinstance(document.get("irc_validation"), Mapping):
            problems.append(f"{path}: eligible document lacks irc_validation")
        if not isinstance(document.get("graph"), Mapping):
            problems.append(f"{path}: eligible document lacks the graph payload")
    elif not str(document.get("status_detail") or ""):
        problems.append(f"{path}: non-eligible status without a status_detail")
    return problems


def verify_p1(config: Mapping[str, Any]) -> P1Verification:
    """Re-read the P1 tree and reconcile it with the summary and manifest."""
    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
    settings = p1_settings(config)
    shard_size = int(settings["shard_size"])
    summary_path = interim_dir / P1_SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing P1 summary {summary_path}; run `g1 resolve-map --allow-truth` first"
        )
    manifest_path = manifests_dir / P1_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing P1 manifest {manifest_path}; run `g1 resolve-map --allow-truth` first"
        )
    rows = read_parquet(summary_path).to_pylist()
    problems: list[str] = []
    document_ids: set[str] = set()
    for row in rows:
        reaction_id = str(row["reaction_id"])
        path = p1_document_path(interim_dir, reaction_id, shard_size)
        if not path.is_file():
            problems.append(f"summary row {reaction_id} has no document at {path}")
            continue
        document = read_json(path)
        if not isinstance(document, Mapping):
            problems.append(f"{path}: document is not a JSON object")
            continue
        document_ids.add(reaction_id)
        problems.extend(_document_problems(document, reaction_id, path))
        if bool(row["g2_eligible"]) != (str(row["status"]) in G2_ELIGIBLE_STATUSES):
            problems.append(f"{path}: summary g2_eligible disagrees with the status")
    mapping_root = interim_dir / G1_TRUTH_DIRNAME / P1_MAPPING_DIRNAME
    if mapping_root.is_dir():
        for path in sorted(mapping_root.rglob("*.json")):
            if path.stem not in document_ids:
                problems.append(f"stray P1 document {path} has no summary row")
    status_histogram = Counter(str(row["status"]) for row in rows)
    n_eligible = sum(1 for row in rows if bool(row["g2_eligible"]))
    manifest = read_json(manifest_path)
    if isinstance(manifest, Mapping):
        if manifest.get("n_total") != len(rows):
            problems.append("manifest n_total disagrees with the summary")
        if manifest.get("n_eligible") != n_eligible:
            problems.append("manifest n_eligible disagrees with the summary")
        expected_by_status = {
            status: status_histogram[status]
            for status in P1_STATUSES
            if status_histogram.get(status)
        }
        if manifest.get("by_status") != expected_by_status:
            problems.append("manifest by_status disagrees with the summary")
        if manifest.get("summary_sha256") != sha256_file(summary_path):
            problems.append("manifest summary_sha256 does not match the summary Parquet")
        if manifest.get("schema_version") != P1_MANIFEST_SCHEMA_VERSION:
            problems.append("manifest schema_version mismatch")
    else:
        problems.append(f"{manifest_path}: manifest is not a JSON object")
    return P1Verification(len(rows), n_eligible, tuple(problems))


__all__ = [
    "P1Verification",
    "verify_p1",
]
