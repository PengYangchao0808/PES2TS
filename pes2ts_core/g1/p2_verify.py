"""P2 artifact verification: recompute the taxonomy and reconcile the tree.

:func:`verify_p2` re-reads every P2 document, recomputes the classification
from the underlying P1 document with the recorded taxonomy settings, and
refuses any drift: cluster ids must be reproducible byte-for-byte from the
inputs (the determinism the plan's P2 acceptance demands), statuses must lie
in the enumeration, and the manifest counts/digests must agree with the
tree and the summary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pes2ts_core.g1.p1_truth import p1_document_path, p1_settings
from pes2ts_core.g1.p2_build import p2_document_path, p2_settings
from pes2ts_core.g1.reaction_class import LEVEL_L1
from pes2ts_core.g1.truth_schema import (
    G2_ELIGIBLE_STATUSES,
    P2_DOCUMENT_SCHEMA_VERSION,
    P2_MANIFEST_FILENAME,
    P2_MANIFEST_SCHEMA_VERSION,
    P2_STATUSES,
    P2_SUMMARY_FILENAME,
    STATUS_CLASSIFIED,
    STATUS_UNCLASSIFIABLE,
)
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json
from pes2ts_core.utils.parquet_io import read_parquet


@dataclass(frozen=True, slots=True)
class P2Verification:
    """Outcome of one P2 verification run."""

    n_total: int
    n_classified: int
    problems: tuple[str, ...]


def verify_p2(config: Mapping[str, Any]) -> P2Verification:
    """Recompute every classification and reconcile the P2 artifacts."""
    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
    settings = p2_settings(config)
    shard_size = int(p1_settings(config)["shard_size"])
    summary_path = interim_dir / P2_SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing P2 summary {summary_path}; run `g1 classify` first"
        )
    manifest_path = manifests_dir / P2_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing P2 manifest {manifest_path}; run `g1 classify` first"
        )
    rows = read_parquet(summary_path).to_pylist()
    problems: list[str] = []
    n_classified = 0
    for row in rows:
        reaction_id = str(row["reaction_id"])
        path = p2_document_path(interim_dir, reaction_id, shard_size)
        if not path.is_file():
            problems.append(f"summary row {reaction_id} has no document at {path}")
            continue
        document = read_json(path)
        if not isinstance(document, Mapping):
            problems.append(f"{path}: document is not a JSON object")
            continue
        if document.get("schema_version") != P2_DOCUMENT_SCHEMA_VERSION:
            problems.append(f"{path}: schema_version mismatch")
        status = str(document.get("status"))
        if status not in P2_STATUSES:
            problems.append(f"{path}: unknown status {status!r}")
        if status == STATUS_CLASSIFIED:
            n_classified += 1
        else:
            reasons = document.get("status_detail")
            if not reasons:
                problems.append(f"{path}: unclassifiable without a status_detail")
        p1_path = p1_document_path(interim_dir, reaction_id, shard_size)
        p1_document = read_json(p1_path)
        if isinstance(p1_document, Mapping):
            from pes2ts_core.g1.p2_build import _classify_document

            recomputed = _classify_document(p1_document, settings)
            if recomputed.get("levels", {}).get(LEVEL_L1, {}).get("cluster_id") != (
                document.get("levels", {}).get(LEVEL_L1, {}).get("cluster_id")
            ):
                problems.append(f"{path}: L1 cluster id not reproducible from the P1 document")
            if recomputed.get("family_labels") != document.get("family_labels"):
                problems.append(f"{path}: family labels not reproducible")
        else:
            problems.append(f"{p1_path}: underlying P1 document is not a JSON object")
    status_histogram = Counter(str(row["status"]) for row in rows)
    manifest = read_json(manifest_path)
    if isinstance(manifest, Mapping):
        if manifest.get("n_total") != len(rows):
            problems.append("manifest n_total disagrees with the summary")
        if manifest.get("n_classified") != n_classified:
            problems.append("manifest n_classified disagrees with the summary")
        if manifest.get("taxonomy_version") != settings["taxonomy_version"]:
            problems.append("manifest taxonomy_version disagrees with the configuration")
        expected_by_status = {
            status: status_histogram[status]
            for status in P2_STATUSES
            if status_histogram.get(status)
        }
        if manifest.get("by_status") != expected_by_status:
            problems.append("manifest by_status disagrees with the summary")
        if manifest.get("summary_sha256") != sha256_file(summary_path):
            problems.append("manifest summary_sha256 does not match the summary Parquet")
        if manifest.get("schema_version") != P2_MANIFEST_SCHEMA_VERSION:
            problems.append("manifest schema_version mismatch")
    else:
        problems.append(f"{manifest_path}: manifest is not a JSON object")
    return P2Verification(len(rows), n_classified, tuple(problems))


__all__ = [
    "P2Verification",
    "verify_p2",
]
