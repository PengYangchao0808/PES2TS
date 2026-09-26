"""Post-build verification of the persisted G1 reaction-change tree.

:func:`verify_g1` re-reads every written document and reconciles the tree with
the summary Parquet and the manifest: exactly one document per summary row and
no stray files, matching schema version and reaction id, non-empty component
tables on valid documents, a failure code on rejected documents, no private
``_union_edges`` carrier, and manifest counts, histogram, file count, and
summary digest that all agree with the tree.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pes2ts_core.g1 import G1_MANIFEST_SCHEMA_VERSION
from pes2ts_core.g1.build import (
    BUILD_HINT, DEFAULT_SHARD_SIZE, MANIFEST_FILENAME, SUMMARY_FILENAME,
    reaction_change_path, reactions_dir,
)
from pes2ts_core.g1.document import CHANGE_SCHEMA_VERSION
from pes2ts_core.utils.hashing import JSONValue, sha256_file
from pes2ts_core.utils.jsonio import read_json
from pes2ts_core.utils.parquet_io import read_parquet


@dataclass(frozen=True, slots=True)
class G1Verification:
    """Outcome of one :func:`verify_g1` reconciliation."""

    n_total: int
    n_valid: int
    n_rejected: int
    problems: tuple[str, ...]


def _shard_size(config: Mapping[str, Any]) -> int:
    """Return the configured shard size (build default when absent)."""
    raw = config.get("g1")
    settings: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    return int(settings.get("shard_size", DEFAULT_SHARD_SIZE))


def _document_problems(document: Mapping[str, Any], reaction_id: str, path: Path) -> list[str]:
    """Return every schema/contract violation of one persisted document."""
    problems: list[str] = []
    if document.get("schema_version") != CHANGE_SCHEMA_VERSION:
        problems.append(f"{path}: schema_version mismatch")
    if str(document.get("reaction_id")) != reaction_id or path.stem != reaction_id:
        problems.append(f"{path}: reaction_id does not match the summary row")
    validation = document.get("validation")
    if not isinstance(validation, Mapping):
        problems.append(f"{path}: missing validation block")
    elif validation.get("status") == "valid":
        if validation.get("failure_code") is not None:
            problems.append(f"{path}: valid document carries a failure code")
        for table in ("reactants", "products"):
            entries = document.get(table)
            if not (isinstance(entries, list) and entries):
                problems.append(f"{path}: valid document has an empty {table} table")
    elif validation.get("status") == "rejected":
        if not validation.get("failure_code"):
            problems.append(f"{path}: rejected document lacks a failure code")
        if document.get("reactants") or document.get("products"):
            problems.append(f"{path}: rejected document carries component tables")
    else:
        problems.append(f"{path}: unknown validation status")
    if "_union_edges" in path.read_text(encoding="utf-8"):
        problems.append(f"{path}: private _union_edges carrier persisted")
    return problems


def verify_g1(config: Mapping[str, Any]) -> G1Verification:
    """Re-read the written tree and reconcile it with the summary and manifest.

    A missing summary or manifest raises :class:`FileNotFoundError` with the
    ``g1 build`` hint; every contract violation is collected (not
    short-circuited) so one verification run reports the whole damage.
    """
    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
    summary_path = interim_dir / SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary {summary_path}; {BUILD_HINT}")
    manifest_path = manifests_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest {manifest_path}; {BUILD_HINT}")
    rows = read_parquet(summary_path).to_pylist()
    problems: list[str] = []
    document_ids: set[str] = set()
    for row in rows:
        reaction_id = str(row["reaction_id"])
        path = reaction_change_path(interim_dir, reaction_id, _shard_size(config))
        if not path.is_file():
            problems.append(f"summary row {reaction_id} has no document at {path}")
            continue
        document = read_json(path)
        if not isinstance(document, Mapping):
            problems.append(f"{path}: document is not a JSON object")
            continue
        document_ids.add(reaction_id)
        problems.extend(_document_problems(document, reaction_id, path))
    reactions_path = reactions_dir(interim_dir)
    if reactions_path.is_dir():
        for path in sorted(reactions_path.rglob("*.json")):
            if path.stem not in document_ids:
                problems.append(f"stray document {path} has no summary row")
    n_valid = sum(1 for row in rows if str(row["status"]) == "valid")
    expected_by_code = Counter(str(row["failure_code"]) for row in rows if row["failure_code"])
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        problems.append(f"{manifest_path}: manifest is not a JSON object")
    else:
        expected: dict[str, JSONValue] = {
            "n_total": len(rows), "n_valid": n_valid, "n_rejected": len(rows) - n_valid,
            "n_files": len(rows),
            "by_code": {code: expected_by_code[code] for code in sorted(expected_by_code)},
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                problems.append(f"manifest {key}={manifest.get(key)!r} but the tree says {value!r}")
        if manifest.get("schema_version") != G1_MANIFEST_SCHEMA_VERSION:
            problems.append("manifest schema_version mismatch")
        if manifest.get("summary_sha256") != sha256_file(summary_path):
            problems.append("manifest summary_sha256 does not match the summary Parquet")
    return G1Verification(len(rows), n_valid, len(rows) - n_valid, tuple(problems))


__all__ = ["G1Verification", "verify_g1"]
