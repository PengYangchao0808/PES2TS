"""Final split manifest contract and its atomic writers.

The freeze stage's halted and committed manifests are both produced from
:class:`FreezeManifestState` through a single :func:`manifest_document`
builder, so a new manifest key or count can never land in one outcome only.
All writers delegate to the atomic JSON/Parquet utilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.split_policy import OffenseTuple
from pes2ts_core.g0.split_sources import SPLIT_LABELS
from pes2ts_core.utils.hashing import JSONValue
from pes2ts_core.utils.jsonio import atomic_writer, read_json
from pes2ts_core.utils.parquet_io import write_parquet

#: Actionable hint attached to a missing adoption manifest.
SPLIT_HINT: Final[str] = "run `g0 split` first"


@dataclass(frozen=True, slots=True)
class FreezeManifestState:
    """Validated inputs of one final split manifest."""

    strategy: str
    counts: dict[str, int]
    covered: int
    n_inventory: int
    n_not_in_split: int
    n_split_only: int
    leak_status: str
    offending: tuple[OffenseTuple, ...]
    remediation: dict[str, JSONValue]
    audit: dict[str, JSONValue]


def read_manifest(path: Path) -> dict[str, JSONValue]:
    """Return the JSON object stored at *path* (a non-object raises)."""
    document = read_json(path)
    if not isinstance(document, dict):
        msg = f"{path} is not a JSON object"
        raise ValueError(msg)
    return document


def stored_int(document: Mapping[str, JSONValue], key: str, fallback: int) -> int:
    """Return *document[key]* when it is a plain integer, else *fallback*."""
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    return value


def stored_counts(document: Mapping[str, JSONValue]) -> dict[str, int]:
    """Return the integer count fields of a stored manifest."""
    raw = document.get("counts")
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in raw.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def assignment_counts(assignment: Mapping[str, str]) -> dict[str, int]:
    """Return the train/valid/test counts plus total of *assignment*."""
    counts = dict.fromkeys(SPLIT_LABELS, 0)
    for label in assignment.values():
        counts[label] += 1
    counts["total"] = sum(counts[label] for label in SPLIT_LABELS)
    return counts


def assignment_coverage(
    assignment: Mapping[str, str], inventory_ids: list[str]
) -> tuple[list[str], list[str], int]:
    """Return ``(covered_ids, split_only_ids, n_inventory)``."""
    inventory = set(inventory_ids)
    covered = sorted(inventory & assignment.keys())
    split_only = sorted(assignment.keys() - inventory)
    return covered, split_only, len(inventory)


def audit_summary(
    n_pairs: int, n_known: int, max_similarity: float
) -> dict[str, JSONValue]:
    """Return the manifest's audit summary block."""
    return {
        "complete": True,
        "n_pairs_over_threshold": n_pairs,
        "n_cross_split_known_duplicates": n_known,
        "max_similarity": max_similarity,
    }


def manifest_document(
    config: Mapping[str, Any],
    previous: Mapping[str, JSONValue],
    state: FreezeManifestState,
) -> dict[str, JSONValue]:
    """Build the final split manifest, preserving the adoption facts."""
    source_sha256 = previous.get("source_sha256")
    if not isinstance(source_sha256, dict):
        msg = f"Split manifest has no source_sha256; {SPLIT_HINT}"
        raise ValueError(msg)
    stored_version = previous.get("dataset_version")
    counts: dict[str, JSONValue] = {
        key: value for key, value in state.counts.items()
    }
    offending: list[JSONValue] = [entry.to_record() for entry in state.offending]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": (
            stored_version
            if isinstance(stored_version, str)
            else dataset_version(config)
        ),
        "strategy": state.strategy,
        "seed": int(config["split"]["seed"]),
        "source_sha256": {key: value for key, value in source_sha256.items()},
        "counts": counts,
        "covered": state.covered,
        "n_inventory": state.n_inventory,
        "n_not_in_split": state.n_not_in_split,
        "n_split_only": state.n_split_only,
        "leak_status": state.leak_status,
        "threshold": float(config["near_dup"]["threshold"]),
        "offending_tuples": offending,
        "remediation": {
            key: value for key, value in state.remediation.items()
        },
        "audit": {key: value for key, value in state.audit.items()},
        "frozen": True,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_assignment(path: Path, assignment: Mapping[str, str]) -> None:
    """Atomically write the sorted two-column assignment Parquet."""
    ordered = sorted(assignment)
    write_parquet(
        path,
        {
            "reaction_id": ordered,
            "split": [assignment[reaction_id] for reaction_id in ordered],
        },
    )


def restore_bytes(path: Path, payload: bytes) -> None:
    """Atomically restore *path* to the previously captured *payload*."""
    with atomic_writer(path) as handle:
        handle.write(payload)


__all__ = [
    "SPLIT_HINT",
    "FreezeManifestState",
    "assignment_counts",
    "assignment_coverage",
    "audit_summary",
    "manifest_document",
    "read_manifest",
    "restore_bytes",
    "stored_counts",
    "stored_int",
    "write_assignment",
]
