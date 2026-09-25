"""Adoption of the authors' official reaction-level train/valid/test split.

:func:`adopt_official_split` reads the three official split CSVs through
:mod:`pes2ts_core.g0.split_sources` (located by their ``_train.csv`` /
``_valid.csv`` / ``_test.csv`` suffixes, never hard-coded) and validates the
resulting assignment against the TS-free inventory.  Every inventory reaction
absent from all three files is recorded as a ``NOT_IN_SPLIT`` rejection in the
unified ledger (never dropped silently), and the assignment Parquet plus the
split manifest are written atomically with ``leak_status="pending"``; the
leakage verdict belongs to the later freeze stage.  The split CSVs are
read-only and are never mutated.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.rejections import (
    LEDGER_FILENAME,
    Rejection,
    RejectionCode,
    RejectionLedger,
)
from pes2ts_core.g0.split_sources import (
    SPLIT_LABELS,
    SplitSchemaError,
    build_assignment,
    read_inventory_ids,
    resolve_split_files,
)
from pes2ts_core.utils.hashing import JSONValue, sha256_file
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.utils.parquet_io import write_parquet

logger = logging.getLogger(__name__)

#: Assignment table filename written under ``config["paths"]["interim"]``.
SPLIT_ASSIGNMENT_FILENAME: Final[str] = "split_assignment.parquet"
#: Split manifest filename written under ``config["paths"]["manifests"]``.
SPLIT_MANIFEST_FILENAME: Final[str] = "split_manifest.json"
#: Stage name stamped into every rejection emitted by this module.
SPLIT_STAGE: Final[str] = "adopt_official_split"
#: Strategy value recorded in the manifest.
SPLIT_STRATEGY: Final[str] = "authors_official"
#: Leakage verdict placeholder; the freeze stage owns the final decision.
LEAK_STATUS_PENDING: Final[str] = "pending"
#: CLI exit code for an authoritative split CSV violating the expected schema.
EXIT_SPLIT_SCHEMA_ERROR: Final[int] = 5
#: Maximum number of split-only IDs quoted in the manifest.
MAX_SPLIT_ONLY_EXAMPLES: Final[int] = 20


@dataclass(frozen=True, slots=True)
class SplitAdoptionResult:
    """Outcome of one :func:`adopt_official_split` run."""

    covered: int
    counts: dict[str, int]
    manifest_path: Path
    assignment_path: Path
    n_not_in_split: int


def _read_existing_rejections(ledger_path: Path) -> list[Rejection]:
    """Return the rejections already persisted in *ledger_path*.

    The unified ledger format is owned by :mod:`pes2ts_core.g0.rejections`;
    this minimal JSONL read-back lets the split stage append without clobbering
    entries written by earlier stages.
    """
    if not ledger_path.is_file():
        return []
    rejections: list[Rejection] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rejections.append(
            Rejection(
                reaction_id=record["reaction_id"],
                stage=record["stage"],
                code=RejectionCode(record["code"]),
                detail=record["detail"],
                source_pointer=record["source_pointer"],
            )
        )
    return rejections


def _append_rejections(manifests_dir: Path, rejections: Sequence[Rejection]) -> None:
    """Persist *rejections* after every ledger entry already on disk.

    Exact duplicates of pre-existing entries are skipped so repeated runs stay
    idempotent; entries written by other stages are preserved verbatim.
    """
    ledger = RejectionLedger(manifests_dir)
    existing = _read_existing_rejections(manifests_dir / LEDGER_FILENAME)
    known = set(existing)
    for rejection in existing:
        ledger.add(rejection)
    for rejection in rejections:
        if rejection in known:
            continue
        ledger.add(rejection)
        known.add(rejection)
    ledger.write()


def adopt_official_split(config: Mapping[str, Any]) -> SplitAdoptionResult:
    """Adopt the authors' official split and persist the assignment map.

    The three split CSVs are resolved by suffix from
    ``config["source"]["files"]`` and read from ``config["paths"]["raw"]``; the
    inventory is read from ``config["paths"]["interim"]``.  Every inventory ID
    absent from all three files becomes a ``NOT_IN_SPLIT`` rejection, IDs
    appearing only in the split files are counted as ``n_split_only`` (the
    inventory may have rejected rows), and only inventory-backed assignments
    are written to the assignment Parquet sorted by reaction ID.  The manifest
    records the strategy, seed, source digests, and counts with
    ``leak_status="pending"``; no leakage verdict is produced here.
    """
    files = resolve_split_files(config)
    assignment = build_assignment(files)
    inventory_path, inventory_ids = read_inventory_ids(config)
    inventory = set(inventory_ids)

    counts: dict[str, int] = {
        label: sum(1 for value in assignment.values() if value == label)
        for label in SPLIT_LABELS
    }
    counts["total"] = sum(counts.values())
    covered_ids = sorted(inventory & assignment.keys())
    not_in_split = sorted(inventory - assignment.keys())
    split_only = sorted(assignment.keys() - inventory)

    assignment_path = Path(config["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME
    manifest_path = Path(config["paths"]["manifests"]) / SPLIT_MANIFEST_FILENAME
    write_parquet(
        assignment_path,
        {
            "reaction_id": covered_ids,
            "split": [assignment[reaction_id] for reaction_id in covered_ids],
        },
    )

    _append_rejections(
        manifest_path.parent,
        [
            Rejection(
                reaction_id=reaction_id,
                stage=SPLIT_STAGE,
                code=RejectionCode.NOT_IN_SPLIT,
                detail=(
                    f"Inventory reaction {reaction_id} is absent from all "
                    "official split files"
                ),
                source_pointer=str(inventory_path),
            )
            for reaction_id in not_in_split
        ],
    )

    counts_json: dict[str, JSONValue] = {key: value for key, value in counts.items()}
    split_only_examples: list[JSONValue] = list(
        split_only[:MAX_SPLIT_ONLY_EXAMPLES]
    )
    manifest: dict[str, JSONValue] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "strategy": SPLIT_STRATEGY,
        "seed": config["split"]["seed"],
        "source_sha256": {
            filename: sha256_file(path) for filename, path in files.values()
        },
        "counts": counts_json,
        "covered": len(covered_ids),
        "n_inventory": len(inventory),
        "n_not_in_split": len(not_in_split),
        "n_split_only": len(split_only),
        "split_only_examples": split_only_examples,
        "leak_status": LEAK_STATUS_PENDING,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_json(manifest_path, manifest)

    logger.info(
        "Split adoption: train=%d valid=%d test=%d covered=%d/%d "
        "not_in_split=%d split_only=%d -> %s",
        counts["train"],
        counts["valid"],
        counts["test"],
        len(covered_ids),
        len(inventory),
        len(not_in_split),
        len(split_only),
        manifest_path,
    )
    return SplitAdoptionResult(
        covered=len(covered_ids),
        counts=counts,
        manifest_path=manifest_path,
        assignment_path=assignment_path,
        n_not_in_split=len(not_in_split),
    )


__all__ = [
    "EXIT_SPLIT_SCHEMA_ERROR",
    "LEAK_STATUS_PENDING",
    "MAX_SPLIT_ONLY_EXAMPLES",
    "SPLIT_ASSIGNMENT_FILENAME",
    "SPLIT_LABELS",
    "SPLIT_MANIFEST_FILENAME",
    "SPLIT_STAGE",
    "SPLIT_STRATEGY",
    "SplitAdoptionResult",
    "SplitSchemaError",
    "adopt_official_split",
]
