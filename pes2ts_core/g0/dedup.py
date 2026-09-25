"""Full-dataset exact-duplicate and reverse-reaction detection.

Rows are grouped by the map- and direction-invariant
:func:`~pes2ts_core.g0.identity.canonical_reaction_identity` in one O(N)
hashing pass (never a pairwise SMILES comparison); each group's lowest
``reaction_id`` is canonical and every other member is classified against it
with the DIRECTIONAL :func:`~pes2ts_core.g0.identity.directional_preimage`:
same written side order -> ``DUPLICATE_OF``, swapped -> ``REVERSE_OF``. The
inventory is never modified; memberships go to ``duplicate_groups.json`` and
``duplicate_ledger.json``, and rejections are APPENDED to the unified ledger
(loaded first, because its ``write`` rewrites the full in-memory state).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.identity import (
    canonical_reaction_identity,
    directional_preimage,
)
from pes2ts_core.g0.rejections import Rejection, RejectionCode, RejectionLedger
from pes2ts_core.utils.hashing import JSONValue
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.utils.parquet_io import read_parquet

logger = logging.getLogger(__name__)

#: Group memberships written under ``config["paths"]["interim"]``.
DUPLICATE_GROUPS_FILENAME: Final[str] = "duplicate_groups.json"
#: Counts, groups, and classified pairs written under ``config["paths"]["manifests"]``.
DUPLICATE_LEDGER_FILENAME: Final[str] = "duplicate_ledger.json"
#: Stage name stamped into every rejection emitted by this module.
DEDUP_STAGE: Final[str] = "detect_duplicates"
#: Pair kind for a member written in the same direction as its canonical member.
EXACT_DUPLICATE: Final[str] = "duplicate"
#: Pair kind for a member written as the reverse of its canonical member.
REVERSE: Final[str] = "reverse"
#: Inventory column holding the normalized reaction identifier.
_ID_COLUMN: Final[str] = "reaction_id"
#: Inventory column holding the reaction SMILES whose identity is computed.
_SMILES_COLUMN: Final[str] = "reaction_smiles"
_CODE_BY_KIND: Final[dict[str, RejectionCode]] = {
    EXACT_DUPLICATE: RejectionCode.DUPLICATE_OF,
    REVERSE: RejectionCode.REVERSE_OF,
}


@dataclass(frozen=True, slots=True)
class DedupResult:
    """Outcome of one :func:`detect_duplicates` run.

    ``n_unique`` counts distinct direction-agnostic identities, so ``n_rows ==
    n_unique + n_exact_duplicates + n_reverse_pairs`` always holds.
    """

    n_rows: int
    n_unique: int
    n_duplicate_groups: int
    n_exact_duplicates: int
    n_reverse_pairs: int
    groups_path: Path
    ledger_path: Path


@dataclass(frozen=True, slots=True)
class _DuplicatePair:
    """One non-canonical member and its classification against the canonical."""

    kind: str
    canonical_reaction_id: str
    other_reaction_id: str
    detail: str


def _read_inventory(inventory_path: Path) -> tuple[list[str], list[str]]:
    """Return the inventory's ``(reaction_ids, reaction_smiles)`` columns."""
    table = read_parquet(inventory_path)
    missing = [
        name for name in (_ID_COLUMN, _SMILES_COLUMN) if name not in table.column_names
    ]
    if missing:
        msg = (
            f"Inventory {inventory_path} is missing required column(s) {missing}; "
            f"found {sorted(table.column_names)}"
        )
        raise ValueError(msg)
    reaction_ids = [str(value) for value in table.column(_ID_COLUMN).to_pylist()]
    smiles = [str(value) for value in table.column(_SMILES_COLUMN).to_pylist()]
    return reaction_ids, smiles


def _group_by_identity(
    reaction_ids: Sequence[str], smiles: Sequence[str]
) -> dict[str, list[str]]:
    """Group reaction IDs by direction-agnostic identity in one hashing pass."""
    groups: dict[str, list[str]] = {}
    for reaction_id, reaction_smiles in zip(reaction_ids, smiles, strict=True):
        identity = canonical_reaction_identity(reaction_smiles)
        groups.setdefault(identity, []).append(reaction_id)
    return groups


def _classify_group(
    members: Sequence[str], preimages: Mapping[str, tuple[str, str]]
) -> list[_DuplicatePair]:
    """Classify ``members[1:]`` (``members`` is sorted) against ``members[0]``."""
    canonical_id = members[0]
    canonical_sides = preimages[canonical_id]
    swapped_sides = (canonical_sides[1], canonical_sides[0])
    pairs: list[_DuplicatePair] = []
    for other_id in members[1:]:
        other_sides = preimages[other_id]
        if other_sides == canonical_sides:
            kind = EXACT_DUPLICATE
            detail = f"Reaction {other_id} is an exact duplicate of {canonical_id}"
        elif other_sides == swapped_sides:
            kind = REVERSE
            detail = f"Reaction {other_id} is the written reverse of {canonical_id}"
        else:
            # Unreachable while identity and directional preimage share one
            # canonicalization path; log loudly, still classify as duplicate.
            kind = EXACT_DUPLICATE
            logger.warning(
                "Identity group %s contains %s with an unexpected side order",
                canonical_id,
                other_id,
            )
            order_note = "unexpected side order; classified as a duplicate"
            detail = f"Reaction {other_id} shares the identity of {canonical_id} ({order_note})"
        pairs.append(
            _DuplicatePair(
                kind=kind,
                canonical_reaction_id=canonical_id,
                other_reaction_id=other_id,
                detail=detail,
            )
        )
    return pairs


def _group_records(groups: Mapping[str, Sequence[str]]) -> list[JSONValue]:
    """Return the identity-sorted group records shared by both artifacts."""
    records: list[JSONValue] = []
    for identity in sorted(groups):
        members = sorted(groups[identity])
        member_ids: list[JSONValue] = [member for member in members]
        records.append(
            {
                "identity": identity,
                "member_reaction_ids": member_ids,
                "canonical_reaction_id": members[0],
            }
        )
    return records


def _pair_records(pairs: Sequence[_DuplicatePair]) -> list[JSONValue]:
    """Return classified pairs sorted by canonical ID, other ID, then kind."""
    ordered = sorted(
        pairs, key=lambda p: (p.canonical_reaction_id, p.other_reaction_id, p.kind)
    )
    records: list[JSONValue] = []
    for pair in ordered:
        records.append(
            {
                "kind": pair.kind,
                "canonical_reaction_id": pair.canonical_reaction_id,
                "other_reaction_id": pair.other_reaction_id,
            }
        )
    return records


def detect_duplicates(
    inventory_path: str | Path, config: Mapping[str, Any]
) -> DedupResult:
    """Detect exact duplicates and written reverses across an inventory table.

    Only the ``reaction_id`` and ``reaction_smiles`` columns are read; the file
    is never modified and its rows are never removed. Rejections are APPENDED
    to the unified ledger, which is loaded first so earlier stages survive.
    Both JSON artifacts are timestamp-free, so runs over one inventory are
    byte-identical.

    Raises
    ------
    ValueError
        When the inventory table lacks a required column.
    IdentityError
        When a stored reaction SMILES cannot be canonicalized (fail loudly,
        never skip a row silently).
    """
    path = Path(inventory_path)
    reaction_ids, smiles = _read_inventory(path)
    smiles_by_id = dict(zip(reaction_ids, smiles, strict=True))
    groups = _group_by_identity(reaction_ids, smiles)

    duplicate_groups: dict[str, list[str]] = {}
    pairs: list[_DuplicatePair] = []
    for identity in sorted(groups):
        members = sorted(groups[identity])
        if len(members) < 2:
            continue
        duplicate_groups[identity] = members
        preimages = {
            member: directional_preimage(smiles_by_id[member]) for member in members
        }
        pairs.extend(_classify_group(members, preimages))
    n_exact_duplicates = sum(1 for pair in pairs if pair.kind == EXACT_DUPLICATE)
    n_reverse_pairs = sum(1 for pair in pairs if pair.kind == REVERSE)

    groups_path = Path(config["paths"]["interim"]) / DUPLICATE_GROUPS_FILENAME
    ledger_path = Path(config["paths"]["manifests"]) / DUPLICATE_LEDGER_FILENAME
    group_records = _group_records(duplicate_groups)
    version = dataset_version(config)
    write_json(
        groups_path,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset_version": version,
            "groups": group_records,
        },
    )
    write_json(
        ledger_path,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset_version": version,
            "n_inventory_rows": len(reaction_ids),
            "n_unique_identities": len(groups),
            "n_duplicate_groups": len(duplicate_groups),
            "n_exact_duplicates": n_exact_duplicates,
            "n_reverse_pairs": n_reverse_pairs,
            "groups": group_records,
            "pairs": _pair_records(pairs),
        },
    )

    ledger = RejectionLedger.load(ledger_path.parent)
    for pair in pairs:
        ledger.add(
            Rejection(
                reaction_id=pair.other_reaction_id,
                stage=DEDUP_STAGE,
                code=_CODE_BY_KIND[pair.kind],
                detail=pair.detail,
                source_pointer=f"{path}:{pair.other_reaction_id}",
            )
        )
    ledger.write()

    logger.info(
        "Dedup: rows=%d unique=%d groups=%d exact=%d reverse=%d -> %s",
        len(reaction_ids),
        len(groups),
        len(duplicate_groups),
        n_exact_duplicates,
        n_reverse_pairs,
        ledger_path,
    )
    return DedupResult(
        n_rows=len(reaction_ids),
        n_unique=len(groups),
        n_duplicate_groups=len(duplicate_groups),
        n_exact_duplicates=n_exact_duplicates,
        n_reverse_pairs=n_reverse_pairs,
        groups_path=groups_path,
        ledger_path=ledger_path,
    )


__all__ = [
    "DEDUP_STAGE",
    "DUPLICATE_GROUPS_FILENAME",
    "DUPLICATE_LEDGER_FILENAME",
    "EXACT_DUPLICATE",
    "REVERSE",
    "DedupResult",
    "detect_duplicates",
]
