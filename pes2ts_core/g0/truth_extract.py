"""Pure extraction helpers for the ground-truth quarantine.

These functions turn the combined Reaction-QM HDF5 into the quarantined
transition-state table and the shape-only IRC index.  They take explicit paths
and never decide where an artifact lives: the quarantine orchestrator owns
paths, relocation, and the manifest, so this module stays a pure
source-to-rows transformation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0.irc_reader import index_irc_reaction, iter_irc_reactions
from pes2ts_core.g0.reader import (
    H5SchemaError,
    iter_h5_reactions,
    read_reaction_info_csv,
)

#: Column contract of the extracted TS table.
TS_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id",
    "atomic_numbers",
    "coordinates",
    "EHG",
    "charge",
    "multiplicity",
    "reaction_smiles",
)
#: Column contract of the shape-only IRC index.
IRC_INDEX_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id",
    "n_atoms",
    "n_frames",
    "has_forces",
    "ts_index",
)


def reaction_smiles_map(info_path: Path | None) -> dict[str, str]:
    """Return ``reaction_id -> reaction_smiles`` from the reaction-info CSV.

    A missing CSV yields an empty mapping (callers fall back to an empty
    ``reaction_smiles``), and invalid rows are ignored: they are the inventory
    stage's concern, not the quarantine's.
    """
    if info_path is None or not info_path.is_file():
        return {}
    records, _rejections = read_reaction_info_csv(info_path)
    return {record["reaction_id"]: record["reaction_smiles"] for record in records}


def extract_ts_rows(
    main_h5: Path, reaction_smiles_by_id: Mapping[str, str]
) -> tuple[list[dict[str, Any]], int]:
    """Extract one row per ``TS`` species from the combined HDF5.

    Returns the rows (sorted by reaction ID) and the number of reactions that
    carry no TS species; a reaction with more than one TS species raises
    :class:`H5SchemaError`.
    """
    rows: list[dict[str, Any]] = []
    n_reactions_without_ts = 0
    for reaction_id, _rp_species, ts_species in iter_h5_reactions(main_h5):
        if not ts_species:
            n_reactions_without_ts += 1
            continue
        if len(ts_species) > 1:
            msg = (
                f"Reaction {reaction_id}: {len(ts_species)} TS species found; "
                "expected exactly one"
            )
            raise H5SchemaError(msg)
        species = ts_species[0]
        rows.append(
            {
                "reaction_id": reaction_id,
                "atomic_numbers": [int(value) for value in species.atomic_numbers],
                "coordinates": [
                    [float(component) for component in atom]
                    for atom in species.coordinates
                ],
                "EHG": [float(value) for value in species.EHG],
                "charge": int(species.charge),
                "multiplicity": int(species.multiplicity),
                "reaction_smiles": reaction_smiles_by_id.get(reaction_id, ""),
            }
        )
    rows.sort(key=lambda row: row["reaction_id"])
    return rows, n_reactions_without_ts


def build_irc_index_rows(irc_h5: Path) -> list[dict[str, Any]]:
    """Build the shape-only IRC index rows (no trajectory frames copied)."""
    rows: list[dict[str, Any]] = []
    for reaction_id, group in iter_irc_reactions(irc_h5):
        entry = index_irc_reaction(reaction_id, group)
        rows.append(
            {
                "reaction_id": entry.reaction_id,
                "n_atoms": entry.n_atoms,
                "n_frames": entry.n_frames,
                "has_forces": entry.has_forces,
                "ts_index": 0,
            }
        )
    rows.sort(key=lambda row: row["reaction_id"])
    return rows


def rows_to_columns(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> dict[str, list[Any]]:
    """Transpose *rows* into the column mapping expected by ``write_parquet``."""
    return {column: [row[column] for row in rows] for column in columns}


__all__ = [
    "IRC_INDEX_COLUMNS",
    "TS_COLUMNS",
    "build_irc_index_rows",
    "extract_ts_rows",
    "reaction_smiles_map",
    "rows_to_columns",
]
