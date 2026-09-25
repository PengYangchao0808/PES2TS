"""Discovery and schema validation of the authors' official split CSVs.

The three official split files are located by suffix from
``config["source"]["files"]`` and parsed into a ``reaction_id -> label``
assignment map, which :func:`~pes2ts_core.g0.split.adopt_official_split` then
validates against the inventory.  The split files are authoritative inputs: a
missing reaction-id column, a malformed ID, a duplicate assignment within one
file, or an ID assigned to two different splits raises
:class:`SplitSchemaError` instead of being silently repaired.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0.ids import BadReactionId, normalize_reaction_id
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.utils.parquet_io import read_parquet

logger = logging.getLogger(__name__)

#: Actionable hint attached to a missing split-source error.
FETCH_HINT: Final[str] = "run `g0 fetch` first"
#: Actionable hint attached to a missing inventory error.
INVENTORY_HINT: Final[str] = "run `g0 inventory` first"

#: Candidate reaction-id columns, in priority order (first present wins).
ID_COLUMN_CANDIDATES: Final[tuple[str, ...]] = ("reaction_id", "rxn_id", "id")
#: Split labels in fixed processing order; also the manifest/Parquet values.
SPLIT_LABELS: Final[tuple[str, ...]] = ("train", "valid", "test")
#: Assignment table filename written under ``config["paths"]["interim"]``.
SPLIT_ASSIGNMENT_FILENAME: Final[str] = "split_assignment.parquet"
#: Split manifest filename written under ``config["paths"]["manifests"]``.
SPLIT_MANIFEST_FILENAME: Final[str] = "split_manifest.json"
#: Configured-source suffix per split label.
SPLIT_SUFFIXES: Final[dict[str, str]] = {
    "train": "_train.csv",
    "valid": "_valid.csv",
    "test": "_test.csv",
}


class SplitSchemaError(ValueError):
    """Raised when the authoritative split inputs violate the expected schema.

    Unlike a row-level reader rejection, a schema violation here would silently
    change the frozen evaluation split, so the stage aborts instead of repairing
    or skipping anything.
    """


def _configured_filenames(config: Mapping[str, Any]) -> list[str]:
    """Return every configured source filename, honoring the key fallback."""
    return [
        str(entry.get("filename", key)) for key, entry in config["source"]["files"].items()
    ]


def resolve_split_files(config: Mapping[str, Any]) -> dict[str, tuple[str, Path]]:
    """Return ``label -> (filename, path)`` for the three official split CSVs.

    Each label's file is discovered by suffix; zero or multiple matches for a
    suffix raise :class:`ValueError` listing the configured names, and a missing
    file raises :class:`FileNotFoundError` with the ``g0 fetch`` hint.
    """
    filenames = _configured_filenames(config)
    raw_dir = Path(config["paths"]["raw"])
    resolved: dict[str, tuple[str, Path]] = {}
    for label in SPLIT_LABELS:
        suffix = SPLIT_SUFFIXES[label]
        matches = [name for name in filenames if name.endswith(suffix)]
        if len(matches) != 1:
            msg = (
                f"Expected exactly one configured source file ending in {suffix!r}, "
                f"found {matches}"
            )
            raise ValueError(msg)
        filename = matches[0]
        path = raw_dir / filename
        if not path.is_file():
            msg = f"Missing split CSV for {label!r} ({path}); {FETCH_HINT}"
            raise FileNotFoundError(msg)
        resolved[label] = (filename, path)
    return resolved


def _read_split_csv(path: Path, label: str) -> list[str]:
    """Return the normalized reaction IDs *path* assigns to *label*.

    Extra columns (for example SMILES) are ignored.  A missing reaction-id
    column, a malformed ID, or an ID repeated within this file raises
    :class:`SplitSchemaError` naming the file, line, and value.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        id_column = next(
            (name for name in ID_COLUMN_CANDIDATES if name in columns), None
        )
        if id_column is None:
            msg = f"Split CSV {path} has no reaction-id column; found: {columns}"
            raise SplitSchemaError(msg)
        seen: set[str] = set()
        reaction_ids: list[str] = []
        for row in reader:
            raw_value = row.get(id_column)
            try:
                reaction_id = normalize_reaction_id(raw_value)
            except BadReactionId as exc:
                msg = (
                    f"Split CSV {path} line {reader.line_num} ({label}) carries an "
                    f"invalid reaction id {raw_value!r}: {exc}"
                )
                raise SplitSchemaError(msg) from exc
            if reaction_id in seen:
                msg = f"Split CSV {path} assigns {reaction_id} more than once ({label})"
                raise SplitSchemaError(msg)
            seen.add(reaction_id)
            reaction_ids.append(reaction_id)
    return reaction_ids


def build_assignment(files: Mapping[str, tuple[str, Path]]) -> dict[str, str]:
    """Return ``reaction_id -> label``, enforcing cross-file disjointness.

    An ID assigned to two different split files raises
    :class:`SplitSchemaError` naming the ID and both files.
    """
    assignment: dict[str, str] = {}
    for label in SPLIT_LABELS:
        filename, path = files[label]
        for reaction_id in _read_split_csv(path, label):
            previous = assignment.get(reaction_id)
            if previous is not None:
                previous_filename, _ = files[previous]
                msg = (
                    f"Reaction {reaction_id} is assigned to both {previous!r} "
                    f"({previous_filename}) and {label!r} ({filename})"
                )
                raise SplitSchemaError(msg)
            assignment[reaction_id] = label
    return assignment


def read_inventory_ids(config: Mapping[str, Any]) -> tuple[Path, list[str]]:
    """Return ``(inventory_path, reaction_ids)`` from the interim inventory.

    A missing inventory raises :class:`FileNotFoundError` with the
    ``g0 inventory`` hint; a table without the ID column is a hard error.
    """
    path = Path(config["paths"]["interim"]) / INVENTORY_PARQUET_FILENAME
    if not path.is_file():
        msg = f"Missing inventory {path}; {INVENTORY_HINT}"
        raise FileNotFoundError(msg)
    table = read_parquet(path)
    if "reaction_id" not in table.column_names:
        msg = f"Inventory {path} has no reaction_id column; found: {table.column_names}"
        raise ValueError(msg)
    reaction_ids: list[str] = table.column("reaction_id").to_pylist()
    return path, reaction_ids


__all__ = [
    "FETCH_HINT",
    "ID_COLUMN_CANDIDATES",
    "INVENTORY_HINT",
    "SPLIT_ASSIGNMENT_FILENAME",
    "SPLIT_LABELS",
    "SPLIT_MANIFEST_FILENAME",
    "SPLIT_SUFFIXES",
    "SplitSchemaError",
    "build_assignment",
    "read_inventory_ids",
    "resolve_split_files",
]
