"""DRFP fingerprints over the inventory for the cross-split leakage audit.

:func:`compute_fingerprints` encodes every inventory reaction SMILES in one
batched :meth:`drfp.DrfpEncoder.encode` call (never row by row) and persists the
result to ``fingerprints.parquet`` under the interim path. The split assignment
must already exist because the audit compares valid/test probes against train
references, so a missing assignment aborts with the ``g0 split`` hint before any
encoding work starts.

Canonical bit representation
----------------------------
``DrfpEncoder.encode`` returns one ``numpy.uint8`` 0/1 array of length
``near_dup.fp_size`` per reaction. The canonical persisted form is the raw 0/1
string of exactly that length, so the ``fp_bits`` column always matches
``[01]{fp_size}`` (2048 characters by default). This is the one form RDKit's
:func:`rdkit.DataStructs.CreateFromBitString` consumes directly, so the audit
rebuilds its bit vectors from the persisted strings without re-encoding.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g0.split_sources import SPLIT_ASSIGNMENT_FILENAME, SPLIT_LABELS
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

logger = logging.getLogger(__name__)

#: Fingerprint table filename written under ``config["paths"]["interim"]``.
FINGERPRINTS_PARQUET_FILENAME: Final[str] = "fingerprints.parquet"
#: DrfpEncoder substructure radius fixed by the audit protocol.
FP_RADIUS: Final[int] = 3
#: DrfpEncoder full-ring inclusion fixed by the audit protocol.
FP_RINGS: Final[bool] = True
#: Actionable hint for a missing inventory.
_INVENTORY_HINT: Final[str] = "run `g0 inventory` first"
#: Actionable hint for a missing split assignment.
_SPLIT_HINT: Final[str] = "run `g0 split` first"
#: Actionable hint for a missing fingerprint table.
_AUDIT_HINT: Final[str] = "run `g0 audit` first"
_REQUIRED_COLUMNS: Final[tuple[str, str]] = ("reaction_id", "reaction_smiles")
_FINGERPRINT_COLUMNS: Final[tuple[str, str]] = ("reaction_id", "fp_bits")


@dataclass(frozen=True, slots=True)
class FingerprintResult:
    """Outcome of one :func:`compute_fingerprints` run.

    ``fingerprints`` carries the in-memory ``(reaction_id, bit vector)`` table
    so the audit can skip the Parquet round-trip; it is excluded from ``repr``
    and ``eq`` because comparing hundreds of thousands of bit vectors is never
    intended.
    """

    n: int
    fp_size: int
    path: Path
    fingerprints: tuple[tuple[str, ExplicitBitVect], ...] = field(
        default=(), repr=False, compare=False
    )


def _missing_columns(path: Path, columns: tuple[str, str], actual: list[str]) -> None:
    """Raise when *columns* are not all present in *actual*."""
    missing = [name for name in columns if name not in actual]
    if missing:
        msg = f"{path} is missing required column(s) {missing}; found {actual}"
        raise ValueError(msg)


def _read_inventory(inventory_path: Path) -> tuple[list[str], list[str]]:
    """Return the inventory's ``(reaction_ids, reaction_smiles)`` columns."""
    if not inventory_path.is_file():
        msg = f"Missing inventory {inventory_path}; {_INVENTORY_HINT}"
        raise FileNotFoundError(msg)
    table = read_parquet(inventory_path)
    _missing_columns(inventory_path, _REQUIRED_COLUMNS, table.column_names)
    reaction_ids = [str(value) for value in table.column("reaction_id").to_pylist()]
    smiles = [str(value) for value in table.column("reaction_smiles").to_pylist()]
    return reaction_ids, smiles


def read_split_assignment(config: Mapping[str, Any]) -> dict[str, str]:
    """Return ``reaction_id -> split label`` from the assignment Parquet.

    The table is the frozen per-reaction assignment written by the split stage.
    A missing file raises :class:`FileNotFoundError` with the ``g0 split`` hint;
    a table without the two expected columns, or with a label outside
    ``train``/``valid``/``test``, raises :class:`ValueError` instead of being
    silently repaired.
    """
    path = Path(config["paths"]["interim"]) / SPLIT_ASSIGNMENT_FILENAME
    if not path.is_file():
        msg = f"Missing split assignment {path}; {_SPLIT_HINT}"
        raise FileNotFoundError(msg)
    table = read_parquet(path)
    _missing_columns(path, ("reaction_id", "split"), table.column_names)
    assignment: dict[str, str] = {}
    for reaction_id, label in zip(
        table.column("reaction_id").to_pylist(),
        table.column("split").to_pylist(),
        strict=True,
    ):
        if label not in SPLIT_LABELS:
            msg = f"Split assignment {path} carries unknown split {label!r} for {reaction_id}"
            raise ValueError(msg)
        assignment[str(reaction_id)] = str(label)
    return assignment


def _pack_bits(array: NDArray[np.uint8], fp_size: int, reaction_id: str) -> str:
    """Return *array* as the canonical 0/1 string of length *fp_size*."""
    if array.size != fp_size:
        msg = f"DRFP returned {array.size} bit(s) for {reaction_id}; expected {fp_size}"
        raise ValueError(msg)
    binary = np.asarray(array, dtype=np.uint8)
    if binary.size and int(binary.max()) > 1:
        msg = f"DRFP returned non-binary fingerprint values for {reaction_id}"
        raise ValueError(msg)
    return (binary + np.uint8(ord("0"))).tobytes().decode("ascii")


def _bit_vector(bits: str, reaction_id: str) -> ExplicitBitVect:
    """Rebuild an :class:`ExplicitBitVect` from a canonical 0/1 string."""
    vector = DataStructs.CreateFromBitString(bits)
    if vector.GetNumBits() != len(bits):
        msg = f"Fingerprint for {reaction_id} decoded to {vector.GetNumBits()} bits, not {len(bits)}"
        raise ValueError(msg)
    return vector


def compute_fingerprints(config: Mapping[str, Any]) -> FingerprintResult:
    """Encode every inventory reaction and persist ``fingerprints.parquet``.

    The split assignment is required (``g0 split`` hint when missing) and every
    assigned reaction must exist in the inventory. All SMILES go through a
    single batched :meth:`DrfpEncoder.encode` call with the configured
    ``near_dup.fp_size``, ``radius=3``, and ``rings=True``; the returned 0/1
    arrays are packed into the canonical 0/1 strings documented in the module
    docstring and written as ``[reaction_id, fp_bits]``. The in-memory bit
    vectors are returned for the audit.

    Raises
    ------
    FileNotFoundError
        When the inventory or the split assignment is missing.
    ValueError
        When a required column is absent, a stored split label is unknown, an
        assigned reaction is not in the inventory, or the inventory is empty.
    """
    try:
        from drfp import DrfpEncoder
    except ImportError as exc:
        raise ImportError(
            "DRFP is required for `g0 audit` and `g0 freeze`; install project dependencies from requirements.txt"
        ) from exc

    inventory_path = Path(config["paths"]["interim"]) / INVENTORY_PARQUET_FILENAME
    reaction_ids, smiles = _read_inventory(inventory_path)
    if not reaction_ids:
        msg = f"Inventory {inventory_path} has zero reactions; refusing to write an empty fingerprint table"
        raise ValueError(msg)
    assignment = read_split_assignment(config)
    unknown = sorted(set(assignment) - set(reaction_ids))
    if unknown:
        msg = (
            f"Split assignment references {len(unknown)} reaction(s) absent from "
            f"the inventory, e.g. {unknown[:10]}"
        )
        raise ValueError(msg)

    fp_size = int(config["near_dup"]["fp_size"])
    arrays = DrfpEncoder.encode(
        smiles, n_folded_length=fp_size, radius=FP_RADIUS, rings=FP_RINGS
    )
    if len(arrays) != len(reaction_ids):
        msg = f"DRFP encoded {len(arrays)} fingerprint(s) for {len(reaction_ids)} reaction(s)"
        raise RuntimeError(msg)
    bits = [
        _pack_bits(array, fp_size, reaction_id)
        for reaction_id, array in zip(reaction_ids, arrays, strict=True)
    ]
    path = Path(config["paths"]["interim"]) / FINGERPRINTS_PARQUET_FILENAME
    write_parquet(path, {"reaction_id": reaction_ids, "fp_bits": bits})
    fingerprints = tuple(
        (reaction_id, _bit_vector(value, reaction_id))
        for reaction_id, value in zip(reaction_ids, bits, strict=True)
    )
    logger.info(
        "Fingerprints: %d reaction(s), fp_size=%d -> %s",
        len(reaction_ids),
        fp_size,
        path,
    )
    return FingerprintResult(
        n=len(reaction_ids), fp_size=fp_size, path=path, fingerprints=fingerprints
    )


def load_fingerprints(
    config: Mapping[str, Any],
) -> tuple[tuple[str, ExplicitBitVect], ...]:
    """Rebuild the in-memory fingerprint table from ``fingerprints.parquet``.

    A missing table raises :class:`FileNotFoundError` with the ``g0 audit``
    hint. Every row must carry a ``fp_bits`` string of exactly the configured
    ``near_dup.fp_size`` consisting of 0/1 characters only; anything else is a
    hard :class:`ValueError` (a truncated fingerprint table must never be
    silently tolerated by an audit).
    """
    path = Path(config["paths"]["interim"]) / FINGERPRINTS_PARQUET_FILENAME
    if not path.is_file():
        msg = f"Missing fingerprint table {path}; {_AUDIT_HINT}"
        raise FileNotFoundError(msg)
    table = read_parquet(path)
    _missing_columns(path, _FINGERPRINT_COLUMNS, table.column_names)
    fp_size = int(config["near_dup"]["fp_size"])
    pairs: list[tuple[str, ExplicitBitVect]] = []
    for raw_id, bits in zip(
        table.column("reaction_id").to_pylist(),
        table.column("fp_bits").to_pylist(),
        strict=True,
    ):
        reaction_id = str(raw_id)
        if not isinstance(bits, str) or len(bits) != fp_size or set(bits) - {"0", "1"}:
            msg = f"Fingerprint for {reaction_id} in {path} is not a {fp_size}-bit 0/1 string"
            raise ValueError(msg)
        pairs.append((reaction_id, _bit_vector(bits, reaction_id)))
    return tuple(pairs)


def resolve_fingerprint_table(
    config: Mapping[str, Any],
    fingerprints: Sequence[tuple[str, ExplicitBitVect]] | None = None,
) -> dict[str, ExplicitBitVect]:
    """Return ``reaction_id -> bit vector`` from memory or the interim table.

    An explicit *fingerprints* sequence is used as-is; ``None`` rebuilds the
    table with :func:`load_fingerprints`. Every vector must match the
    configured ``near_dup.fp_size`` and duplicate reaction IDs are a hard
    :class:`ValueError`, so an audit can never run against an ambiguous or
    truncated table. A missing table raises :class:`FileNotFoundError` with the
    ``g0 audit`` hint.
    """
    if fingerprints is None:
        vector_pairs: Sequence[tuple[str, ExplicitBitVect]] = load_fingerprints(config)
    else:
        vector_pairs = fingerprints
    fp_size = int(config["near_dup"]["fp_size"])
    by_id: dict[str, ExplicitBitVect] = {}
    for reaction_id, vector in vector_pairs:
        if reaction_id in by_id:
            msg = f"Duplicate reaction ID {reaction_id} in fingerprint input"
            raise ValueError(msg)
        if vector.GetNumBits() != fp_size:
            msg = (
                f"Fingerprint for {reaction_id} has {vector.GetNumBits()} bits; "
                f"expected {fp_size}"
            )
            raise ValueError(msg)
        by_id[reaction_id] = vector
    return by_id


__all__ = [
    "FINGERPRINTS_PARQUET_FILENAME",
    "FP_RADIUS",
    "FP_RINGS",
    "FingerprintResult",
    "compute_fingerprints",
    "load_fingerprints",
    "read_split_assignment",
    "resolve_fingerprint_table",
]
