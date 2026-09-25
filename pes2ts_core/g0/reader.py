"""Readers for the Reaction-QM reaction-info sources.

This module currently implements the CSV path, reading the
``*_reaction_info.csv`` table with the standard-library :mod:`csv` module
(pandas is deliberately avoided) and applying row-level validation:

* reaction IDs are normalized via
  :func:`pes2ts_core.g0.ids.normalize_reaction_id`;
* duplicate IDs are detected against earlier *accepted* rows only;
* the reaction SMILES must contain exactly one ``>>`` and every dot-separated
  component on both sides must parse with RDKit.

Every row that does not become a record produces exactly one typed
:class:`~pes2ts_core.g0.rejections.Rejection`; the caller accumulates the
ledger, this module never writes it.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final, TypedDict

from rdkit import Chem, RDLogger

from pes2ts_core.g0.ids import BadReactionId, normalize_reaction_id
from pes2ts_core.g0.rejections import Rejection, RejectionCode

logger = logging.getLogger(__name__)

#: RDKit reports parse failures on stderr by default; they are surfaced as
#: typed rejections instead, so the C++ logger is muted at import time.
#: (rdkit-stubs omits ``DisableLog``, hence the targeted ignore.)
RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]

#: Stage name stamped into every rejection emitted by this module.
READ_STAGE: Final[str] = "read_reaction_info_csv"

#: Separator between the reactant and product sides of a reaction SMILES.
ARROW: Final[str] = ">>"

#: Exact name of the reaction identifier column in the source table.
ID_COLUMN: Final[str] = "reaction_id"

#: Accepted reaction-SMILES column names, in priority order (first present wins).
SMILES_COLUMNS: Final[tuple[str, ...]] = ("reaction_smiles", "SMILES", "smiles")

#: Energy columns copied into every record when present (else ``None``), floats.
ENERGY_COLUMNS: Final[tuple[str, ...]] = (
    "dE",
    "dE_dagger",
    "dH",
    "dH_dagger",
    "dG",
    "dG_dagger",
)


class CsvSchemaError(Exception):
    """Raised when the reaction-info CSV does not match the expected schema."""


class ReactionInfoRecord(TypedDict):
    """One accepted row of the reaction-info CSV."""

    reaction_id: str
    reaction_smiles: str
    dE: float | None
    dE_dagger: float | None
    dH: float | None
    dH_dagger: float | None
    dG: float | None
    dG_dagger: float | None


def _require_column(header: Sequence[str], column: str) -> None:
    """Raise :class:`CsvSchemaError` when *column* is absent from *header*."""
    if column not in header:
        msg = f"Missing {column} column; found: {header}"
        raise CsvSchemaError(msg)


def _resolve_smiles_column(header: Sequence[str]) -> str:
    """Return the first accepted reaction-SMILES column name present."""
    for candidate in SMILES_COLUMNS:
        if candidate in header:
            return candidate
    accepted = ", ".join(SMILES_COLUMNS)
    msg = f"Missing reaction SMILES column (accepted names: {accepted}); found: {header}"
    raise CsvSchemaError(msg)


def _parse_energy(value: str | None) -> float | None:
    """Coerce a present energy cell to float; missing or blank cells become ``None``.

    A non-numeric cell is logged and treated as missing rather than aborting the
    read of the whole table; the energy columns are auxiliary annotations, not
    part of the reaction identity.
    """
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring non-numeric energy value %r", value)
        return None


def _iter_rows(
    reader: csv.DictReader[str],
    first_row: dict[str | None, str | None] | None,
) -> Iterator[tuple[dict[str | None, str | None], int]]:
    """Yield ``(row, physical_csv_line_number)`` including the peeked row.

    The line number is the physical position in the file (the header is line
    1), so a rejection's ``source_pointer`` can be opened directly.
    """
    if first_row is not None:
        yield first_row, reader.line_num
    for row in reader:
        yield row, reader.line_num


def _validate_reaction_smiles(
    reaction_smiles: str,
    *,
    reaction_id: str,
    source_pointer: str,
) -> Rejection | None:
    """Return the typed rejection for invalid *reaction_smiles*, else ``None``.

    Validation order: exactly one ``>>`` (``NO_ARROW``), no empty side or
    component (``BAD_SMILES``), and every component parseable by RDKit naming
    the first unparseable component (``UNPARSEABLE_MAPPED``).
    """

    def reject(code: RejectionCode, detail: str) -> Rejection:
        return Rejection(
            reaction_id=reaction_id,
            stage=READ_STAGE,
            code=code,
            detail=detail,
            source_pointer=source_pointer,
        )

    arrow_count = reaction_smiles.count(ARROW)
    if arrow_count != 1:
        detail = (
            f"Expected exactly one {ARROW!r} in reaction SMILES "
            f"{reaction_smiles!r}, found {arrow_count}"
        )
        return reject(RejectionCode.NO_ARROW, detail)
    reactant_side, product_side = reaction_smiles.split(ARROW)
    for label, side in (("reactant", reactant_side), ("product", product_side)):
        if not side:
            detail = f"Empty {label} side in reaction SMILES {reaction_smiles!r}"
            return reject(RejectionCode.BAD_SMILES, detail)
        for component in side.split("."):
            if not component:
                detail = (
                    f"Empty {label} component in reaction SMILES {reaction_smiles!r}"
                )
                return reject(RejectionCode.BAD_SMILES, detail)
            # rdkit-stubs types MolFromSmiles as -> Mol, but it returns None on
            # parse failure; the comparison is the documented runtime contract.
            if Chem.MolFromSmiles(component) is None:  # pyright: ignore[reportUnnecessaryComparison]
                detail = (
                    f"RDKit cannot parse {label} component {component!r} of "
                    f"reaction SMILES {reaction_smiles!r}"
                )
                return reject(RejectionCode.UNPARSEABLE_MAPPED, detail)
    return None


def _build_record(
    reaction_id: str,
    reaction_smiles: str,
    row: dict[str | None, str | None],
) -> ReactionInfoRecord:
    """Assemble one accepted record, coercing present energy columns to float."""
    return {
        "reaction_id": reaction_id,
        "reaction_smiles": reaction_smiles,
        "dE": _parse_energy(row.get("dE")),
        "dE_dagger": _parse_energy(row.get("dE_dagger")),
        "dH": _parse_energy(row.get("dH")),
        "dH_dagger": _parse_energy(row.get("dH_dagger")),
        "dG": _parse_energy(row.get("dG")),
        "dG_dagger": _parse_energy(row.get("dG_dagger")),
    }


def read_reaction_info_csv(
    path: str | Path,
) -> tuple[list[ReactionInfoRecord], list[Rejection]]:
    """Read the reaction-info CSV at *path* into records and rejections.

    The header must contain a ``reaction_id`` column and one of the accepted
    reaction-SMILES column names (:data:`SMILES_COLUMNS`); energy columns are
    optional. The file may legally be header-only, in which case both returned
    lists are empty.

    Rows are validated in order and skipped with exactly one typed rejection
    when the ID is invalid (``BAD_ID``), the ID repeats an earlier accepted row
    (``DUPLICATE_ID``), the SMILES does not contain exactly one ``>>``
    (``NO_ARROW``), or a side/component is empty or unparseable by RDKit
    (``BAD_SMILES`` / ``UNPARSEABLE_MAPPED``).

    Returns
    -------
    tuple[list[ReactionInfoRecord], list[Rejection]]
        Accepted records in file order and the rejections in file order.

    Raises
    ------
    CsvSchemaError
        When the header has no reaction-ID column or no accepted SMILES column;
        the message lists the actual columns found.
    """
    csv_path = Path(path)
    records: list[ReactionInfoRecord] = []
    rejections: list[Rejection] = []
    accepted_rownums: dict[str, int] = {}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)
        header = reader.fieldnames
        if header is None:
            # Zero-byte file: there is no header to inspect.
            msg = f"Missing {ID_COLUMN} column; found: []"
            raise CsvSchemaError(msg)
        _require_column(header, ID_COLUMN)
        smiles_column = _resolve_smiles_column(header)

        for row, rownum in _iter_rows(reader, first_row):
            source_pointer = f"{csv_path}:{rownum}"
            raw_id = row.get(ID_COLUMN) or ""
            try:
                reaction_id = normalize_reaction_id(raw_id)
            except BadReactionId as exc:
                rejections.append(
                    Rejection(
                        reaction_id=raw_id,
                        stage=READ_STAGE,
                        code=RejectionCode.BAD_ID,
                        detail=str(exc),
                        source_pointer=source_pointer,
                    )
                )
                continue

            first_rownum = accepted_rownums.get(reaction_id)
            if first_rownum is not None:
                rejections.append(
                    Rejection(
                        reaction_id=reaction_id,
                        stage=READ_STAGE,
                        code=RejectionCode.DUPLICATE_ID,
                        detail=(
                            f"Reaction id {reaction_id} was already accepted from "
                            f"{csv_path}:{first_rownum}"
                        ),
                        source_pointer=source_pointer,
                    )
                )
                continue

            raw_smiles = row.get(smiles_column) or ""
            rejection = _validate_reaction_smiles(
                raw_smiles, reaction_id=reaction_id, source_pointer=source_pointer
            )
            if rejection is not None:
                rejections.append(rejection)
                continue

            accepted_rownums[reaction_id] = rownum
            records.append(_build_record(reaction_id, raw_smiles, row))

    logger.info(
        "Read %d accepted record(s) and %d rejection(s) from %s",
        len(records),
        len(rejections),
        csv_path,
    )
    return records, rejections


__all__ = [
    "ARROW",
    "CsvSchemaError",
    "ENERGY_COLUMNS",
    "ID_COLUMN",
    "READ_STAGE",
    "SMILES_COLUMNS",
    "ReactionInfoRecord",
    "read_reaction_info_csv",
]
