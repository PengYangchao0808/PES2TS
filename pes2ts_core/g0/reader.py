"""Readers for the Reaction-QM reaction-info sources.

Two source formats are implemented:

* the ``*_reaction_info.csv`` table, read with the standard-library :mod:`csv`
  module (pandas is deliberately avoided) and validated row by row: reaction
  IDs are normalized via :func:`pes2ts_core.g0.ids.normalize_reaction_id`,
  duplicate IDs are detected against earlier *accepted* rows only, and the
  reaction SMILES must contain exactly one ``>>`` with every dot-separated
  component on both sides parseable by RDKit;
* the combined ``B3LYPD3_TZVP.h5`` file, read with :mod:`h5py` over the
  bundle-group hierarchy (bundle group → ``RXN_<10-digit>`` reaction group →
  species node). Bundle and reaction names are never parsed beyond the
  ``RXN_`` prefix; species may be sub-groups or flattened compound datasets,
  and each one carries ``smiles``, ``EHG``, ``charge``, ``multiplicity``,
  ``atomic_numbers``, and ``coordinates``. Transition-state species are
  returned separately from reactants/products so callers can quarantine them.

Every HDF5 node that does not match the expected schema raises
:class:`H5SchemaError` naming the reaction and the offending dataset/species;
the CSV path rejects rows with typed
:class:`~pes2ts_core.g0.rejections.Rejection` objects instead. Neither reader
writes anything: the caller accumulates the ledger and the manifest.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

import h5py
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, RDLogger

from pes2ts_core.g0.ids import (
    REACTION_ID_PREFIX,
    BadReactionId,
    normalize_reaction_id,
)
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


# --------------------------------------------------------------------------- #
# HDF5 reader: combined B3LYP-D3/TZVP artifact
# --------------------------------------------------------------------------- #

#: Species tag of the transition state inside a reaction group.
TS_TAG: Final[str] = "TS"

#: Expected shape of the ``EHG`` energy vector (E, H, G in Hartree).
EHG_SHAPE: Final[tuple[int, ...]] = (3,)


class H5SchemaError(Exception):
    """Raised when a Reaction-QM HDF5 node does not match the expected schema.

    The message always names the reaction (or, for a malformed bundle entry,
    the offending group) and the dataset or species involved.
    """


@dataclass(frozen=True, slots=True, eq=False)
class SpeciesRecord:
    """One molecular species read from the combined Reaction-QM HDF5.

    ``coordinates`` has shape ``(n_atoms, 3)`` in Å and ``EHG`` holds the
    ``(E, H, G)`` energies in Hartree. Generated equality is disabled because
    the numpy-array fields make it ambiguous; compare fields with
    :func:`numpy.array_equal` instead.
    """

    tag: str
    smiles: str
    atomic_numbers: NDArray[np.int64]
    coordinates: NDArray[np.float64]
    charge: int
    multiplicity: int
    EHG: NDArray[np.float64]


def _reaction_id_from_name(name: str) -> str:
    """Return the normalized reaction ID embedded in an HDF5 object path.

    The last path segment starting with ``RXN_`` wins; when it cannot be
    normalized it is returned verbatim, and a node with no such segment is
    identified by its full path.
    """
    for segment in reversed(name.split("/")):
        if segment.startswith(REACTION_ID_PREFIX):
            try:
                return normalize_reaction_id(segment)
            except BadReactionId:
                return segment
    return name or "<unknown reaction>"


def _require_dataset(node: h5py.Group | h5py.Dataset, name: str, prefix: str) -> object:
    """Return the raw value of dataset/field *name* on a species node.

    A species node is either a sub-group with one dataset per field (the
    official layout, which also covers datasets reached via ``R0/smiles``
    paths) or a single flattened compound dataset with the same field names.
    """
    if isinstance(node, h5py.Group):
        dataset = node.get(name)
        if dataset is None:
            msg = f"{prefix}: missing dataset {name!r}"
            raise H5SchemaError(msg)
        return dataset
    fields = node.dtype.names
    if fields is None:
        msg = f"{prefix}: flattened species {node.name!r} is not a compound dataset"
        raise H5SchemaError(msg)
    if name not in fields:
        msg = f"{prefix}: missing dataset {name!r} in flattened species {node.name!r}"
        raise H5SchemaError(msg)
    return node[name]


def _read_smiles(value: object, prefix: str) -> str:
    """Decode a species SMILES from a string dataset, bytes field, or string."""
    raw = value
    if isinstance(raw, h5py.Dataset):
        try:
            raw = raw.asstr()[()]
        except (TypeError, ValueError) as exc:
            msg = f"{prefix}: dataset 'smiles' is not a string dataset ({exc})"
            raise H5SchemaError(msg) from exc
    if isinstance(raw, np.ndarray):
        if raw.size != 1:
            msg = f"{prefix}: dataset 'smiles' holds {raw.size} values; expected exactly one"
            raise H5SchemaError(msg)
        raw = raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            msg = f"{prefix}: dataset 'smiles' is not valid UTF-8"
            raise H5SchemaError(msg) from exc
    if isinstance(raw, str):
        return raw
    msg = f"{prefix}: dataset 'smiles' has unsupported type {type(raw).__name__}"
    raise H5SchemaError(msg)


def _read_float_array(value: object, name: str, prefix: str) -> NDArray[np.float64]:
    """Materialize *value* as a float64 array (1-D or 2-D)."""
    raw = value[()] if isinstance(value, h5py.Dataset) else value
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{prefix}: dataset {name!r} is not numeric ({exc})"
        raise H5SchemaError(msg) from exc


def _read_atomic_numbers(value: object, prefix: str) -> NDArray[np.int64]:
    """Read the 1-D integer array of atomic numbers of a species."""
    raw = value[()] if isinstance(value, h5py.Dataset) else value
    array = np.asarray(raw)
    if not np.issubdtype(array.dtype, np.integer):
        msg = f"{prefix}: dataset 'atomic_numbers' has non-integer dtype {array.dtype}"
        raise H5SchemaError(msg)
    if array.ndim != 1:
        msg = f"{prefix}: dataset 'atomic_numbers' has shape {array.shape}; expected a 1-D array"
        raise H5SchemaError(msg)
    return array.astype(np.int64)


def _read_scalar_int(value: object, name: str, prefix: str) -> int:
    """Read a single integer scalar such as ``charge`` or ``multiplicity``."""
    raw = value[()] if isinstance(value, h5py.Dataset) else value
    array = np.asarray(raw)
    if array.size != 1:
        msg = f"{prefix}: dataset {name!r} holds {array.size} values; expected exactly one"
        raise H5SchemaError(msg)
    scalar = array.reshape(-1)[0]
    if not isinstance(scalar, int | np.integer):
        msg = f"{prefix}: dataset {name!r} is not an integer ({scalar!r})"
        raise H5SchemaError(msg)
    return int(scalar)


def _read_species(
    node: h5py.Group | h5py.Dataset, *, tag: str, reaction_id: str
) -> SpeciesRecord:
    """Read and validate one species node (sub-group or flattened dataset)."""
    prefix = f"Reaction {reaction_id}, species {tag!r}"
    smiles = _read_smiles(_require_dataset(node, "smiles", prefix), prefix)
    EHG = _read_float_array(_require_dataset(node, "EHG", prefix), "EHG", prefix)
    if EHG.shape != EHG_SHAPE:
        msg = f"{prefix}: dataset 'EHG' has shape {EHG.shape}; expected {EHG_SHAPE}"
        raise H5SchemaError(msg)
    charge = _read_scalar_int(_require_dataset(node, "charge", prefix), "charge", prefix)
    multiplicity = _read_scalar_int(
        _require_dataset(node, "multiplicity", prefix), "multiplicity", prefix
    )
    atomic_numbers = _read_atomic_numbers(
        _require_dataset(node, "atomic_numbers", prefix), prefix
    )
    coordinates = _read_float_array(
        _require_dataset(node, "coordinates", prefix), "coordinates", prefix
    )
    expected_coordinates = (atomic_numbers.shape[0], 3)
    if coordinates.shape != expected_coordinates:
        msg = (
            f"{prefix}: dataset 'coordinates' has shape {coordinates.shape}; "
            f"expected {expected_coordinates}"
        )
        raise H5SchemaError(msg)
    return SpeciesRecord(
        tag=tag,
        smiles=smiles,
        atomic_numbers=atomic_numbers,
        coordinates=coordinates,
        charge=charge,
        multiplicity=multiplicity,
        EHG=EHG,
    )


def read_h5_species(species_group: h5py.Group | h5py.Dataset) -> SpeciesRecord:
    """Read one species node of the combined Reaction-QM HDF5.

    *species_group* may be a species sub-group (the official layout) or a
    flattened compound dataset carrying the same fields; the last path
    segment supplies the tag and the reaction ID is derived from the HDF5
    path. Required datasets are ``smiles``, ``EHG``, ``charge``,
    ``multiplicity``, ``atomic_numbers``, and ``coordinates`` (see
    :class:`SpeciesRecord`).

    Raises
    ------
    H5SchemaError
        When a required dataset is missing, ``smiles`` cannot be decoded, or
        the shapes violate ``coordinates.shape == (len(atomic_numbers), 3)``
        or ``EHG.shape == (3,)``.
    """
    name = species_group.name
    tag = name.rsplit("/", 1)[-1]
    return _read_species(species_group, tag=tag, reaction_id=_reaction_id_from_name(name))


def read_h5_reaction(
    group: h5py.Group | h5py.Dataset,
) -> tuple[list[SpeciesRecord], list[SpeciesRecord]]:
    """Read one ``RXN_<10-digit>`` reaction node.

    Returns ``(rp_species, ts_species)``: species tagged ``R<n>``/``P<n>`` in
    the first list, the ``TS`` species in the second, each sorted by tag. The
    component count is discovered from the node (a bimolecular reaction simply
    carries ``R1``/``P1``), never assumed.

    Raises
    ------
    H5SchemaError
        When *group* is not an HDF5 group, a child tag is neither ``R<n>``,
        ``P<n>``, nor ``TS``, or any species violates the species schema.
    """
    if not isinstance(group, h5py.Group):
        identifier = _reaction_id_from_name(group.name)
        msg = (
            f"Reaction {identifier}: expected a reaction group, got "
            f"{type(group).__name__} {group.name!r}"
        )
        raise H5SchemaError(msg)
    reaction_id = _reaction_id_from_name(group.name)
    rp_species: list[SpeciesRecord] = []
    ts_species: list[SpeciesRecord] = []
    for tag, node in group.items():
        if tag == TS_TAG:
            ts_species.append(_read_species(node, tag=tag, reaction_id=reaction_id))
        elif tag.startswith(("R", "P")):
            rp_species.append(_read_species(node, tag=tag, reaction_id=reaction_id))
        else:
            msg = (
                f"Reaction {reaction_id}: unknown species tag {tag!r}; "
                f"expected 'R<n>', 'P<n>', or {TS_TAG!r}"
            )
            raise H5SchemaError(msg)
    rp_species.sort(key=lambda record: record.tag)
    ts_species.sort(key=lambda record: record.tag)
    return rp_species, ts_species


def iter_h5_reactions(
    path: str | Path,
) -> Iterator[tuple[str, list[SpeciesRecord], list[SpeciesRecord]]]:
    """Iterate the combined Reaction-QM HDF5 one reaction at a time.

    The file is opened read-only inside the generator and never mutated. Every
    top-level group is treated as a bundle and every ``RXN_``-prefixed child
    as a reaction; bundles are consumed in HDF5 (name) order and reactions in
    sorted key order. Non-``RXN_`` children (bundle metadata) are skipped with
    a debug log; this leniency stops at the reaction level, where any species
    problem raises.

    Yields
    ------
    tuple[str, list[SpeciesRecord], list[SpeciesRecord]]
        ``(normalized_reaction_id, rp_species, ts_species)`` per reaction.

    Raises
    ------
    H5SchemaError
        When an ``RXN_``-prefixed key cannot be normalized, the same reaction
        ID occurs in two bundles (naming both bundles), or a reaction/species
        violates the schema.
    """
    seen_bundles: dict[str, str] = {}
    with h5py.File(path, "r") as handle:
        for bundle_name, bundle_node in handle.items():
            if not isinstance(bundle_node, h5py.Group):
                logger.debug(
                    "Skipping non-group top-level node %r in %s", bundle_name, path
                )
                continue
            for reaction_key in sorted(bundle_node.keys()):
                if not reaction_key.startswith(REACTION_ID_PREFIX):
                    logger.debug(
                        "Skipping non-reaction key %r in bundle %r",
                        reaction_key,
                        bundle_name,
                    )
                    continue
                try:
                    reaction_id = normalize_reaction_id(reaction_key)
                except BadReactionId as exc:
                    msg = (
                        f"Bundle {bundle_name!r}: reaction group {reaction_key!r} "
                        f"has an invalid reaction id: {exc}"
                    )
                    raise H5SchemaError(msg) from exc
                previous_bundle = seen_bundles.get(reaction_id)
                if previous_bundle is not None:
                    msg = (
                        f"Reaction id {reaction_id} appears in bundles "
                        f"{previous_bundle!r} and {bundle_name!r}"
                    )
                    raise H5SchemaError(msg)
                seen_bundles[reaction_id] = bundle_name
                rp_species, ts_species = read_h5_reaction(bundle_node[reaction_key])
                yield reaction_id, rp_species, ts_species


__all__ = [
    "ARROW",
    "CsvSchemaError",
    "EHG_SHAPE",
    "ENERGY_COLUMNS",
    "H5SchemaError",
    "ID_COLUMN",
    "READ_STAGE",
    "SMILES_COLUMNS",
    "TS_TAG",
    "ReactionInfoRecord",
    "SpeciesRecord",
    "iter_h5_reactions",
    "read_h5_reaction",
    "read_h5_species",
    "read_reaction_info_csv",
]
