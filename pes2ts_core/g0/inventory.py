"""TS-free reactant/product inventory over the verified Reaction-QM sources.

:func:`build_inventory` joins the reaction-info CSV with the combined HDF5
reactant/product species on the normalized reaction ID, validates every joined
reaction with :func:`~pes2ts_core.g0.rp_checks.validate_rp_reaction`, and
writes one Parquet row per surviving reaction plus the inventory manifest and
the unified rejection ledger.

The inventory is the G0 gate that keeps transition-state data out of every
downstream path-generation stage: ``TS`` species are never copied into a row,
and no column or component field carries TS coordinates or energies.  A
reaction is skipped only through a typed rejection (charge, spin, atom-count,
element, or atom-map mismatch, or a missing join partner), so nothing leaves
the pipeline silently.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pyarrow as pa

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.g0.fetch import dataset_version, verify_sources
from pes2ts_core.g0.reader import (
    ReactionInfoRecord,
    SpeciesRecord,
    iter_h5_reactions,
    read_reaction_info_csv,
)
from pes2ts_core.g0.rejections import Rejection, RejectionCode, RejectionLedger
from pes2ts_core.g0.rp_checks import element_symbols, validate_rp_reaction
from pes2ts_core.utils.hashing import JSONValue
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.utils.parquet_io import write_parquet

logger = logging.getLogger(__name__)

#: Inventory table filename written under ``config["paths"]["interim"]``.
INVENTORY_PARQUET_FILENAME: Final[str] = "inventory.parquet"
#: Inventory manifest filename written under ``config["paths"]["manifests"]``.
INVENTORY_MANIFEST_FILENAME: Final[str] = "inventory_manifest.json"
#: Stage name stamped into every rejection emitted by this module.
INVENTORY_STAGE: Final[str] = "build_inventory"
#: Suffix identifying the reaction-info table among the configured sources.
CSV_SOURCE_SUFFIX: Final[str] = "_reaction_info.csv"
#: Suffix identifying the main combined archive among the configured sources.
H5_SOURCE_SUFFIX: Final[str] = "_TZVP.h5"
#: Suffix identifying ground-truth path data, which is never inventory input.
IRC_SOURCE_SUFFIX: Final[str] = "_IRC.h5"
#: Actionable hint attached to a missing-source error.
FETCH_HINT: Final[str] = "run `g0 fetch` first"

#: Inventory columns; every row mapping carries exactly these keys.
ROW_FIELDS: Final[tuple[str, ...]] = (
    "reaction_id", "dataset_version", "reaction_smiles",
    "n_reactant_components", "n_product_components", "elements",
    "total_atoms_reactants", "total_atoms_products",
    "charge_total_reactants", "charge_total_products",
    "multiplicity_max", "components",
)


@dataclass(frozen=True, slots=True)
class InventoryResult:
    """Outcome of one :func:`build_inventory` run."""

    n_rows: int
    n_rejections: int
    manifest_path: Path
    parquet_path: Path


def _unique_source(names: Sequence[str], *, suffix: str) -> str:
    """Return the single configured filename ending in *suffix*."""
    if len(names) == 1:
        return names[0]
    msg = (
        f"Expected exactly one configured source file ending in {suffix!r}, "
        f"found {list(names)}"
    )
    raise ValueError(msg)


def _resolve_source_filenames(config: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(csv_filename, h5_filename)`` declared under ``source.files``.

    Both names are discovered by suffix, never hard-coded, and every ``_IRC``
    entry is excluded because it is ground-truth path data rather than
    inventory input.
    """
    filenames = [
        str(entry.get("filename", key)) for key, entry in config["source"]["files"].items()
    ]
    csv_names = [name for name in filenames if name.endswith(CSV_SOURCE_SUFFIX)]
    h5_names = [
        name
        for name in filenames
        if name.endswith(H5_SOURCE_SUFFIX) and not name.endswith(IRC_SOURCE_SUFFIX)
    ]
    return (
        _unique_source(csv_names, suffix=CSV_SOURCE_SUFFIX),
        _unique_source(h5_names, suffix=H5_SOURCE_SUFFIX),
    )


def _component_record(species: SpeciesRecord) -> dict[str, object]:
    """Return the nested per-component record embedded in an inventory row."""
    return {
        "tag": species.tag,
        "smiles": species.smiles,
        "atomic_numbers": species.atomic_numbers.tolist(),
        "coordinates": species.coordinates.tolist(),
        "charge": species.charge,
        "multiplicity": species.multiplicity,
        "EHG": species.EHG.tolist(),
    }


def _build_row(
    record: ReactionInfoRecord,
    species: Sequence[SpeciesRecord],
    dataset_version_value: str,
) -> dict[str, object]:
    """Assemble one inventory row from an already validated reaction.

    The HDF5 reader returns TS species through a separate channel that this
    module discards, so only R/P data can reach the row.
    """
    reactants = [item for item in species if item.tag.startswith("R")]
    products = [item for item in species if item.tag.startswith("P")]
    reactant_atoms = [
        number for item in reactants for number in item.atomic_numbers.tolist()
    ]
    product_atoms = [
        number for item in products for number in item.atomic_numbers.tolist()
    ]
    return {
        "reaction_id": record["reaction_id"],
        "dataset_version": dataset_version_value,
        "reaction_smiles": record["reaction_smiles"],
        "n_reactant_components": len(reactants),
        "n_product_components": len(products),
        "elements": element_symbols(reactant_atoms),
        "total_atoms_reactants": len(reactant_atoms),
        "total_atoms_products": len(product_atoms),
        "charge_total_reactants": sum(item.charge for item in reactants),
        "charge_total_products": sum(item.charge for item in products),
        "multiplicity_max": max((item.multiplicity for item in species), default=1),
        "components": [_component_record(item) for item in species],
    }


def build_inventory(config: Mapping[str, Any]) -> InventoryResult:
    """Build the TS-free reactant/product inventory for *config*.

    Sources are resolved from ``config["source"]["files"]`` and read from
    ``config["paths"]["raw"]``; a missing main HDF5 or reaction-info CSV raises
    :class:`FileNotFoundError` (run ``g0 fetch``), and a checksum mismatch
    propagates from :func:`~pes2ts_core.g0.fetch.verify_sources`.

    Every CSV record without an HDF5 reaction is rejected as ``MISSING_IN_H5``
    and every HDF5 reaction without a CSV record as ``MISSING_IN_CSV``; joined
    reactions failing the R/P consistency checks are rejected with their typed
    code.  Surviving reactions become one Parquet row each (nested component
    records included), all rejections accumulate in one ledger, and the
    manifest plus ledger are written atomically, so re-running overwrites.
    """
    csv_filename, h5_filename = _resolve_source_filenames(config)
    raw_dir = Path(config["paths"]["raw"])
    csv_path = raw_dir / csv_filename
    h5_path = raw_dir / h5_filename
    for label, path in (("reaction-info CSV", csv_path), ("combined HDF5", h5_path)):
        if not path.is_file():
            msg = f"Missing {label} {path}; {FETCH_HINT}"
            raise FileNotFoundError(msg)
    verify_sources(config)

    version = dataset_version(config)
    csv_records, rejections = read_reaction_info_csv(csv_path)
    records_by_id = {record["reaction_id"]: record for record in csv_records}
    seen_in_h5: set[str] = set()
    rows: list[dict[str, object]] = []
    n_reactions_in_h5 = 0
    for reaction_id, rp_species, _ts_species in iter_h5_reactions(h5_path):
        n_reactions_in_h5 += 1
        seen_in_h5.add(reaction_id)
        record = records_by_id.get(reaction_id)
        if record is None:
            rejections.append(
                Rejection(
                    reaction_id=reaction_id,
                    stage=INVENTORY_STAGE,
                    code=RejectionCode.MISSING_IN_CSV,
                    detail=(
                        "Reaction present in the combined HDF5 but not in "
                        "the reaction-info CSV"
                    ),
                    source_pointer=f"{h5_path}:{reaction_id}",
                )
            )
            continue
        failure = validate_rp_reaction(record, rp_species)
        if failure is not None:
            code, detail = failure
            rejections.append(
                Rejection(
                    reaction_id=reaction_id,
                    stage=INVENTORY_STAGE,
                    code=code,
                    detail=detail,
                    source_pointer=f"{h5_path}:{reaction_id}",
                )
            )
            continue
        rows.append(_build_row(record, rp_species, version))

    for reaction_id in records_by_id:
        if reaction_id not in seen_in_h5:
            rejections.append(
                Rejection(
                    reaction_id=reaction_id,
                    stage=INVENTORY_STAGE,
                    code=RejectionCode.MISSING_IN_H5,
                    detail=(
                        "Reaction present in the reaction-info CSV but not in "
                        "the combined HDF5"
                    ),
                    source_pointer=f"{csv_path}:{reaction_id}",
                )
            )

    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
    parquet_path = interim_dir / INVENTORY_PARQUET_FILENAME
    manifest_path = manifests_dir / INVENTORY_MANIFEST_FILENAME

    table = pa.table({field: [row[field] for row in rows] for field in ROW_FIELDS})
    write_parquet(parquet_path, table)

    ledger = RejectionLedger(manifests_dir)
    for rejection in rejections:
        ledger.add(rejection)
    ledger.write()

    histogram = Counter(rejection.code.value for rejection in rejections)
    manifest: dict[str, JSONValue] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": version,
        "n_rows": len(rows),
        "n_reactions_in_csv": len(csv_records),
        "n_reactions_in_h5": n_reactions_in_h5,
        "rejection_histogram": {code: histogram[code] for code in sorted(histogram)},
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_json(manifest_path, manifest)

    histogram_text = (
        ", ".join(f"{code}={count}" for code, count in sorted(histogram.items())) or "none"
    )
    logger.info(
        "Inventory: %d row(s) from %d CSV and %d HDF5 reaction(s); %d rejection(s) [%s] -> %s",
        len(rows), len(csv_records), n_reactions_in_h5, len(rejections),
        histogram_text, parquet_path,
    )
    return InventoryResult(
        n_rows=len(rows),
        n_rejections=len(rejections),
        manifest_path=manifest_path,
        parquet_path=parquet_path,
    )


__all__ = [
    "INVENTORY_MANIFEST_FILENAME",
    "INVENTORY_PARQUET_FILENAME",
    "INVENTORY_STAGE",
    "InventoryResult",
    "build_inventory",
]
