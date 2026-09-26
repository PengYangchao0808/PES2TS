"""Authoritative G1 strata computation and cohort re-export.

``g0 cohorts`` publishes only a non-authoritative preview; this module
discharges G1's obligation to recompute the bond-change strata from the
authoritative counts.  The counters are **not** recomputed from SMILES: they
are read from the summary Parquet written by ``g1 build`` (a full build over
every inventory row), the single source of truth for them.  The stratum-key
axes (``element_set``, ``n_components``, ``heavy_atom_bucket``) still come from
the imported :func:`pes2ts_core.g0.strata.compute_strata`, so preview and
authoritative strata remain row-comparable.

Rejected and unbuilt reactions stay in the records -- and therefore in the
denominator -- with the shared ``-1`` unavailability sentinel, so no inventory
row is ever dropped silently.  Cohort selection is delegated to the imported
:func:`pes2ts_core.g0.strata.select_cohorts_from_records`, whose deterministic
rules (largest-remainder quotas, hash ranking, nested trial) are never
reimplemented or drifted here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g0.strata import (
    INVENTORY_HINT,
    compute_strata,
    select_cohorts_from_records,
    strata_index,
)
from pes2ts_core.g1 import G1_MANIFEST_SCHEMA_VERSION
from pes2ts_core.g1.build import BUILD_HINT, SUMMARY_FILENAME
from pes2ts_core.utils.hashing import JSONValue, sha256_bytes
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.utils.parquet_io import read_parquet

logger = logging.getLogger(__name__)

#: Strata provenance label for the authoritative report.
STRATA_SOURCE_G1: Final[str] = "g1_authoritative"
#: Authoritative counters of a rejected or unbuilt reaction (shared sentinel).
SENTINEL: Final[int] = -1
#: Side manifest filename written under ``config["paths"]["manifests"]``.
STRATA_MANIFEST_FILENAME: Final[str] = "g1_strata_manifest.json"
#: Raised text when the summary does not cover every inventory row.
FULL_BUILD_ERROR: Final[str] = (
    "authoritative strata require a full build over every inventory row; "
    "run `g1 build` without --cohort/--limit"
)


@dataclass(frozen=True, slots=True)
class StrataAuthResult:
    """Outcome of one authoritative strata rebuild."""

    n_inventory: int
    n_valid: int
    n_rejected: int
    report_path: Path
    trial_path: Path
    stratified_path: Path
    manifest_path: Path


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def authoritative_records(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Return the authoritative strata records and their build basis.

    One record per inventory row is derived through the imported
    :func:`pes2ts_core.g0.strata.compute_strata` (key axes and row order stay
    G0's).  The three bond-change counters are then REPLACED by the
    authoritative values read from the ``g1 build`` summary Parquet:
    ``status="valid"`` rows take ``(n_formed, n_broken, n_order_changed)``,
    while rejected or unbuilt rows take the shared ``-1`` sentinel so they stay
    in the records with unavailable counters instead of being silently
    dropped.

    The basis dict carries ``n_inventory``, ``n_built``, ``n_valid``,
    ``n_rejected``, ``n_unbuilt``, and ``n_counter_agreements`` (valid rows
    whose authoritative counters equal the imported preview counters).  The
    build already enforces its preview cross-check, so any disagreement is
    real semantic drift and raises :class:`ValueError` naming up to three
    offenders.  A missing inventory raises :class:`FileNotFoundError` with the
    ``g0 inventory`` hint; a missing summary with the ``g1 build`` hint.
    """
    interim_dir = Path(config["paths"]["interim"])
    inventory_path = interim_dir / INVENTORY_PARQUET_FILENAME
    summary_path = interim_dir / SUMMARY_FILENAME
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Missing inventory {inventory_path}; {INVENTORY_HINT}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary {summary_path}; {BUILD_HINT}")

    records = compute_strata(read_parquet(inventory_path))
    summary_by_id = {
        str(row["reaction_id"]): row for row in read_parquet(summary_path).to_pylist()
    }
    n_built = 0
    n_valid = 0
    n_counter_agreements = 0
    offenders: list[str] = []
    for record in records:
        reaction_id = str(record["reaction_id"])
        summary_row = summary_by_id.get(reaction_id)
        counters = (SENTINEL, SENTINEL, SENTINEL)
        if summary_row is not None:
            n_built += 1
            if str(summary_row["status"]) == "valid":
                counters = (
                    int(summary_row["n_formed"]),
                    int(summary_row["n_broken"]),
                    int(summary_row["n_order_changed"]),
                )
                n_valid += 1
                preview = (
                    record["n_bonds_formed"],
                    record["n_bonds_broken"],
                    record["n_bond_order_changed"],
                )
                if counters == preview:
                    n_counter_agreements += 1
                elif len(offenders) < 3:
                    offenders.append(reaction_id)
        (
            record["n_bonds_formed"],
            record["n_bonds_broken"],
            record["n_bond_order_changed"],
        ) = counters
    if offenders:
        msg = f"authoritative counters disagree with the preview for {offenders}"
        raise ValueError(msg)
    n_inventory = len(records)
    basis = {
        "n_inventory": n_inventory,
        "n_built": n_built,
        "n_valid": n_valid,
        "n_rejected": n_built - n_valid,
        "n_unbuilt": n_inventory - n_built,
        "n_counter_agreements": n_counter_agreements,
    }
    return records, basis


def rebuild_authoritative_strata(config: Mapping[str, Any]) -> StrataAuthResult:
    """Re-derive the strata and cohorts authoritatively and write the artifacts.

    A full build over every inventory row is required: a summary that does not
    cover the whole inventory raises :class:`ValueError` naming the required
    ``g1 build`` invocation.  The authoritative records are selected through
    the imported :func:`pes2ts_core.g0.strata.select_cohorts_from_records` with
    ``strata_source="g1_authoritative"`` and ``authoritative=true``, so the
    preview-only ``require_authoritative_strata`` halt never applies; the three
    G0 cohort artifacts are rewritten in place.  A side manifest
    ``g1_strata_manifest.json`` additionally records the dataset version, the
    build basis, the seed pair, the stratum count, and the artifact paths.
    """
    records, basis = authoritative_records(config)
    if basis["n_unbuilt"]:
        raise ValueError(FULL_BUILD_ERROR)
    selection = select_cohorts_from_records(
        records,
        config,
        strata_source=STRATA_SOURCE_G1,
        authoritative=True,
        g1_obligation=None,
    )
    seed = int(config["split"]["seed"])
    manifest_path = Path(config["paths"]["manifests"]) / STRATA_MANIFEST_FILENAME
    manifest: dict[str, JSONValue] = {
        "schema_version": G1_MANIFEST_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "basis": {
            "n_inventory": basis["n_inventory"],
            "n_built": basis["n_built"],
            "n_valid": basis["n_valid"],
            "n_rejected": basis["n_rejected"],
            "n_unbuilt": basis["n_unbuilt"],
            "n_counter_agreements": basis["n_counter_agreements"],
        },
        "seed": seed,
        "seed_stratified": int(sha256_bytes(f"{seed}:stratified".encode("utf-8")), 16),
        "n_strata": len(strata_index(records)),
        "artifacts": {
            "strata_report": str(selection.report_path),
            "cohort_trial": str(selection.trial_path),
            "cohort_stratified": str(selection.stratified_path),
        },
        "generated_at": _now(),
    }
    write_json(manifest_path, manifest)
    logger.debug("G1 strata manifest written: %s", manifest_path)
    return StrataAuthResult(
        n_inventory=basis["n_inventory"],
        n_valid=basis["n_valid"],
        n_rejected=basis["n_rejected"],
        report_path=selection.report_path,
        trial_path=selection.trial_path,
        stratified_path=selection.stratified_path,
        manifest_path=manifest_path,
    )


__all__ = [
    "FULL_BUILD_ERROR",
    "SENTINEL",
    "STRATA_MANIFEST_FILENAME",
    "STRATA_SOURCE_G1",
    "StrataAuthResult",
    "authoritative_records",
    "rebuild_authoritative_strata",
]
