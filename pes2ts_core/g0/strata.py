"""Deterministic cohort selection over the G0 stratification preview.

Cohorts never use an RNG: within a stratum, members are ranked by
``sha256(f"{seed}:{reaction_id}")`` and each stratum gets a largest-remainder
quota proportional to its inventory share.  The stratified cohort is selected
first and the trial cohort is then selected from the stratified members, so
the trial is nested inside the stratified cohort whenever both requests fit.

:func:`select_cohorts_from_records` is the record-level selection core: it
takes already-computed strata records and stamps the caller's provenance
(``strata_source``, ``authoritative``, and an optional ``g1_obligation`` note)
into ``strata_report.json``, so G1 can reuse the exact same selection rules
without reading the inventory or tripping the preview-only halt.
:func:`select_cohorts` is the G0 wrapper: it reads the inventory Parquet,
computes the preview strata, stops while only the preview exists when
``cohorts.require_authoritative_strata`` is set, and delegates to the
record-level core with ``strata_source="preview"``.

The strata -- including the non-authoritative bond-change preview -- are
computed by :mod:`pes2ts_core.g0.strata_preview`, whose public functions are
re-exported here so ``pes2ts_core.g0.strata`` stays the documented entry point
G1 must import from.  The pure selection primitives live in
:mod:`pes2ts_core.g0.strata_select` and are re-exported here too.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g0.strata_preview import (
    StratumKey,
    compute_bond_changes_preview,
    compute_strata,
)
from pes2ts_core.g0.strata_select import (
    cohort_document,
    key_text,
    largest_remainder,
    member_counts,
    select_members,
    selection_key,
    strata_index,
)
from pes2ts_core.utils.hashing import JSONValue, sha256_bytes
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.utils.parquet_io import read_parquet

logger = logging.getLogger(__name__)

#: Trial cohort filename written under ``config["paths"]["interim"]``.
COHORT_TRIAL_FILENAME: Final[str] = "cohort_trial.json"
#: Stratified cohort filename written under ``config["paths"]["interim"]``.
COHORT_STRATIFIED_FILENAME: Final[str] = "cohort_stratified.json"
#: Strata report filename written under ``config["paths"]["manifests"]``.
STRATA_REPORT_FILENAME: Final[str] = "strata_report.json"
#: Actionable hint attached to a missing-inventory error.
INVENTORY_HINT: Final[str] = "run `g0 inventory` first"
#: Default cohort sizes used when the config omits them.
DEFAULT_TRIAL_SIZE: Final[int] = 200
DEFAULT_STRATIFIED_SIZE: Final[int] = 1000
#: Exit code for a cohort run halted by ``require_authoritative_strata``.
EXIT_AUTHORITATIVE_STRATA_REQUIRED: Final[int] = 20
#: Strata provenance label: the bond-change axes come from the preview diff.
STRATA_SOURCE_PREVIEW: Final[str] = "preview"
#: Note stamped into the strata report so G1 cannot mistake preview for truth.
G1_OBLIGATION: Final[str] = (
    "G1 must recompute bond-change strata authoritatively via "
    "pes2ts_core.g0.strata.compute_bond_changes_preview import and re-derive "
    "cohorts"
)


class AuthoritativeStrataRequiredError(RuntimeError):
    """Raised when cohorts are requested from preview strata while G1 is pending."""


@dataclass(frozen=True, slots=True)
class CohortsResult:
    """Outcome of one cohort selection run."""

    trial_size: int
    stratified_size: int
    trial_path: Path
    stratified_path: Path
    report_path: Path


def select_cohorts_from_records(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    strata_source: str,
    authoritative: bool,
    g1_obligation: str | None,
) -> CohortsResult:
    """Select and write deterministic cohorts from precomputed strata records.

    Each record must carry ``reaction_id``, ``element_set``, ``n_components``,
    and ``heavy_atom_bucket``; the preview bond-change counters are not read
    here (they only reach a stratum key through the bucket).  The inventory
    Parquet is never opened and the ``require_authoritative_strata`` halt is
    never applied: the caller states the provenance, which is stamped into
    ``strata_report.json`` as ``strata_source`` and ``authoritative``, plus a
    ``g1_obligation`` note when one is given (the key is omitted entirely when
    it is ``None``, keeping preview reports unchanged).

    The stratified cohort is selected with ``seed2 =
    int(sha256(f"{seed}:stratified"), 16)`` (recorded as ``seed_stratified`` in
    the report); the trial cohort is selected *from the stratified members*
    with the raw ``split.seed``, so it is a subset of the stratified cohort
    whenever both requests fit.  A request larger than its pool logs a warning
    and selects the whole pool.  Both cohort files and the strata report are
    written atomically; the report lists every selected stratum, including
    strata with zero cohort members.
    """
    cohorts_config = config["cohorts"]
    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
    version = dataset_version(config)
    grouped = strata_index(records)
    trial_requested = int(cohorts_config.get("trial", DEFAULT_TRIAL_SIZE))
    stratified_requested = int(
        cohorts_config.get("stratified", DEFAULT_STRATIFIED_SIZE)
    )
    seed = int(config["split"]["seed"])
    seed2 = int(sha256_bytes(f"{seed}:stratified".encode("utf-8")), 16)

    if len(records) < stratified_requested:
        logger.warning(
            "Stratified cohort requested %d but the inventory holds %d; selecting all inventory reactions",
            stratified_requested,
            len(records),
        )
    stratified_members = select_members(grouped, stratified_requested, str(seed2))
    if len(stratified_members) < trial_requested:
        logger.warning(
            "Trial cohort requested %d but the stratified cohort holds %d; selecting all stratified reactions",
            trial_requested,
            len(stratified_members),
        )
    stratified_ids = set(stratified_members)
    stratified_grouped = {
        key: [rid for rid in ids if rid in stratified_ids]
        for key, ids in grouped.items()
    }
    trial_members = select_members(stratified_grouped, trial_requested, str(seed))

    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    trial_counts = member_counts(grouped, trial_members)
    stratified_counts = member_counts(grouped, stratified_members)
    trial_document = cohort_document(
        "trial", version, trial_members, generated_at, grouped, trial_counts
    )
    stratified_document = cohort_document(
        "stratified", version, stratified_members, generated_at, grouped,
        stratified_counts,
    )

    strata_rows: list[JSONValue] = [
        {
            "key": key_text(key),
            "n_inventory": len(grouped[key]),
            "n_cohort_trial": trial_counts.get(key, 0),
            "n_cohort_stratified": stratified_counts.get(key, 0),
        }
        for key in sorted(grouped, key=key_text)
    ]
    totals: dict[str, JSONValue] = {
        "n_inventory": len(records),
        "n_cohort_trial": len(trial_members),
        "n_cohort_stratified": len(stratified_members),
        "n_strata": len(grouped),
    }
    report: dict[str, JSONValue] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": version,
        "strata_source": strata_source,
        "authoritative": authoritative,
        "seed": seed,
        "seed_stratified": seed2,
        "strata": strata_rows,
        "totals": totals,
        "generated_at": generated_at,
    }
    if g1_obligation is not None:
        report["g1_obligation"] = g1_obligation

    trial_path = interim_dir / COHORT_TRIAL_FILENAME
    stratified_path = interim_dir / COHORT_STRATIFIED_FILENAME
    report_path = manifests_dir / STRATA_REPORT_FILENAME
    write_json(trial_path, trial_document)
    write_json(stratified_path, stratified_document)
    write_json(report_path, report)

    logger.info(
        "Cohorts: trial=%d, stratified=%d, strata=%d (%s, authoritative=%s)",
        len(trial_members),
        len(stratified_members),
        len(grouped),
        strata_source,
        str(authoritative).lower(),
    )
    return CohortsResult(
        trial_size=len(trial_members),
        stratified_size=len(stratified_members),
        trial_path=trial_path,
        stratified_path=stratified_path,
        report_path=report_path,
    )


def select_cohorts(config: Mapping[str, Any]) -> CohortsResult:
    """Select and write the deterministic trial and stratified cohorts.

    The inventory Parquet is required (a missing file raises
    :class:`FileNotFoundError` with the ``g0 inventory`` hint).  When
    ``cohorts.require_authoritative_strata`` is true, no cohort is produced and
    :class:`AuthoritativeStrataRequiredError` names the G1 obligation instead.
    Authoritative callers use :func:`select_cohorts_from_records` directly and
    never pass through this preview-only halt.

    The strata come from the non-authoritative preview and the report carries
    ``strata_source="preview"``, ``authoritative=false``, and
    :data:`G1_OBLIGATION`; selection delegates to
    :func:`select_cohorts_from_records`, which owns the deterministic rules.
    """
    cohorts_config = config["cohorts"]
    interim_dir = Path(config["paths"]["interim"])
    inventory_path = interim_dir / INVENTORY_PARQUET_FILENAME
    if not inventory_path.is_file():
        msg = f"Missing inventory {inventory_path}; {INVENTORY_HINT}"
        raise FileNotFoundError(msg)
    if bool(cohorts_config.get("require_authoritative_strata", False)):
        msg = (
            "cohorts.require_authoritative_strata=true: G0 only publishes a "
            "non-authoritative preview; G1 must recompute bond-change strata "
            "authoritatively via "
            "pes2ts_core.g0.strata.compute_bond_changes_preview (import, never "
            "reimplement) and re-derive cohorts"
        )
        raise AuthoritativeStrataRequiredError(msg)

    records = compute_strata(read_parquet(inventory_path))
    return select_cohorts_from_records(
        records,
        config,
        strata_source=STRATA_SOURCE_PREVIEW,
        authoritative=False,
        g1_obligation=G1_OBLIGATION,
    )


__all__ = [
    "COHORT_STRATIFIED_FILENAME", "COHORT_TRIAL_FILENAME",
    "EXIT_AUTHORITATIVE_STRATA_REQUIRED", "G1_OBLIGATION",
    "STRATA_REPORT_FILENAME", "STRATA_SOURCE_PREVIEW", "StratumKey",
    "AuthoritativeStrataRequiredError", "CohortsResult",
    "cohort_document", "compute_bond_changes_preview", "compute_strata",
    "key_text", "largest_remainder", "member_counts", "select_cohorts",
    "select_cohorts_from_records", "select_members", "selection_key",
    "strata_index",
]
