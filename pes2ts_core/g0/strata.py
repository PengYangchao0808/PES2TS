"""Deterministic cohort selection over the G0 stratification preview.

Cohorts never use an RNG: within a stratum, members are ranked by
``sha256(f"{seed}:{reaction_id}")`` and each stratum gets a largest-remainder
quota proportional to its inventory share.  The stratified cohort is selected
first and the trial cohort is then selected from the stratified members, so
the trial is nested inside the stratified cohort whenever both requests fit.

The strata -- including the non-authoritative bond-change preview -- are
computed by :mod:`pes2ts_core.g0.strata_preview`, whose public functions are
re-exported here so ``pes2ts_core.g0.strata`` stays the documented entry point
G1 must import from.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
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
    """Outcome of one :func:`select_cohorts` run."""

    trial_size: int
    stratified_size: int
    trial_path: Path
    stratified_path: Path
    report_path: Path


def _strata_index(records: Sequence[Mapping[str, Any]]) -> dict[StratumKey, list[str]]:
    """Group reaction ids by stratum key in first-seen order."""
    grouped: dict[StratumKey, list[str]] = {}
    for record in records:
        key: StratumKey = (
            str(record["element_set"]),
            int(record["n_components"]),
            str(record["heavy_atom_bucket"]),
        )
        grouped.setdefault(key, []).append(str(record["reaction_id"]))
    return grouped


def _key_text(key: StratumKey) -> str:
    """Return the stable JSON text form of a stratum key (``|``-joined)."""
    return f"{key[0]}|{key[1]}|{key[2]}"


def _selection_key(seed_text: str, reaction_id: str) -> str:
    """Return the deterministic ranking key of *reaction_id* under *seed_text*."""
    return sha256_bytes(f"{seed_text}:{reaction_id}".encode("utf-8"))


def _largest_remainder(counts: Mapping[StratumKey, int], size: int) -> dict[StratumKey, int]:
    """Allocate *size* slots proportionally with the largest-remainder method.

    Exact quotas are rational, floors sum to at most *size*, leftover slots go
    to the largest fractional parts (ties by stratum-key sort), and the
    allocation is capped at the total pool.
    """
    total = sum(counts.values())
    if total == 0 or size <= 0:
        return dict.fromkeys(counts, 0)
    size = min(size, total)
    exact = {key: Fraction(count * size, total) for key, count in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    leftover = size - sum(quotas.values())
    ranked = sorted(exact, key=lambda key: (-(exact[key] - int(exact[key])), key))
    for key in ranked[:leftover]:
        quotas[key] += 1
    return quotas


def _select_members(grouped: Mapping[StratumKey, list[str]], size: int, seed_text: str) -> list[str]:
    """Select *size* reaction ids, quota per stratum, hash-ranked within."""
    quotas = _largest_remainder(
        {key: len(ids) for key, ids in grouped.items()}, size
    )
    selected: list[str] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key], key=lambda rid: (_selection_key(seed_text, rid), rid)
        )
        selected.extend(ranked[: quotas[key]])
    return selected


def _counts(grouped: Mapping[StratumKey, list[str]], members: Sequence[str]) -> Counter[StratumKey]:
    """Count *members* per stratum key."""
    owner = {rid: key for key, ids in grouped.items() for rid in ids}
    return Counter(owner[rid] for rid in members)


def _cohort_document(
    cohort: str, version: str, members: Sequence[str], generated_at: str,
    grouped: Mapping[StratumKey, list[str]], counts: Mapping[StratumKey, int],
) -> dict[str, JSONValue]:
    """Build the JSON document of one cohort (all strata, zero counts included)."""
    strata: dict[str, JSONValue] = {
        _key_text(key): counts.get(key, 0) for key in grouped
    }
    members_json: list[JSONValue] = list(sorted(members))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": version,
        "cohort": cohort,
        "size": len(members),
        "members": members_json,
        "strata": strata,
        "generated_at": generated_at,
    }


def select_cohorts(config: Mapping[str, Any]) -> CohortsResult:
    """Select and write the deterministic trial and stratified cohorts.

    The inventory Parquet is required (a missing file raises
    :class:`FileNotFoundError` with the ``g0 inventory`` hint).  When
    ``cohorts.require_authoritative_strata`` is true, no cohort is produced and
    :class:`AuthoritativeStrataRequiredError` names the G1 obligation instead.

    The stratified cohort is selected with ``seed2 =
    int(sha256(f"{seed}:stratified"), 16)`` (recorded as ``seed_stratified`` in
    the report); the trial cohort is selected *from the stratified members*
    with the raw ``split.seed``, so it is a subset of the stratified cohort
    whenever both requests fit.  A request larger than its pool logs a warning
    and selects the whole pool.  Both cohort files and the strata report are
    written atomically; the report carries ``authoritative=false`` plus
    :data:`G1_OBLIGATION` and lists every inventory stratum, including strata
    with zero cohort members.
    """
    cohorts_config = config["cohorts"]
    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
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

    version = dataset_version(config)
    records = compute_strata(read_parquet(inventory_path))
    grouped = _strata_index(records)
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
    stratified_members = _select_members(grouped, stratified_requested, str(seed2))
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
    trial_members = _select_members(stratified_grouped, trial_requested, str(seed))

    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    trial_counts = _counts(grouped, trial_members)
    stratified_counts = _counts(grouped, stratified_members)
    trial_document = _cohort_document(
        "trial", version, trial_members, generated_at, grouped, trial_counts
    )
    stratified_document = _cohort_document(
        "stratified", version, stratified_members, generated_at, grouped,
        stratified_counts,
    )

    strata_rows: list[JSONValue] = [
        {
            "key": _key_text(key),
            "n_inventory": len(grouped[key]),
            "n_cohort_trial": trial_counts.get(key, 0),
            "n_cohort_stratified": stratified_counts.get(key, 0),
        }
        for key in sorted(grouped, key=_key_text)
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
        "strata_source": STRATA_SOURCE_PREVIEW,
        "authoritative": False,
        "g1_obligation": G1_OBLIGATION,
        "seed": seed,
        "seed_stratified": seed2,
        "strata": strata_rows,
        "totals": totals,
        "generated_at": generated_at,
    }

    trial_path = interim_dir / COHORT_TRIAL_FILENAME
    stratified_path = interim_dir / COHORT_STRATIFIED_FILENAME
    report_path = manifests_dir / STRATA_REPORT_FILENAME
    write_json(trial_path, trial_document)
    write_json(stratified_path, stratified_document)
    write_json(report_path, report)

    logger.info(
        "Cohorts: trial=%d, stratified=%d, strata=%d (preview, authoritative=false)",
        len(trial_members), len(stratified_members), len(grouped),
    )
    return CohortsResult(
        trial_size=len(trial_members),
        stratified_size=len(stratified_members),
        trial_path=trial_path,
        stratified_path=stratified_path,
        report_path=report_path,
    )


__all__ = [
    "COHORT_STRATIFIED_FILENAME", "COHORT_TRIAL_FILENAME",
    "EXIT_AUTHORITATIVE_STRATA_REQUIRED", "G1_OBLIGATION",
    "STRATA_REPORT_FILENAME", "AuthoritativeStrataRequiredError",
    "CohortsResult", "compute_bond_changes_preview", "compute_strata",
    "select_cohorts",
]
