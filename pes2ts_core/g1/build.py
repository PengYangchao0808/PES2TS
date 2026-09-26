"""G1 build stage: per-reaction documents, summary, manifest, and ledger.

:func:`build_g1` walks ``inventory.parquet`` (every row or an explicit cohort
subset), assembles one versioned reaction-change document per reaction, and
writes every document -- rejected reactions included -- sharded under
``interim/g1/reaction_change/<shard>/<reaction_id>.json``, so each exclusion
has a persisted machine-readable reason.  A summary Parquet, the G1 manifest,
and the coverage report follow, and every rejected reaction appends exactly one
idempotent ``g1_build`` entry to the unified rejection ledger (an identical
``(reaction_id, stage, code, detail)`` signature is skipped on re-runs).

The shard directory is the five-digit zero-padded numeric reaction-id suffix
divided by ``g1.shard_size``; an id without a numeric suffix falls back to a
deterministic hash bucket.  :func:`cohort_member_ids` resolves the CLI cohort
modes, and post-build reconciliation lives in
:mod:`pes2ts_core.g1.verify`.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g0.rejections import (
    LEDGER_FILENAME, Rejection, RejectionCode, RejectionLedger,
)
from pes2ts_core.g0.strata import COHORT_STRATIFIED_FILENAME, COHORT_TRIAL_FILENAME
from pes2ts_core.g1 import G1_MANIFEST_SCHEMA_VERSION
from pes2ts_core.g1.coverage import build_coverage_report, write_coverage
from pes2ts_core.g1.document import (
    CATEGORY_NAMES, DEFAULT_BOND_TOLERANCE, DEFAULT_MATCH_CAP,
    DEFAULT_MAX_CANDIDATES, DEFAULT_NEIGHBORHOOD_SHELL, build_reaction_change,
)
from pes2ts_core.utils.hashing import JSONValue, sha256_bytes, sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

#: Exit code returned when a build or verification cannot complete cleanly.
EXIT_G1_BUILD_FAILED: Final[int] = 22
#: Directory (under ``paths.interim``) holding the sharded per-reaction JSON.
G1_DIRNAME: Final[str] = "g1"
REACTION_CHANGE_DIRNAME: Final[str] = "reaction_change"
#: Summary Parquet, manifest, and coverage filenames (interim/manifests).
SUMMARY_FILENAME: Final[str] = "g1_reaction_change_summary.parquet"
MANIFEST_FILENAME: Final[str] = "g1_manifest.json"
COVERAGE_FILENAME: Final[str] = "g1_coverage.json"
#: Stage name stamped into every rejection emitted by this module.
BUILD_STAGE: Final[str] = "g1_build"
#: Default shard size when ``g1.shard_size`` is absent.
DEFAULT_SHARD_SIZE: Final[int] = 1000
#: Actionable hints attached to missing-input errors.
INVENTORY_HINT: Final[str] = "run `g0 inventory` first"
COHORT_HINT: Final[str] = "run `g0 cohorts` first"
BUILD_HINT: Final[str] = "run `g1 build` first"
#: CLI cohort mode -> ``g0 cohorts`` artifact filename.
COHORT_FILENAMES: Final[dict[str, str]] = {
    "trial": COHORT_TRIAL_FILENAME, "stratified": COHORT_STRATIFIED_FILENAME,
}
#: Summary Parquet columns (stored alphabetically by :func:`write_parquet`).
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id", "status", "failure_code", "n_formed", "n_broken", "n_order_changed",
    "n_h_migration", "core_size", "shell_size", "index_status", "pairing_status",
    "geometry_ok", "n_atoms", *CATEGORY_NAMES,
)


@dataclass(frozen=True, slots=True)
class G1Result:
    """Outcome of one :func:`build_g1` run."""

    n_total: int
    n_valid: int
    n_rejected: int
    summary_path: Path
    manifest_path: Path
    reactions_dir: Path


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``g1`` configuration block (or an empty mapping)."""
    g1_config = config.get("g1")
    return g1_config if isinstance(g1_config, Mapping) else {}


def _shard_size(config: Mapping[str, Any]) -> int:
    """Return the configured shard size."""
    return int(_settings(config).get("shard_size", DEFAULT_SHARD_SIZE))


def _numeric_suffix(reaction_id: str) -> int | None:
    """Return the integer after the last ``_`` of *reaction_id*, or ``None``."""
    suffix = reaction_id.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def shard_name(reaction_id: str, shard_size: int) -> str:
    """Return the five-digit shard directory name of *reaction_id*.

    The numeric reaction-id suffix divided by *shard_size* names the shard; an
    id without a numeric suffix falls back to
    ``(int(sha256(id), 16) % 10**9) // shard_size``, a stable hash bucket.
    """
    suffix = _numeric_suffix(reaction_id)
    if suffix is None:
        suffix = int(sha256_bytes(reaction_id.encode("utf-8")), 16) % 10**9
    return f"{suffix // shard_size:05d}"


def reactions_dir(interim_dir: str | Path) -> Path:
    """Return the sharded per-reaction document directory under *interim_dir*."""
    return Path(interim_dir) / G1_DIRNAME / REACTION_CHANGE_DIRNAME


def reaction_change_path(interim_dir: str | Path, reaction_id: str, shard_size: int) -> Path:
    """Return the sharded document path of *reaction_id*."""
    return reactions_dir(interim_dir) / shard_name(reaction_id, shard_size) / f"{reaction_id}.json"


def _summary_row(document: Mapping[str, Any]) -> dict[str, JSONValue]:
    """Return the summary Parquet row of one reaction-change document."""
    changes, center = document["bond_changes"], document["reaction_center"]
    categories, validation = document["categories"], document["validation"]
    ambiguity = document["ambiguity"]
    components = [*document["reactants"], *document["products"]]
    row: dict[str, JSONValue] = {
        "reaction_id": document["reaction_id"], "status": validation["status"],
        "failure_code": validation["failure_code"], "n_formed": len(changes["formed"]),
        "n_broken": len(changes["broken"]), "n_order_changed": len(changes["order_changed"]),
        "n_h_migration": len(changes["hydrogen_migration"]), "core_size": len(center["core"]),
        "shell_size": len(center["with_shell"]), "index_status": ambiguity["index"],
        "pairing_status": ambiguity["pairing"],
        "geometry_ok": all(component["geometry_check_ok"] for component in components),
        "n_atoms": document["mapping"]["n_atoms"],
    }
    for name in CATEGORY_NAMES:
        row[name] = categories[name]
    return row


def _summary_columns(rows: Sequence[Mapping[str, JSONValue]]) -> dict[str, list[JSONValue]]:
    """Transpose summary rows into the Parquet column mapping."""
    return {name: [row[name] for row in rows] for name in SUMMARY_COLUMNS}


def _ledger_signatures(manifests_dir: Path) -> set[tuple[str, str, str, str]]:
    """Return the persisted ``(reaction_id, stage, code, detail)`` signatures."""
    path = manifests_dir / LEDGER_FILENAME
    if not path.is_file():
        return set()
    signatures: set[tuple[str, str, str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record: object = json.loads(line)
        if isinstance(record, dict):
            signatures.add(
                (
                    str(record.get("reaction_id")),
                    str(record.get("stage")),
                    str(record.get("code")),
                    str(record.get("detail")),
                )
            )
    return signatures


def build_g1(config: Mapping[str, Any], *, reaction_ids: Sequence[str] | None = None) -> G1Result:
    """Build every G1 reaction-change document, summary, manifest, and ledger.

    ``reaction_ids=None`` builds every inventory row; otherwise exactly the
    given ids are built (an unknown id raises :class:`ValueError` listing up to
    five offenders).  A missing inventory raises :class:`FileNotFoundError`
    with the ``g0 inventory`` hint.  Rejected reactions are persisted like
    valid ones, and each appends one ledger entry whose signature is skipped
    when a previous run already recorded it.
    """
    logger = logging.getLogger(__name__)
    settings = _settings(config)
    shard_size = _shard_size(config)
    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
    inventory_path = interim_dir / INVENTORY_PARQUET_FILENAME
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Missing inventory {inventory_path}; {INVENTORY_HINT}")
    rows_by_id = {str(row["reaction_id"]): row for row in read_parquet(inventory_path).to_pylist()}
    if reaction_ids is None:
        selected = sorted(rows_by_id)
    else:
        unknown = sorted(set(reaction_ids) - set(rows_by_id))
        if unknown:
            raise ValueError(f"unknown reaction id(s): {unknown[:5]}")
        selected = sorted(set(reaction_ids))
    documents: list[dict[str, Any]] = []
    document_paths: dict[str, Path] = {}
    for reaction_id in selected:
        document = build_reaction_change(rows_by_id[reaction_id], config)
        path = reaction_change_path(interim_dir, reaction_id, shard_size)
        write_json(path, document)
        documents.append(document)
        document_paths[reaction_id] = path
    summary_rows = [_summary_row(document) for document in documents]
    summary_path = interim_dir / SUMMARY_FILENAME
    write_parquet(summary_path, _summary_columns(summary_rows))
    n_valid = sum(1 for row in summary_rows if row["status"] == "valid")
    histogram = Counter(str(row["failure_code"]) for row in summary_rows if row["failure_code"])
    manifest: dict[str, JSONValue] = {
        "schema_version": G1_MANIFEST_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "n_total": len(summary_rows), "n_valid": n_valid, "n_rejected": len(summary_rows) - n_valid,
        "by_code": {code: histogram[code] for code in sorted(histogram)},
        "shard_size": shard_size, "n_files": len(summary_rows),
        "summary_sha256": sha256_file(summary_path),
        "config": {
            "match_cap": int(settings.get("match_cap", DEFAULT_MATCH_CAP)),
            "max_candidates": int(settings.get("max_candidates", DEFAULT_MAX_CANDIDATES)),
            "bond_tolerance": float(settings.get("bond_tolerance", DEFAULT_BOND_TOLERANCE)),
            "neighborhood_shell": int(settings.get("neighborhood_shell", DEFAULT_NEIGHBORHOOD_SHELL)),
            "shard_size": shard_size,
        },
        "generated_at": _now(),
    }
    manifest_path = manifests_dir / MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    ledger = RejectionLedger.load(manifests_dir)
    existing = _ledger_signatures(manifests_dir)
    for document in documents:
        validation = document["validation"]
        if validation["status"] != "rejected":
            continue
        reaction_id, code = str(document["reaction_id"]), str(validation["failure_code"])
        detail = str(validation["failure_detail"])
        if (reaction_id, BUILD_STAGE, code, detail) in existing:
            continue
        ledger.add(Rejection(reaction_id=reaction_id, stage=BUILD_STAGE, code=RejectionCode(code),
                             detail=detail, source_pointer=str(document_paths[reaction_id])))
    ledger.write()
    write_coverage(build_coverage_report(summary_rows, config), manifests_dir / COVERAGE_FILENAME)
    logger.info("G1 build: %d reaction(s), %d valid, %d rejected -> %s",
                len(summary_rows), n_valid, len(summary_rows) - n_valid, summary_path)
    return G1Result(len(summary_rows), n_valid, len(summary_rows) - n_valid, summary_path,
                    manifest_path, reactions_dir(interim_dir))


def cohort_member_ids(config: Mapping[str, Any], cohort: str, *, limit: int | None = None) -> list[str] | None:
    """Return the sorted reaction ids of one CLI cohort mode.

    ``trial``/``stratified`` read the matching ``g0 cohorts`` artifact (a
    missing file raises :class:`FileNotFoundError` with the ``g0 cohorts``
    hint).  ``all`` returns ``None`` -- build every inventory row -- unless
    *limit* is given; every mode truncates to the first *limit* ids.
    """
    if cohort == "all":
        if limit is None:
            return None
        inventory_path = Path(config["paths"]["interim"]) / INVENTORY_PARQUET_FILENAME
        if not inventory_path.is_file():
            raise FileNotFoundError(f"Missing inventory {inventory_path}; {INVENTORY_HINT}")
        members = sorted({str(row["reaction_id"]) for row in read_parquet(inventory_path).to_pylist()})
    else:
        path = Path(config["paths"]["interim"]) / COHORT_FILENAMES[cohort]
        if not path.is_file():
            raise FileNotFoundError(f"Missing cohort {path}; {COHORT_HINT}")
        document = read_json(path)
        raw_members = document.get("members") if isinstance(document, Mapping) else None
        if not isinstance(raw_members, list):
            raise ValueError(f"cohort artifact {path} has no members list")
        members = sorted(str(member) for member in raw_members)
    return members[:limit] if limit is not None else members


__all__ = [
    "BUILD_HINT", "BUILD_STAGE", "COHORT_FILENAMES", "COVERAGE_FILENAME", "DEFAULT_SHARD_SIZE",
    "EXIT_G1_BUILD_FAILED", "G1_DIRNAME", "G1Result", "MANIFEST_FILENAME",
    "REACTION_CHANGE_DIRNAME", "SUMMARY_FILENAME", "build_g1", "cohort_member_ids",
    "reaction_change_path", "reactions_dir", "shard_name",
]
