"""Run-report assembly for the idempotent end-to-end G0 pipeline.

The report is written on every run, including a run that stops at a failed
stage, and is assembled from the artifacts already on disk: totals, the
rejection histogram, and the final leak verdict are read best-effort so a
partially completed pipeline still produces an honest document.  The config
digest covers only the deterministic configuration blocks (never a timestamp),
and the document is written through the atomic JSON writer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.g0.dedup import DUPLICATE_LEDGER_FILENAME
from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.inventory import INVENTORY_MANIFEST_FILENAME
from pes2ts_core.g0.neardup import LEAK_AUDIT_FILENAME
from pes2ts_core.g0.rejections import SUMMARY_FILENAME
from pes2ts_core.g0.split_sources import (
    SPLIT_LABELS,
    SPLIT_MANIFEST_FILENAME,
)
from pes2ts_core.g0.strata import (
    COHORT_TRIAL_FILENAME,
    STRATA_REPORT_FILENAME,
)
from pes2ts_core.utils.hashing import sha256_bytes, stable_json_dumps
from pes2ts_core.utils.jsonio import read_json

#: Run-report filename written under ``config["paths"]["manifests"]``.
RUN_REPORT_FILENAME: Final[str] = "g0_run_report.json"


def run_report_path(config: Mapping[str, Any]) -> Path:
    """Return the path of ``g0_run_report.json`` for *config*."""
    return Path(config["paths"]["manifests"]) / RUN_REPORT_FILENAME


def _optional_document(path: Path) -> dict[str, Any] | None:
    """Return the JSON object at *path*, or ``None`` when unreadable."""
    if not path.is_file():
        return None
    try:
        document = read_json(path)
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _put_int(
    target: dict[str, Any],
    key: str,
    document: Mapping[str, Any] | None,
    source_key: str,
) -> None:
    """Copy ``document[source_key]`` into *target* when it is an integer."""
    if document is None:
        return
    value = document.get(source_key)
    if isinstance(value, int) and not isinstance(value, bool):
        target[key] = value


def collect_totals(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return best-effort row and count totals from the stage manifests."""
    manifests = Path(config["paths"]["manifests"])
    interim = Path(config["paths"]["interim"])
    totals: dict[str, Any] = {}
    inventory = _optional_document(manifests / INVENTORY_MANIFEST_FILENAME)
    _put_int(totals, "n_inventory_rows", inventory, "n_rows")
    histogram = inventory.get("rejection_histogram") if inventory else None
    if isinstance(histogram, dict):
        totals["n_rejections"] = sum(
            value for value in histogram.values() if isinstance(value, int)
        )
    ledger = _optional_document(manifests / DUPLICATE_LEDGER_FILENAME)
    for key in ("n_unique_identities", "n_exact_duplicates", "n_reverse_pairs"):
        _put_int(totals, key, ledger, key)
    audit = _optional_document(manifests / LEAK_AUDIT_FILENAME)
    for key in ("n_pairs_over_threshold", "n_cross_split_known_duplicates"):
        _put_int(totals, f"audit_{key}", audit, key)
    split = _optional_document(manifests / SPLIT_MANIFEST_FILENAME)
    counts = split.get("counts") if split else None
    if isinstance(counts, dict):
        for label in SPLIT_LABELS:
            _put_int(totals, f"n_{label}", counts, label)
    strata = _optional_document(manifests / STRATA_REPORT_FILENAME)
    strata_totals = strata.get("totals") if strata else None
    if isinstance(strata_totals, dict):
        for key in ("n_strata", "n_cohort_stratified"):
            _put_int(totals, key, strata_totals, key)
    trial = _optional_document(interim / COHORT_TRIAL_FILENAME)
    _put_int(totals, "n_cohort_trial", trial, "size")
    return totals


def rejection_histogram(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the rejection histogram from ``rejection_summary.json``."""
    summary = _optional_document(
        Path(config["paths"]["manifests"]) / SUMMARY_FILENAME
    )
    by_code = summary.get("by_code") if summary else None
    if not isinstance(by_code, dict):
        return {}
    return {
        str(code): count
        for code, count in by_code.items()
        if isinstance(count, int) and not isinstance(count, bool)
    }


def current_leak_status(config: Mapping[str, Any]) -> str | None:
    """Return the frozen split manifest's ``leak_status``, when present."""
    split = _optional_document(
        Path(config["paths"]["manifests"]) / SPLIT_MANIFEST_FILENAME
    )
    status = split.get("leak_status") if split else None
    return status if isinstance(status, str) else None


def config_digest(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 of the deterministic configuration blocks."""
    source = config.get("source")
    md5s: dict[str, Any] = {}
    if isinstance(source, Mapping):
        files = source.get("files")
        if isinstance(files, Mapping):
            for key, entry in files.items():
                if isinstance(entry, Mapping):
                    md5s[str(entry.get("filename", key))] = entry.get("md5")
    payload: dict[str, Any] = {
        "source": {
            "zenodo_record": str(source.get("zenodo_record")) if isinstance(source, Mapping) else "",
            "zenodo_revision": str(source.get("zenodo_revision")) if isinstance(source, Mapping) else "",
            "md5": md5s,
        },
        "paths": dict(config["paths"]),
        "split": dict(config.get("split", {})),
        "near_dup": dict(config.get("near_dup", {})),
        "cohorts": dict(config.get("cohorts", {})),
    }
    return sha256_bytes(stable_json_dumps(payload).encode("utf-8"))


def build_document(
    config: Mapping[str, Any],
    stage_records: Sequence[Mapping[str, Any]],
    *,
    started_at: str,
    duration_seconds: float,
    leak_status: str | None,
) -> dict[str, Any]:
    """Build the ``g0_run_report.json`` document for one pipeline run."""
    document: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "stages": [dict(record) for record in stage_records],
        "totals": collect_totals(config),
        "leak_status": leak_status,
        "rejection_histogram": rejection_histogram(config),
        "config_digest": config_digest(config),
        "data_root": str(config["paths"].get("data_root", "")),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return document


__all__ = [
    "RUN_REPORT_FILENAME",
    "build_document",
    "collect_totals",
    "config_digest",
    "current_leak_status",
    "rejection_histogram",
    "run_report_path",
]
