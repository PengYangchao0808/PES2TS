"""G2 batch artifacts: per-reaction documents, summary, manifest, coverage.

Every writer uses the shared atomic primitives and digests; two builds over
identical inputs are byte-identical (``generated_at`` injectable is the only
volatile field; the manifest ``run`` block varies with call semantics).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pyarrow as pa

from pes2ts_core.g1.document import CATEGORY_NAMES
from pes2ts_core.g2 import SCHEMA_COVERAGE, SCHEMA_MANIFEST, SCHEMA_PATH
from pes2ts_core.g2.status import STATUS_VALID
from pes2ts_core.utils.hashing import JSONValue, sha256_bytes, stable_json_dumps
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.utils.parquet_io import write_parquet

#: Frames-parquet columns, in the task-7 ``frame_metrics`` record order plus
#: the task-9 ``energy_rel_kcal_raw`` extension (raw == ``energy_rel_kcal`` on
#: forward rows; the direction-renormalized value on reverse rows).
FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id", "frame_index", "energy_rel_kcal", "energy_rel_kcal_raw",
    "rmsd_to_start", "rmsd_to_end",
    "step_max", "step_rmsd", "min_nonbonded_distance", "event_distances",
)
#: Summary-parquet columns (the parquet writer stores them sorted by name).
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "reaction_id", "status", "failure_code", "direction", "direction_recovered",
    "n_frames", "n_attempts", "energy_min", "energy_max",
)
#: Keys the G2 verifier rejects anywhere inside a JSON artifact.
FORBIDDEN_DOCUMENT_KEYS: Final[tuple[str, ...]] = ("coordinates", "EHG", "forces")

#: Explicit schema of the frames parquet (dict columns pre-serialized to JSON).
_FRAME_SPEC: Final[tuple[tuple[str, pa.DataType], ...]] = (
    ("reaction_id", pa.string()), ("frame_index", pa.int64()),
    ("energy_rel_kcal", pa.float64()), ("energy_rel_kcal_raw", pa.float64()),
    ("rmsd_to_start", pa.float64()),
    ("rmsd_to_end", pa.float64()), ("step_max", pa.float64()),
    ("step_rmsd", pa.float64()), ("min_nonbonded_distance", pa.float64()),
    ("event_distances", pa.string()),
)
_FRAME_SCHEMA: Final[pa.Schema] = pa.schema([pa.field(name, kind) for name, kind in _FRAME_SPEC])


class ManifestCountMismatch(ValueError):
    """Raised when explicit manifest counts contradict the summary rows."""


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def config_digest(config: Mapping[str, Any], *, xtb_sha256: str, path_inp: str) -> str:
    """SHA-256 of the g2 configuration block plus the xTB/path.inp identity.

    This is g2-specific on purpose: g0's ``pipeline_report.config_digest``
    covers different keys, so it must never be reused here.
    """
    payload = {"g2": config.get("g2", {}), "xtb_sha256": str(xtb_sha256), "path_inp": str(path_inp)}
    return sha256_bytes(stable_json_dumps(payload).encode("utf-8"))


def _frames_summary(frames: Sequence[Mapping[str, Any]]) -> dict[str, JSONValue]:
    """Scalar-only per-reaction path summary (never per-frame arrays)."""
    energies = [
        float(record["energy_rel_kcal"])
        for record in frames
        if record.get("energy_rel_kcal") is not None
    ]
    nonbonded = [
        float(record["min_nonbonded_distance"])
        for record in frames
        if record.get("min_nonbonded_distance") is not None
    ]
    steps = [float(record["step_max"]) for record in frames]
    return {
        "n_frames": len(frames), "energy_min": min(energies) if energies else None,
        "energy_max": max(energies) if energies else None,
        "worst_step_max": max(steps) if steps else None,
        "min_nonbonded": min(nonbonded) if nonbonded else None,
    }


def reaction_document(  # noqa: PLR0913 -- pipeline-facing record builder
    *,
    reaction_id: str,
    status: str,
    failure_code: str | None = None,
    failure_detail: str | None = None,
    direction: str = "forward",
    direction_recovered: bool = False,
    dataset_version: str | None = None,
    sources: Mapping[str, str] | None = None,
    endpoints: Mapping[str, JSONValue] | None = None,
    attempts: Sequence[Mapping[str, JSONValue]] | None = None,
    frames: Sequence[Mapping[str, Any]] | None = None,
    validity: Mapping[str, JSONValue] | None = None,
    config_digest: str | None = None,
) -> dict[str, JSONValue]:
    """Build one ``g2_path_v1`` document from pipeline outputs.

    Only digests and scalar summaries are stored: no coordinate or energy
    arrays (the verifier rejects the keys ``coordinates``/``EHG``/``forces``).
    """
    document: dict[str, JSONValue] = {
        "schema_version": SCHEMA_PATH,
        "reaction_id": str(reaction_id),
        "status": str(status),
        "failure_code": None if failure_code is None else str(failure_code),
        "failure_detail": None if failure_detail is None else str(failure_detail),
        "direction": str(direction),
        "direction_recovered": bool(direction_recovered),
        "sources": dict(sources or {}),
        "endpoints": dict(endpoints or {}),
        "attempts": [dict(attempt) for attempt in (attempts or ())],
        "frames": _frames_summary(frames or ()),
        "validity": dict(validity or {}),
        "config_digest": None if config_digest is None else str(config_digest),
    }
    if dataset_version is not None:
        document["dataset_version"] = str(dataset_version)
    return document


def write_frames_parquet(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write one parquet row per frame metric record.

    ``event_distances`` maps are serialized with :func:`stable_json_dumps`
    (sorted keys, compact separators); the shared parquet writer fixes the
    column order, so identical rows always produce identical bytes.
    """
    records = [
        {**{name: row[name] for name in FRAME_COLUMNS if name != "event_distances"},
         "event_distances": stable_json_dumps(dict(row["event_distances"]))}
        for row in rows
    ]
    write_parquet(path, pa.Table.from_pylist(records, schema=_FRAME_SCHEMA))


def build_summary(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, JSONValue]]:
    """One deterministic summary row per reaction path document."""
    return [
        {
            "reaction_id": str(document["reaction_id"]), "status": str(document["status"]),
            "failure_code": document["failure_code"], "direction": str(document["direction"]),
            "direction_recovered": bool(document["direction_recovered"]),
            "n_frames": int(document["frames"]["n_frames"]), "n_attempts": len(document["attempts"]),
            "energy_min": document["frames"]["energy_min"], "energy_max": document["frames"]["energy_max"],
        }
        for document in documents
    ]


def write_summary(path: str | Path, rows: Sequence[Mapping[str, JSONValue]]) -> None:
    """Atomically write the ``g2_summary.parquet`` rows (stable column order)."""
    write_parquet(path, {name: [row.get(name) for row in rows] for name in SUMMARY_COLUMNS})


def build_manifest(  # noqa: PLR0913 -- pipeline-facing artifact builder
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    run_counts: Mapping[str, int],
    xtb_fingerprint: Mapping[str, JSONValue],
    config: Mapping[str, Any],
    generated_at: str | None = None,
    counts: Mapping[str, int] | None = None,
) -> dict[str, JSONValue]:
    """Build the ``g2_manifest_v1`` manifest from one batch of summary rows.

    ``counts`` optionally carries pipeline-claimed ``n_total``/``n_valid``/
    ``n_failed``; any contradiction with the counts derived from the summary
    rows raises :class:`ManifestCountMismatch`.  An empty batch still yields a
    valid manifest.
    """
    n_valid = sum(1 for row in summary_rows if str(row["status"]) == STATUS_VALID)
    derived = {
        "n_total": len(summary_rows),
        "n_valid": n_valid,
        "n_failed": len(summary_rows) - n_valid,
    }
    if counts is not None:
        claimed = {name: int(counts.get(name, derived[name])) for name in derived}
        if claimed != derived:
            raise ManifestCountMismatch(
                f"manifest counts {claimed} contradict summary rows {derived}"
            )
    histogram: dict[str, int] = {}
    recovered = 0
    for row in summary_rows:
        if row["failure_code"]:
            code = str(row["failure_code"])
            histogram[code] = histogram.get(code, 0) + 1
        recovered += bool(row["direction_recovered"])
    manifest: dict[str, JSONValue] = {
        "schema_version": SCHEMA_MANIFEST,
        **derived,
        "by_code": {code: histogram[code] for code in sorted(histogram)},
        "reverse_recovery_rate": round(recovered / derived["n_total"], 4) if derived["n_total"] else 0.0,
        "run": dict(run_counts),
        "summary_sha256": sha256_bytes(stable_json_dumps(list(summary_rows)).encode("utf-8")),
        "xtb": dict(xtb_fingerprint),
        "config_digest": config_digest(
            config,
            xtb_sha256=str(xtb_fingerprint.get("sha256", "")),
            path_inp=str(xtb_fingerprint.get("path_inp", "")),
        ),
        "generated_at": generated_at if generated_at is not None else _now(),
    }
    return manifest


def write_manifest(
    path: str | Path,
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    run_counts: Mapping[str, int],
    xtb_fingerprint: Mapping[str, JSONValue],
    config: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, JSONValue]:
    """Atomically write :func:`build_manifest`'s result and return it."""
    manifest = build_manifest(
        summary_rows,
        run_counts=run_counts,
        xtb_fingerprint=xtb_fingerprint,
        config=config,
        generated_at=generated_at,
    )
    write_json(path, manifest)
    return manifest


def write_coverage(
    path: str | Path,
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    categories_by_reaction: Mapping[str, Mapping[str, bool]],
    current_by_code: Mapping[str, int],
    historical_failed_by_code: Mapping[str, int],
    generated_at: str | None = None,
) -> dict[str, JSONValue]:
    """Write the ``g2_coverage_v1`` per-category cross of the current batch.

    ``categories_by_reaction`` carries the seven G1 category booleans per
    reaction; ``historical_failed_by_code`` counts prior unified-ledger
    ``stage="g2_path"`` failures so ``--force`` reruns keep history visible.
    """
    n_valid = sum(1 for row in summary_rows if str(row["status"]) == STATUS_VALID)
    categories: dict[str, JSONValue] = {}
    for name in CATEGORY_NAMES:
        members = [
            row for row in summary_rows if bool(categories_by_reaction.get(str(row["reaction_id"]), {}).get(name))
        ]
        valid = sum(1 for row in members if str(row["status"]) == STATUS_VALID)
        categories[name] = {
            "n_total": len(members),
            "n_valid": valid,
            "n_failed": len(members) - valid,
        }
    coverage: dict[str, JSONValue] = {
        "schema_version": SCHEMA_COVERAGE,
        "overall": {
            "n_total": len(summary_rows),
            "n_valid": n_valid,
            "n_failed": len(summary_rows) - n_valid,
        },
        "categories": categories,
        "current_by_code": {code: int(current_by_code[code]) for code in sorted(current_by_code)},
        "historical_failed_by_code": {
            code: int(historical_failed_by_code[code]) for code in sorted(historical_failed_by_code)
        },
        "generated_at": generated_at if generated_at is not None else _now(),
    }
    write_json(path, coverage)
    return coverage


__all__ = [
    "FORBIDDEN_DOCUMENT_KEYS", "FRAME_COLUMNS", "SUMMARY_COLUMNS", "ManifestCountMismatch",
    "build_manifest", "build_summary", "config_digest", "reaction_document",
    "write_coverage", "write_frames_parquet", "write_manifest", "write_summary",
]
