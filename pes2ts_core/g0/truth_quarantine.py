"""Quarantine of the transition-state and IRC ground truth.

The combined Reaction-QM HDF5 mixes the reactant/product inventory share with
the answer-bearing ``TS`` species, so reading it from anywhere in the
path-generation tree is a leak.  :func:`quarantine_truth`:

1. extracts every ``TS`` species into ``ts.parquet`` under the configured
   ground-truth directory;
2. builds a shape-only IRC index (``irc_index.parquet``) — the trajectory
   frames themselves are never copied into a non-truth artifact;
3. relocates the truth-bearing HDF5 sources into the configured truth-source
   directory, so the raw tree no longer contains them;
4. writes ``truth_manifest.json`` with SHA-256 digests for the artifacts and
   the relocated sources.

The function is idempotent: when the manifest digests still match the
relocated sources and artifacts (and no raw copy reappeared), it verifies and
returns a skipped result; on any mismatch it re-extracts from the relocated
copies in place.  Only :mod:`pes2ts_core.g0.truth.truth_reader` may read the
resulting artifacts, and every such read is audited.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pes2ts_core.config_loader import ConfigError
from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.truth_extract import (
    IRC_INDEX_COLUMNS,
    TS_COLUMNS,
    build_irc_index_rows,
    extract_ts_rows,
    reaction_smiles_map,
    rows_to_columns,
)
from pes2ts_core.g0.truth_manifest import (
    TRUTH_MANIFEST_FILENAME,
    manifest_matches,
    manifest_row_count,
    relocate_source,
    source_record,
    utc_now_iso,
)
from pes2ts_core.utils.hashing import JSONValue, sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import write_parquet

logger = logging.getLogger(__name__)

#: Extracted transition-state geometries.
TS_PARQUET_FILENAME: Final[str] = "ts.parquet"
#: Shape-only index of the IRC trajectories.
IRC_INDEX_FILENAME: Final[str] = "irc_index.parquet"
#: CLI exit code for a failed quarantine (configuration errors use 2).
EXIT_QUARANTINE_ERROR: Final[int] = 4

_MAIN_H5_SUFFIX: Final[str] = "_TZVP.h5"
_IRC_H5_SUFFIX: Final[str] = "_IRC.h5"
_REACTION_INFO_SUFFIX: Final[str] = "_reaction_info.csv"


@dataclass(frozen=True, slots=True)
class QuarantineResult:
    """Outcome of :func:`quarantine_truth`."""

    skipped: bool
    ts_path: Path
    irc_index_path: Path
    manifest_path: Path
    dataset_version: str
    n_ts_rows: int
    n_irc_rows: int
    n_reactions_without_ts: int
    relocated: tuple[str, ...] = ()


def _configured_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    """Return ``(raw, ground_truth, truth_sources, manifests)`` from config."""
    paths = config["paths"]
    return (
        Path(paths["raw"]),
        Path(paths["ground_truth"]),
        Path(paths["truth_sources"]),
        Path(paths["manifests"]),
    )


def _locate_truth_filenames(
    config: Mapping[str, Any],
) -> tuple[str, str, str | None]:
    """Pick the truth-bearing and reaction-info filenames from the config keys.

    The extraction main file is the configured key ending in ``_TZVP.h5`` but
    not ``_IRC.h5``; the IRC archive ends in ``_IRC.h5``.  No filename literal
    is hardcoded, so a renamed source record keeps working.
    """
    keys = list(config["source"]["files"])
    main = next(
        (key for key in keys if key.endswith(_MAIN_H5_SUFFIX) and not key.endswith(_IRC_H5_SUFFIX)),
        None,
    )
    irc = next((key for key in keys if key.endswith(_IRC_H5_SUFFIX)), None)
    reaction_info = next(
        (key for key in keys if key.endswith(_REACTION_INFO_SUFFIX)), None
    )
    if main is None or irc is None:
        msg = (
            "Config source.files must declare the combined HDF5 "
            f"(*{_MAIN_H5_SUFFIX}) and the IRC HDF5 (*{_IRC_H5_SUFFIX}); "
            f"found keys: {keys}"
        )
        raise ConfigError(msg)
    return main, irc, reaction_info


def quarantine_truth(config: Mapping[str, Any]) -> QuarantineResult:
    """Quarantine the TS geometries and IRC index, then relocate the sources.

    The extraction runs from the raw combined HDF5 when present, else from the
    already-relocated copy (re-extraction after corruption).  Both truth-bearing
    HDF5 files are moved into ``config["paths"]["truth_sources"]``; the CSVs
    stay in the raw tree.
    """
    raw_dir, ground_truth, truth_sources, manifests = _configured_paths(config)
    main_name, irc_name, info_name = _locate_truth_filenames(config)
    main_raw = raw_dir / main_name
    irc_raw = raw_dir / irc_name
    main_truth = truth_sources / main_name
    irc_truth = truth_sources / irc_name
    ts_path = ground_truth / TS_PARQUET_FILENAME
    irc_index_path = ground_truth / IRC_INDEX_FILENAME
    manifest_path = manifests / TRUTH_MANIFEST_FILENAME

    if (
        not main_raw.exists()
        and not irc_raw.exists()
        and manifest_matches(manifest_path, ts_path, irc_index_path, main_truth, irc_truth)
    ):
        document = read_json(manifest_path)
        assert isinstance(document, dict)  # manifest_matches validated the shape
        n_without_ts = document.get("n_reactions_without_ts")
        logger.info("Truth quarantine already current; verified digests in %s", manifest_path)
        return QuarantineResult(
            skipped=True,
            ts_path=ts_path,
            irc_index_path=irc_index_path,
            manifest_path=manifest_path,
            dataset_version=dataset_version(config),
            n_ts_rows=manifest_row_count(document, "ts_parquet"),
            n_irc_rows=manifest_row_count(document, "irc_index"),
            n_reactions_without_ts=n_without_ts if isinstance(n_without_ts, int) else 0,
        )

    main_source = main_raw if main_raw.is_file() else main_truth
    irc_source = irc_raw if irc_raw.is_file() else irc_truth
    if not main_source.is_file():
        msg = f"Combined HDF5 not found in {raw_dir} or {truth_sources}: {main_name}"
        raise FileNotFoundError(msg)
    if not irc_source.is_file():
        msg = f"IRC HDF5 not found in {raw_dir} or {truth_sources}: {irc_name}"
        raise FileNotFoundError(msg)

    info_path = (raw_dir / info_name) if info_name is not None else None
    ts_rows, n_without_ts = extract_ts_rows(main_source, reaction_smiles_map(info_path))
    write_parquet(ts_path, rows_to_columns(ts_rows, TS_COLUMNS))
    logger.info("Wrote %d TS row(s) to %s", len(ts_rows), ts_path)

    irc_rows = build_irc_index_rows(irc_source)
    write_parquet(irc_index_path, rows_to_columns(irc_rows, IRC_INDEX_COLUMNS))
    logger.info("Wrote %d IRC index row(s) to %s", len(irc_rows), irc_index_path)

    relocated: list[str] = []
    if relocate_source(main_source, main_truth):
        relocated.append(main_name)
    if relocate_source(irc_source, irc_truth):
        relocated.append(irc_name)
    if relocated:
        logger.info("Relocated truth-bearing source(s) into %s: %s", truth_sources, relocated)

    manifest: dict[str, JSONValue] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "ts_parquet": {
            "path": str(ts_path),
            "sha256": sha256_file(ts_path),
            "n_rows": len(ts_rows),
        },
        "irc_index": {
            "path": str(irc_index_path),
            "sha256": sha256_file(irc_index_path),
            "n_rows": len(irc_rows),
        },
        "sources": [
            source_record(main_name, main_raw, main_truth),
            source_record(irc_name, irc_raw, irc_truth),
        ],
        "n_reactions_without_ts": n_without_ts,
        "generated_at": utc_now_iso(),
    }
    write_json(manifest_path, manifest)
    logger.info("Wrote truth manifest %s", manifest_path)

    return QuarantineResult(
        skipped=False,
        ts_path=ts_path,
        irc_index_path=irc_index_path,
        manifest_path=manifest_path,
        dataset_version=dataset_version(config),
        n_ts_rows=len(ts_rows),
        n_irc_rows=len(irc_rows),
        n_reactions_without_ts=n_without_ts,
        relocated=tuple(relocated),
    )


__all__ = [
    "EXIT_QUARANTINE_ERROR",
    "IRC_INDEX_COLUMNS",
    "IRC_INDEX_FILENAME",
    "QuarantineResult",
    "TRUTH_MANIFEST_FILENAME",
    "TS_COLUMNS",
    "TS_PARQUET_FILENAME",
    "quarantine_truth",
]
