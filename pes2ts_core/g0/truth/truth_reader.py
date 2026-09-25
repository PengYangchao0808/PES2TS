"""Audited accessors for the quarantined ground-truth artifacts.

This module is the only sanctioned way to read the transition-state geometries
and IRC trajectories.  Every accessor:

* requires ``allow_truth=True`` and raises :class:`PermissionError` otherwise
  (naming the reaction), so an accidental call from a path-generation module
  fails loudly before any file is touched;
* resolves data paths from ``truth_manifest.json`` (never from literals);
* appends an entry to ``truth_access_log.jsonl`` after a successful read, so
  every use of the ground truth is auditable.

IRC frames are read for exactly one reaction at a time; the index returned by
:func:`load_irc_index` contains no trajectory data.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pyarrow.parquet as pq

from pes2ts_core.g0.ids import normalize_reaction_id
from pes2ts_core.g0.irc_reader import IrcFrames, read_irc_frames_at
from pes2ts_core.utils.hashing import JSONValue, stable_json_dumps
from pes2ts_core.utils.jsonio import read_json

logger = logging.getLogger(__name__)

#: Manifest written by the quarantine stage; the only path source of truth.
TRUTH_MANIFEST_FILENAME: Final[str] = "truth_manifest.json"
#: Append-only audit log of every successful truth read.
TRUTH_ACCESS_LOG_FILENAME: Final[str] = "truth_access_log.jsonl"
#: Fallback manifests directory when the caller does not supply one.
DEFAULT_MANIFESTS_DIR: Final[str] = "data/manifests"
#: Suffix identifying the IRC source entry inside the manifest.
IRC_FILENAME_SUFFIX: Final[str] = "_IRC.h5"
#: Permission message required by the isolation contract.
PERMISSION_MESSAGE: Final[str] = (
    "ground-truth access requires allow_truth=True; this is audited"
)


def _require_allow_truth(reaction_id: object, allow_truth: bool) -> None:
    """Raise :class:`PermissionError` unless *allow_truth* is explicitly true."""
    if not allow_truth:
        raise PermissionError(f"{PERMISSION_MESSAGE} (reaction_id={reaction_id!r})")


def _manifests_dir(manifests_dir: str | Path | None) -> Path:
    """Resolve the manifests directory, defaulting to the configured layout."""
    return Path(manifests_dir) if manifests_dir is not None else Path(DEFAULT_MANIFESTS_DIR)


def _load_manifest(manifests_dir: str | Path | None) -> dict[str, Any]:
    """Read ``truth_manifest.json`` or fail with an actionable message."""
    path = _manifests_dir(manifests_dir) / TRUTH_MANIFEST_FILENAME
    if not path.is_file():
        msg = (
            f"Truth manifest not found: {path}; run 'pes2ts g0 quarantine' first"
        )
        raise FileNotFoundError(msg)
    document = read_json(path)
    if not isinstance(document, dict):
        msg = f"Truth manifest is not a JSON object: {path}"
        raise ValueError(msg)
    return document


def _entry(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    """Return the artifact entry *key* of *manifest* as a mapping."""
    entry = manifest.get(key)
    if not isinstance(entry, dict):
        msg = f"Truth manifest is missing the {key!r} entry"
        raise ValueError(msg)
    return entry


def _ts_path(manifest: dict[str, Any]) -> Path:
    """Return the relocated ``ts.parquet`` path recorded in *manifest*."""
    return Path(str(_entry(manifest, "ts_parquet")["path"]))


def _irc_index_path(manifest: dict[str, Any]) -> Path:
    """Return the recorded ``irc_index.parquet`` path."""
    return Path(str(_entry(manifest, "irc_index")["path"]))


def _irc_source_path(manifest: dict[str, Any]) -> Path:
    """Return the relocated IRC HDF5 path recorded in the source list."""
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        msg = "Truth manifest is missing the 'sources' list"
        raise ValueError(msg)
    for source in sources:
        if (
            isinstance(source, dict)
            and str(source.get("filename", "")).endswith(IRC_FILENAME_SUFFIX)
        ):
            return Path(str(source["relocated_path"]))
    msg = f"Truth manifest has no source ending in {IRC_FILENAME_SUFFIX!r}"
    raise ValueError(msg)


def _audit(manifests_dir: str | Path | None, function: str, reaction_id: str) -> None:
    """Append one audit line for a successful read (plain append, never rewrite)."""
    directory = _manifests_dir(manifests_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record: dict[str, JSONValue] = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "accessor_module": __name__,
        "function": function,
        "reaction_id": reaction_id,
    }
    with (directory / TRUTH_ACCESS_LOG_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(stable_json_dumps(record) + "\n")


def _frames_to_dict(frames: IrcFrames) -> dict[str, Any]:
    """Return the public dict view of one reaction's IRC frames."""
    return {
        "reaction_id": frames.reaction_id,
        "n_atoms": frames.n_atoms,
        "n_frames": frames.n_frames,
        "has_forces": frames.has_forces,
        "coordinates": frames.coordinates,
        "EHG": frames.ehg,
    }


def load_ts_geometry(
    reaction_id: str,
    allow_truth: bool = False,
    *,
    manifests_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return the quarantined TS geometry row for *reaction_id*.

    The row carries ``reaction_id``, ``atomic_numbers``, ``coordinates``,
    ``EHG``, ``charge``, ``multiplicity``, and ``reaction_smiles``.

    Raises
    ------
    PermissionError
        When *allow_truth* is not ``True``; the message names the reaction.
    KeyError
        When the reaction has no TS row in ``ts.parquet``.
    """
    _require_allow_truth(reaction_id, allow_truth)
    manifest = _load_manifest(manifests_dir)
    normalized = normalize_reaction_id(reaction_id)
    table = pq.read_table(_ts_path(manifest), filters=[("reaction_id", "=", normalized)])
    rows = table.to_pylist()
    if not rows:
        msg = f"Reaction {normalized} has no TS geometry in the quarantined table"
        raise KeyError(msg)
    _audit(manifests_dir, "load_ts_geometry", normalized)
    return rows[0]


def load_irc_frames(
    reaction_id: str,
    allow_truth: bool = False,
    *,
    manifests_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return the IRC trajectory frames of one reaction.

    The returned mapping has ``reaction_id``, ``n_atoms``, ``n_frames``,
    ``has_forces``, ``coordinates`` (``(n_frames, n_atoms, 3)``), and ``EHG``
    (``(n_frames, 3)`` or ``None`` when the archive stores no per-frame
    energies).

    Raises
    ------
    PermissionError
        When *allow_truth* is not ``True``; the message names the reaction.
    KeyError
        When the reaction is absent from the IRC archive.
    """
    _require_allow_truth(reaction_id, allow_truth)
    manifest = _load_manifest(manifests_dir)
    normalized = normalize_reaction_id(reaction_id)
    frames = read_irc_frames_at(_irc_source_path(manifest), normalized)
    if frames is None:
        msg = f"Reaction {normalized} is not present in the quarantined IRC archive"
        raise KeyError(msg)
    _audit(manifests_dir, "load_irc_frames", normalized)
    return _frames_to_dict(frames)


def load_irc_index(
    allow_truth: bool = False,
    *,
    manifests_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the shape-only IRC index rows (no trajectory data).

    Raises
    ------
    PermissionError
        When *allow_truth* is not ``True``.
    """
    _require_allow_truth("<index>", allow_truth)
    manifest = _load_manifest(manifests_dir)
    rows = pq.read_table(_irc_index_path(manifest)).to_pylist()
    _audit(manifests_dir, "load_irc_index", "<index>")
    return rows


__all__ = [
    "DEFAULT_MANIFESTS_DIR",
    "IRC_FILENAME_SUFFIX",
    "PERMISSION_MESSAGE",
    "TRUTH_ACCESS_LOG_FILENAME",
    "TRUTH_MANIFEST_FILENAME",
    "load_irc_frames",
    "load_irc_index",
    "load_ts_geometry",
]
