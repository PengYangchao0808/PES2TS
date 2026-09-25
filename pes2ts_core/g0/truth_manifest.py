"""Manifest and relocation ledger for the quarantined artifacts.

Pure helpers behind the quarantine orchestration: move a truth-bearing source
across filesystems, record its digest, and verify that a manifest still
describes the quarantined artifacts byte for byte (the idempotency gate).
The module holds no path constants and no artifact filenames; callers supply
every path.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pes2ts_core.utils.hashing import JSONValue, sha256_file
from pes2ts_core.utils.jsonio import read_json

logger = logging.getLogger(__name__)

#: Manifest listing the quarantined artifacts and their digests.
TRUTH_MANIFEST_FILENAME: Final[str] = "truth_manifest.json"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def relocate_source(source: Path, destination: Path) -> bool:
    """Move *source* to *destination*, returning whether a move happened.

    ``shutil.move`` handles cross-device moves; a bare ``OSError`` is retried
    with an explicit copy + unlink so an exotic filesystem cannot lose the file.
    """
    if source.resolve() == destination.resolve():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(source), str(destination))
    except OSError:
        shutil.copy2(source, destination)
        source.unlink()
    return True


def source_record(
    filename: str, original: Path, relocated: Path
) -> dict[str, JSONValue]:
    """Return the manifest entry for one relocated truth-bearing source."""
    return {
        "filename": filename,
        "original_path": str(original),
        "relocated_path": str(relocated),
        "sha256": sha256_file(relocated),
        "size_bytes": relocated.stat().st_size,
    }


def source_sha256(manifest: Mapping[str, JSONValue], filename: str) -> JSONValue:
    """Return the recorded SHA-256 of source *filename*, or ``None``."""
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        return None
    for source in sources:
        if isinstance(source, dict) and source.get("filename") == filename:
            return source.get("sha256")
    return None


def artifact_sha256(manifest: Mapping[str, JSONValue], key: str) -> JSONValue:
    """Return the recorded SHA-256 of the artifact *key* (e.g. ``ts_parquet``)."""
    entry = manifest.get(key)
    if not isinstance(entry, dict):
        return None
    return entry.get("sha256")


def manifest_matches(
    manifest_path: Path,
    ts_path: Path,
    irc_index_path: Path,
    main_truth: Path,
    irc_truth: Path,
) -> bool:
    """Return whether the manifest digests still match the quarantined files."""
    if not manifest_path.is_file() or not main_truth.is_file() or not irc_truth.is_file():
        return False
    try:
        document = read_json(manifest_path)
    except (OSError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    checks: tuple[tuple[Path, JSONValue], ...] = (
        (ts_path, artifact_sha256(document, "ts_parquet")),
        (irc_index_path, artifact_sha256(document, "irc_index")),
        (main_truth, source_sha256(document, main_truth.name)),
        (irc_truth, source_sha256(document, irc_truth.name)),
    )
    for path, expected in checks:
        if not isinstance(expected, str) or not path.is_file():
            return False
        if sha256_file(path) != expected:
            return False
    return True


def manifest_row_count(document: Mapping[str, JSONValue], key: str) -> int:
    """Return the recorded ``n_rows`` of artifact *key* (0 when malformed)."""
    entry = document.get(key)
    if not isinstance(entry, dict):
        return 0
    n_rows = entry.get("n_rows")
    return int(n_rows) if isinstance(n_rows, int) else 0


__all__ = [
    "TRUTH_MANIFEST_FILENAME",
    "artifact_sha256",
    "manifest_matches",
    "manifest_row_count",
    "relocate_source",
    "source_record",
    "source_sha256",
    "utc_now_iso",
]
