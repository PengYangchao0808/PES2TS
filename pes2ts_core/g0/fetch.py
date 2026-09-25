"""Verified acquisition of the Reaction-QM source files from Zenodo.

:func:`fetch_sources` downloads exactly the files declared under
``config["source"]["files"]`` into ``config["paths"]["raw"]`` and skips any
file whose local MD5 already matches the trusted value.  Downloads stream to a
``<name>.part-<pid>`` sibling and are MD5-verified before :func:`os.replace`
publishes them, so a partial or corrupt transfer is never mistaken for a
complete artifact.  :func:`verify_sources` re-checks every MD5 (raising a
:class:`ChecksumMismatch` that lists all offenders) and returns each file's
SHA-256; the source manifest is written only once every file verified.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION
from pes2ts_core.utils.hashing import JSONValue, md5_file, sha256_file
from pes2ts_core.utils.jsonio import write_json

#: Manifest filename written under ``config["paths"]["manifests"]``.
SOURCE_MANIFEST_FILENAME: Final[str] = "source_manifest.json"
#: Download stream chunk size (1 MiB); the response body is never fully buffered.
DOWNLOAD_CHUNK_SIZE: Final[int] = 1024 * 1024
#: Per-request socket timeout for Zenodo transfers.
DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 60.0
#: User-Agent sent with every request.
USER_AGENT: Final[str] = "pes2ts/0.1.0"
#: CLI exit code for a checksum failure (configuration errors use 2).
EXIT_CHECKSUM_MISMATCH: Final[int] = 3

#: One mismatch as (filename, expected MD5, actual MD5 or None when missing).
Mismatch = tuple[str, str, str | None]


def _describe_mismatches(mismatches: Sequence[Mismatch]) -> str:
    """Render *mismatches* as a human-actionable one-line message."""
    if len(mismatches) == 1:
        name, expected, actual = mismatches[0]
        observed = actual if actual is not None else "file missing"
        return f"MD5 mismatch for {name}: expected {expected}, got {observed}"
    details = "; ".join(
        f"{name}: expected {expected}, got {actual if actual is not None else 'file missing'}"
        for name, expected, actual in mismatches
    )
    return f"MD5 mismatch for {len(mismatches)} files: {details}"


class ChecksumMismatch(Exception):
    """Raised when a source file is missing or fails its trusted MD5.

    Always names the offending file(s) together with the expected and observed
    MD5; :attr:`mismatches` carries every offender when several files were
    verified in one call.
    """

    def __init__(
        self,
        filename: str,
        expected: str,
        actual: str | None,
        *,
        additional: Sequence[Mismatch] = (),
    ) -> None:
        self.filename: str = filename
        self.expected: str = expected
        self.actual: str | None = actual
        self.mismatches: tuple[Mismatch, ...] = ((filename, expected, actual), *additional)
        super().__init__(_describe_mismatches(self.mismatches))


@dataclass(frozen=True, slots=True)
class SourceFileSpec:
    """One configured source artifact."""

    filename: str
    url: str
    md5: str


@dataclass(frozen=True, slots=True)
class FetchedSource:
    """Acquisition metadata for one source file, before SHA-256 is known.

    ``downloaded_at`` is ``None`` when a previously downloaded copy was reused.
    """

    filename: str
    url: str
    zenodo_md5: str
    size_bytes: int
    downloaded_at: str | None

    def to_record(self, sha256: str) -> dict[str, JSONValue]:
        """Return the manifest entry for this file given its SHA-256 digest."""
        return {
            "filename": self.filename,
            "url": self.url,
            "zenodo_md5": self.zenodo_md5,
            "sha256": sha256,
            "size_bytes": self.size_bytes,
            "downloaded_at": self.downloaded_at,
        }


def _source_specs(config: Mapping[str, Any]) -> list[SourceFileSpec]:
    """Return the configured source files in declaration order."""
    return [
        SourceFileSpec(
            filename=str(entry.get("filename", key)),
            url=str(entry["url"]),
            md5=str(entry["md5"]),
        )
        for key, entry in config["source"]["files"].items()
    ]


def _raw_dir(config: Mapping[str, Any]) -> Path:
    """Resolve the configured raw directory against the current working dir."""
    return Path(config["paths"]["raw"])


def source_manifest_path(config: Mapping[str, Any]) -> Path:
    """Return the path of ``source_manifest.json`` for *config*."""
    return Path(config["paths"]["manifests"]) / SOURCE_MANIFEST_FILENAME


def dataset_version(config: Mapping[str, Any]) -> str:
    """Return the ``zenodo-<record>-rev<revision>`` dataset version string."""
    source = config["source"]
    return f"zenodo-{source['zenodo_record']}-rev{source['zenodo_revision']}"


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _download_to_part(url: str, part: Path) -> None:
    """Stream *url* into *part* in 1 MiB chunks without buffering the body."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        with part.open("wb") as handle:
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())


def _fetch_one(spec: SourceFileSpec, target: Path) -> None:
    """Download *spec* to *target*, verifying MD5 before atomic publication."""
    logger = logging.getLogger(__name__)
    part = target.parent / f"{target.name}.part-{os.getpid()}"
    try:
        logger.info("%s: downloading %s", spec.filename, spec.url)
        _download_to_part(spec.url, part)
        actual = md5_file(part)
        if actual != spec.md5:
            raise ChecksumMismatch(spec.filename, spec.md5, actual)
        os.replace(part, target)
    finally:
        # A successful os.replace already consumed part; on any failure this
        # removes the residue so no stale .part file can be mistaken for data.
        part.unlink(missing_ok=True)


def fetch_sources(
    config: Mapping[str, Any], force: bool = False
) -> dict[str, FetchedSource]:
    """Ensure every configured source file exists and matches its trusted MD5.

    Files already present with the expected MD5 are reused unless *force* is
    true.  New downloads are MD5-verified before being published, so the call
    either returns a complete verified set or raises :class:`ChecksumMismatch`
    leaving no partial artifact behind.
    """
    logger = logging.getLogger(__name__)
    raw_dir = _raw_dir(config)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetched: dict[str, FetchedSource] = {}
    for spec in _source_specs(config):
        target = raw_dir / spec.filename
        if not force and target.is_file() and md5_file(target) == spec.md5:
            logger.info(
                "%s: verified (md5 %s), skipping download", spec.filename, spec.md5
            )
            downloaded_at = None
        else:
            _fetch_one(spec, target)
            logger.info("%s: downloaded and MD5-verified", spec.filename)
            downloaded_at = _utc_now_iso()
        fetched[spec.filename] = FetchedSource(
            filename=spec.filename,
            url=spec.url,
            zenodo_md5=spec.md5,
            size_bytes=target.stat().st_size,
            downloaded_at=downloaded_at,
        )
    return fetched


def verify_sources(config: Mapping[str, Any]) -> dict[str, str]:
    """Re-check every configured MD5 and return ``filename -> sha256``.

    All files are inspected before failing, so a :class:`ChecksumMismatch`
    reports every offender at once; downstream stages call this to prove the
    raw tree is intact before reading it.
    """
    logger = logging.getLogger(__name__)
    specs = _source_specs(config)
    raw_dir = _raw_dir(config)
    mismatches: list[Mismatch] = []
    for spec in specs:
        target = raw_dir / spec.filename
        actual = md5_file(target) if target.is_file() else None
        if actual != spec.md5:
            mismatches.append((spec.filename, spec.md5, actual))
    if mismatches:
        first, *rest = mismatches
        raise ChecksumMismatch(*first, additional=rest)
    digests = {spec.filename: sha256_file(raw_dir / spec.filename) for spec in specs}
    logger.info("Verified MD5 and SHA-256 for %d source file(s)", len(digests))
    return digests


def write_source_manifest(
    config: Mapping[str, Any],
    fetched: Mapping[str, FetchedSource],
    digests: Mapping[str, str],
) -> Path:
    """Atomically write the source manifest for a fully verified file set.

    Call only after :func:`fetch_sources` and :func:`verify_sources` succeeded:
    *fetched* supplies acquisition metadata and *digests* the SHA-256 values,
    so a manifest cannot be produced for unverified files.
    """
    logger = logging.getLogger(__name__)
    source = config["source"]
    files: list[JSONValue] = [
        fetched[spec.filename].to_record(digests[spec.filename])
        for spec in _source_specs(config)
    ]
    manifest: dict[str, JSONValue] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": dataset_version(config),
        "zenodo_doi": str(source["zenodo_doi"]),
        "zenodo_revision": str(source["zenodo_revision"]),
        "zenodo_modified": str(source["zenodo_modified"]),
        "files": files,
    }
    path = source_manifest_path(config)
    write_json(path, manifest)
    logger.info("Wrote source manifest %s (%d file(s))", path, len(files))
    return path


__all__ = [
    "DOWNLOAD_CHUNK_SIZE",
    "EXIT_CHECKSUM_MISMATCH",
    "SOURCE_MANIFEST_FILENAME",
    "ChecksumMismatch",
    "FetchedSource",
    "SourceFileSpec",
    "dataset_version",
    "fetch_sources",
    "source_manifest_path",
    "verify_sources",
    "write_source_manifest",
]
