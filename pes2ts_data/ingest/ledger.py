"""Tombstone / cost ledger reader for the retired data campaigns (Zone 2).

The retired raw packs and historical campaign directories were deleted during the
2026-11-23 datastore consolidation.  Their audit trail lives in two JSON files::

    data/archives_summary/campaigns_index.json      # campaign tombstones index
    data/archives_summary/retired_raw_packs.json    # retired-pack cost ledger

This module reads those files for *any* root (tests use ``tmp_path`` fixtures;
the real repo data is never touched by the code itself) and re-verifies sha256
hashes when an artifact still exists.  It is read-only: nothing here deletes,
moves or writes.

Constitution mapping: R5 (full-cost retention -- every retired artifact keeps its
recorded cost) and P2 (typed honest rejection -- :class:`LedgerError` with a
stable ``.code``; a missing artifact is reported as ``MISSING``, never raised as
``FileNotFoundError``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "CAMPAIGN_INDEX_SCHEMA",
    "CAMPAIGN_INDEX_RELPATH",
    "RETIRED_LEDGER_SCHEMA",
    "RETIRED_LEDGER_RELPATH",
    "CampaignEntry",
    "CampaignIndex",
    "Ledger",
    "LedgerError",
    "RetiredEntry",
    "Verification",
    "VerifyStatus",
    "load_campaign_index",
    "load_ledger",
    "resolve_artifact_path",
    "sha256_file",
    "summarize",
    "verify_entry",
]

#: Expected schema ids of the two ledger documents.
CAMPAIGN_INDEX_SCHEMA = "pes2ts_campaign_archive_index_v1"
RETIRED_LEDGER_SCHEMA = "pes2ts_retired_packs_ledger_v1"

#: Canonical repo-relative locations.
CAMPAIGN_INDEX_RELPATH = "data/archives_summary/campaigns_index.json"
RETIRED_LEDGER_RELPATH = "data/archives_summary/retired_raw_packs.json"

_CHUNK = 1 << 20


class LedgerError(Exception):
    """Typed ledger-read rejection (P2).

    Stable codes: ``LEDGER_FILE_MISSING``, ``LEDGER_PARSE_ERROR``,
    ``LEDGER_SCHEMA_UNSUPPORTED``, ``LEDGER_MALFORMED``, ``LEDGER_BAD_ARGUMENT``.
    """

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


class VerifyStatus(str, Enum):
    """Verification outcome for one ledger entry.

    ``OK``
        The artifact exists and the recorded sha256 matches.
    ``MISSING``
        The artifact no longer exists (expected for retired packs).
    ``MISMATCH``
        The artifact exists but its sha256 differs from the ledger.
    ``UNVERIFIABLE``
        The artifact exists but the entry records no sha256 (or points at a
        directory), so integrity cannot be re-proven from the ledger.
    """

    OK = "OK"
    MISSING = "MISSING"
    MISMATCH = "MISMATCH"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass(frozen=True)
class CampaignEntry:
    """One campaign tombstone row from ``campaigns_index.json``."""

    campaign_id: str
    bytes: int = 0
    files: int = 0
    tombstone: str | None = None
    tombstone_exists: bool = False
    extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "campaign_id": self.campaign_id,
            "bytes": self.bytes,
            "files": self.files,
            "tombstone": self.tombstone,
            "tombstone_exists": self.tombstone_exists,
        }


@dataclass(frozen=True)
class CampaignIndex:
    """Parsed ``campaigns_index.json``."""

    schema: str
    generated_at: str | None
    archives_dir: str | None
    campaigns: tuple[CampaignEntry, ...]
    total_bytes: int
    source_path: Path
    extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class RetiredEntry:
    """One retired artifact row from ``retired_raw_packs.json``."""

    kind: str
    path: str | None = None
    bytes: int | None = None
    files: int | None = None
    gib: float | None = None
    sha256: str | None = None
    verified_before_delete: bool | None = None
    tombstone: str | None = None
    campaign_id: str | None = None
    deleted_at: str | None = None
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def artifact(self) -> str | None:
        """The path to verify: the retired artifact itself or its tombstone."""
        return self.path if self.path is not None else self.tombstone


@dataclass(frozen=True)
class Ledger:
    """Parsed ``retired_raw_packs.json``."""

    schema: str
    generated_at: str | None
    policy: str | None
    entries: tuple[RetiredEntry, ...]
    source_path: Path
    extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class Verification:
    """Result of re-verifying one ledger entry against the filesystem."""

    entry: RetiredEntry | CampaignEntry
    status: VerifyStatus
    target: Path | None
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    detail: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    """Return the lowercase hex sha256 of ``path`` (streaming, 1 MiB chunks)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact_path(root: str | Path, relpath: str | None) -> Path | None:
    """Resolve a ledger path against ``root`` (repo root or archive dir).

    Ledger paths are stored with Windows backslashes or forward slashes and may
    be absolute.  Relative paths are resolved against ``root``; absolute paths
    are returned unchanged (a WSL path such as ``/pes2ts_data/...`` simply does
    not exist from Windows, and is reported as ``MISSING`` by
    :func:`verify_entry`).
    """
    if not relpath:
        return None
    candidate = Path(str(relpath).replace("\\", "/"))
    if candidate.is_absolute():
        return candidate
    return Path(root) / candidate


def _candidate_paths(root: Path, relpath: str) -> list[Path]:
    """Return the possible locations of a ledger file for a flexible root."""
    name = Path(relpath).name
    return [
        root / relpath,                          # root == repo root
        root / Path(relpath).relative_to("data"),  # root == data/
        root / name,                             # root == archives_summary/
    ]


def _locate(root: str | Path, relpath: str, what: str) -> Path:
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        raise LedgerError(
            "LEDGER_BAD_ARGUMENT", f"root {root_path} is not an existing directory")
    for candidate in _candidate_paths(root_path, relpath):
        if candidate.is_file():
            return candidate
    raise LedgerError(
        "LEDGER_FILE_MISSING",
        f"{what} not found under {root_path} (looked for {relpath})")


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LedgerError("LEDGER_PARSE_ERROR", f"cannot read {what} {path}: {exc}") from exc
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LedgerError(
            "LEDGER_PARSE_ERROR", f"invalid JSON in {what} {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise LedgerError(
            "LEDGER_MALFORMED", f"{what} {path} must contain a JSON object")
    return doc


def _check_schema(doc: Mapping[str, Any], expected: str, path: Path) -> str:
    schema = doc.get("schema")
    if schema != expected:
        raise LedgerError(
            "LEDGER_SCHEMA_UNSUPPORTED",
            f"{path} has schema {schema!r}; expected {expected!r}")
    return str(schema)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_campaign_index(root: str | Path) -> CampaignIndex:
    """Load ``campaigns_index.json`` for an arbitrary root.

    ``root`` may be the repo root, the ``data/`` directory or the
    ``data/archives_summary/`` directory; all three layouts are accepted so
    tests can use a ``tmp_path`` fixture.

    Raises
    ------
    LedgerError
        ``LEDGER_BAD_ARGUMENT`` / ``LEDGER_FILE_MISSING`` /
        ``LEDGER_PARSE_ERROR`` / ``LEDGER_SCHEMA_UNSUPPORTED`` /
        ``LEDGER_MALFORMED``.
    """
    path = _locate(root, CAMPAIGN_INDEX_RELPATH, "campaign index")
    doc = _read_json(path, "campaign index")
    schema = _check_schema(doc, CAMPAIGN_INDEX_SCHEMA, path)

    raw_campaigns = doc.get("campaigns")
    if not isinstance(raw_campaigns, list):
        raise LedgerError(
            "LEDGER_MALFORMED", f"{path}: 'campaigns' must be a list")
    entries: list[CampaignEntry] = []
    for index, row in enumerate(raw_campaigns):
        if not isinstance(row, Mapping):
            raise LedgerError(
                "LEDGER_MALFORMED", f"{path}: campaigns[{index}] is not an object")
        campaign_id = row.get("campaign_id")
        if not isinstance(campaign_id, str) or not campaign_id:
            raise LedgerError(
                "LEDGER_MALFORMED",
                f"{path}: campaigns[{index}] has no string campaign_id")
        tombstone = row.get("tombstone")
        tombstone_path = resolve_artifact_path(Path(root), tombstone)
        entries.append(CampaignEntry(
            campaign_id=campaign_id,
            bytes=_int_or_none(row.get("bytes")) or 0,
            files=_int_or_none(row.get("files")) or 0,
            tombstone=tombstone if isinstance(tombstone, str) else None,
            tombstone_exists=bool(tombstone_path and tombstone_path.is_file()),
            extra={key: value for key, value in row.items()
                   if key not in ("campaign_id", "bytes", "files", "tombstone")},
        ))

    total = _int_or_none(doc.get("total_bytes"))
    if total is None:
        total = sum(entry.bytes for entry in entries)
    return CampaignIndex(
        schema=schema,
        generated_at=doc.get("generated_at") if isinstance(doc.get("generated_at"), str) else None,
        archives_dir=doc.get("archives_dir") if isinstance(doc.get("archives_dir"), str) else None,
        campaigns=tuple(entries),
        total_bytes=total,
        source_path=path,
        extra={key: value for key, value in doc.items()
               if key not in ("schema", "generated_at", "archives_dir",
                              "campaigns", "total_bytes")},
    )


def load_ledger(root: str | Path) -> Ledger:
    """Load ``retired_raw_packs.json`` for an arbitrary root.

    Same root flexibility and typed errors as :func:`load_campaign_index`.
    """
    path = _locate(root, RETIRED_LEDGER_RELPATH, "retired ledger")
    doc = _read_json(path, "retired ledger")
    schema = _check_schema(doc, RETIRED_LEDGER_SCHEMA, path)

    raw_entries = doc.get("retired")
    if not isinstance(raw_entries, list):
        raise LedgerError(
            "LEDGER_MALFORMED", f"{path}: 'retired' must be a list")
    entries: list[RetiredEntry] = []
    for index, row in enumerate(raw_entries):
        if not isinstance(row, Mapping):
            raise LedgerError(
                "LEDGER_MALFORMED", f"{path}: retired[{index}] is not an object")
        kind = row.get("kind")
        if not isinstance(kind, str) or not kind:
            raise LedgerError(
                "LEDGER_MALFORMED", f"{path}: retired[{index}] has no string kind")
        entries.append(RetiredEntry(
            kind=kind,
            path=row.get("path") if isinstance(row.get("path"), str) else None,
            bytes=_int_or_none(row.get("bytes")),
            files=_int_or_none(row.get("files")),
            gib=_float_or_none(row.get("gib")),
            sha256=row.get("sha256") if isinstance(row.get("sha256"), str) else None,
            verified_before_delete=(row.get("verified_before_delete")
                                    if isinstance(row.get("verified_before_delete"), bool)
                                    else None),
            tombstone=(row.get("tombstone")
                       if isinstance(row.get("tombstone"), str) else None),
            campaign_id=(row.get("campaign_id")
                         if isinstance(row.get("campaign_id"), str) else None),
            deleted_at=(row.get("deleted_at")
                        if isinstance(row.get("deleted_at"), str) else None),
            reason=(row.get("reason") if isinstance(row.get("reason"), str) else None),
            extra={key: value for key, value in row.items()
                   if key not in ("kind", "path", "bytes", "files", "gib", "sha256",
                                  "verified_before_delete", "tombstone",
                                  "campaign_id", "deleted_at", "reason")},
        ))
    return Ledger(
        schema=schema,
        generated_at=doc.get("generated_at") if isinstance(doc.get("generated_at"), str) else None,
        policy=doc.get("policy") if isinstance(doc.get("policy"), str) else None,
        entries=tuple(entries),
        source_path=path,
        extra={key: value for key, value in doc.items()
               if key not in ("schema", "generated_at", "policy", "retired")},
    )


# ---------------------------------------------------------------------------
# Verification + summary
# ---------------------------------------------------------------------------

def verify_entry(root: str | Path, entry: RetiredEntry | CampaignEntry) -> Verification:
    """Re-verify the sha256 of one ledger entry when the artifact still exists.

    Never raises for a missing artifact: a missing file is reported as
    :attr:`VerifyStatus.MISSING`.  Raises :class:`LedgerError` only for a caller
    error (``LEDGER_BAD_ARGUMENT`` when ``entry`` has no artifact path at all is
    *not* raised -- that is reported as ``UNVERIFIABLE``).

    Returns
    -------
    Verification
        ``status`` is one of :class:`VerifyStatus`; ``target`` is the resolved
        path that was checked (``None`` when the entry records no path).
    """
    if not isinstance(entry, (RetiredEntry, CampaignEntry)):
        raise LedgerError(
            "LEDGER_BAD_ARGUMENT",
            f"entry must be RetiredEntry or CampaignEntry, got "
            f"{type(entry).__name__!r}")
    relpath = entry.artifact if isinstance(entry, RetiredEntry) else entry.tombstone
    target = resolve_artifact_path(root, relpath)
    expected = entry.sha256 if isinstance(entry, RetiredEntry) else None

    if target is None:
        return Verification(
            entry=entry, status=VerifyStatus.UNVERIFIABLE, target=None,
            detail="ledger entry records no artifact path (cost-only entry)")
    if not target.exists():
        return Verification(
            entry=entry, status=VerifyStatus.MISSING, target=target,
            expected_sha256=expected,
            detail="artifact no longer exists (retired); ledger record retained")
    if target.is_dir():
        return Verification(
            entry=entry, status=VerifyStatus.UNVERIFIABLE, target=target,
            expected_sha256=expected,
            detail="artifact is a directory; sha256 re-verification not applicable")
    if not expected:
        return Verification(
            entry=entry, status=VerifyStatus.UNVERIFIABLE, target=target,
            detail="artifact exists but the ledger records no sha256")

    actual = sha256_file(target)
    if actual.lower() == expected.lower():
        return Verification(
            entry=entry, status=VerifyStatus.OK, target=target,
            expected_sha256=expected, actual_sha256=actual,
            detail="sha256 matches ledger")
    return Verification(
        entry=entry, status=VerifyStatus.MISMATCH, target=target,
        expected_sha256=expected, actual_sha256=actual,
        detail="sha256 differs from ledger")


def summarize(root: str | Path) -> dict[str, Any]:
    """Return totals for the retired campaign/cost ledger under ``root``.

    The result reports cost retention (R5) and artifact verification status::

        {
          "root": "...",
          "campaigns": {"count", "total_bytes", "files", "tombstones_present",
                         "tombstones_missing", "by_campaign_id"},
          "retired": {"entries", "known_bytes", "known_gib", "by_kind",
                       "verified_before_delete"},
          "verification": {"OK": n, "MISSING": n, "MISMATCH": n,
                            "UNVERIFIABLE": n},
          "problems": [{"kind", "path", "status", "detail"}, ...],
          "sources": {"campaign_index": str, "retired_ledger": str},
        }
    """
    index = load_campaign_index(root)
    ledger = load_ledger(root)

    by_kind: dict[str, int] = {}
    known_bytes = 0
    known_gib = 0.0
    verified_before_delete = 0
    counts = {status.value: 0 for status in VerifyStatus}
    problems: list[dict[str, Any]] = []

    for entry in ledger.entries:
        by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
        if entry.bytes is not None:
            known_bytes += entry.bytes
        if entry.gib is not None:
            known_gib += entry.gib
        if entry.verified_before_delete:
            verified_before_delete += 1
        result = verify_entry(root, entry)
        counts[result.status.value] += 1
        if result.status in (VerifyStatus.MISMATCH,):
            problems.append({
                "kind": entry.kind,
                "path": entry.artifact,
                "status": result.status.value,
                "detail": result.detail,
            })

    for entry in index.campaigns:
        result = verify_entry(root, entry)
        if not entry.tombstone_exists:
            problems.append({
                "kind": "campaign_tombstone",
                "path": entry.tombstone,
                "status": result.status.value,
                "detail": "campaign tombstone missing",
            })

    return {
        "root": str(Path(root)),
        "campaigns": {
            "count": len(index.campaigns),
            "total_bytes": index.total_bytes,
            "files": sum(entry.files for entry in index.campaigns),
            "tombstones_present": sum(1 for entry in index.campaigns
                                      if entry.tombstone_exists),
            "tombstones_missing": sum(1 for entry in index.campaigns
                                      if not entry.tombstone_exists),
            "by_campaign_id": {entry.campaign_id: entry.bytes
                               for entry in index.campaigns},
        },
        "retired": {
            "entries": len(ledger.entries),
            "known_bytes": known_bytes,
            "known_gib": round(known_gib, 3),
            "by_kind": dict(sorted(by_kind.items())),
            "verified_before_delete": verified_before_delete,
        },
        "verification": counts,
        "problems": problems,
        "sources": {
            "campaign_index": str(index.source_path),
            "retired_ledger": str(ledger.source_path),
        },
    }
