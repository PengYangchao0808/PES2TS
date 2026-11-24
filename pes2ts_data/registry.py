"""Datastore registry contract (Zone 2).

The unified datastore exposes a single version-lock document,
``01_master_index/datastore_registry.json``, written by
the frozen ``register_datastore.py`` (tag ``archive/data-builders-20261124``).  This module is the read/validate
side of that contract: it parses the registry into a frozen pydantic object and
enforces the machine-checkable invariants (schema id, ``status`` vocabulary,
artifact sha256 shape, positive row counts) with typed errors.

Constitution mapping: P1 (versioned interface -- the registry schema id is
pinned), R5 (row-count/cost accounting is retained and checked) and P2 (typed
honest rejection via :class:`RegistryError`).

The real registry lives on the WSL datastore root; this module only reads the
path explicitly handed to :func:`load_registry`, so tests use ``tmp_path``
fixtures and never touch real data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

__all__ = [
    "REGISTRY_SCHEMA_ID",
    "REGISTRY_STATUSES",
    "SHA256_RE",
    "DatastoreEntry",
    "DatastoreRegistry",
    "RegistryEntry",
    "RegistryError",
    "iter_artifacts",
    "load_registry",
    "validate_registry",
]

#: Schema id of the datastore registry document.
REGISTRY_SCHEMA_ID = "pes2ts_datastore_registry_v1"

#: Allowed lifecycle states of the datastore (``ACTIVE-BUILD`` is intentionally
#: not accepted: a registry handed to training must be fully built).
REGISTRY_STATUSES = frozenset({"ACTIVE", "FROZEN"})

#: Well-formed sha256: exactly 64 hexadecimal characters.
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class RegistryError(Exception):
    """Typed registry rejection (P2).

    Stable codes: ``REGISTRY_FILE_MISSING``, ``REGISTRY_PARSE_ERROR``,
    ``REGISTRY_MALFORMED``, ``REGISTRY_SCHEMA_UNSUPPORTED``,
    ``REGISTRY_BAD_STATUS``, ``REGISTRY_BAD_SHA256``,
    ``REGISTRY_BAD_ROW_COUNT``, ``REGISTRY_BAD_ARGUMENT``.
    """

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


class DatastoreEntry(BaseModel):
    """One artifact (or pointer) recorded in the datastore registry.

    Fields mirror the ``comp()`` helper of
    the frozen ``register_datastore.py`` (tag ``archive/data-builders-20261124``); extra keys such as ``note``
    or ``campaigns`` are preserved (``extra="allow"``) so the registry can grow
    without breaking readers.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    path: str = Field(min_length=1)
    exists: bool = False
    required: bool = False
    bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = None

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.match(str(value)):
            raise ValueError(
                f"sha256 must be 64 hex chars, got {value!r}")
        return value

    def is_verified(self) -> bool:
        """Return ``True`` when the artifact is declared present with a valid hash."""
        return bool(self.exists and self.sha256)


#: ``RegistryEntry`` is the historical name for a single registry artifact entry.
RegistryEntry = DatastoreEntry


def _walk_components(
    name: str,
    value: Any,
    artifacts: dict[str, DatastoreEntry],
) -> None:
    """Recursively collect artifact entries from the registry ``components`` tree."""
    if not isinstance(value, Mapping):
        return
    if isinstance(value.get("path"), str) and value.get("path"):
        payload = dict(value)
        payload["path"] = value["path"]
        artifacts[name] = DatastoreEntry.model_validate(payload)
        return
    children = {key: item for key, item in value.items() if isinstance(item, Mapping)}
    if children:
        for child_name, child in children.items():
            _walk_components(f"{name}.{child_name}", child, artifacts)
        return
    # leaf node keyed by path: {"02_core_tensors/...": {"bytes": ..., "sha256": ...}}
    if "sha256" in value or "bytes" in value:
        artifacts[name] = DatastoreEntry.model_validate({
            "path": name,
            "exists": True,
            "bytes": value.get("bytes"),
            "sha256": value.get("sha256"),
        })


def iter_artifacts(components: Mapping[str, Any] | None) -> dict[str, DatastoreEntry]:
    """Return ``dotted-name -> DatastoreEntry`` for every artifact in ``components``."""
    artifacts: dict[str, DatastoreEntry] = {}
    for name, value in dict(components or {}).items():
        _walk_components(str(name), value, artifacts)
    return artifacts


def _row_counts_from_doc(doc: Mapping[str, Any]) -> dict[str, int]:
    """Extract the positive integer row-count table from a raw registry document."""
    candidates: list[Mapping[str, Any]] = []
    for key in ("row_counts", "channel_row_counts"):
        value = doc.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    components = doc.get("components")
    if isinstance(components, Mapping):
        for key in ("row_counts", "channel_row_counts"):
            value = components.get(key)
            if isinstance(value, Mapping):
                candidates.append(value)

    rows: dict[str, int] = {}
    for table in candidates:
        for name, value in table.items():
            if isinstance(value, bool):
                raise RegistryError(
                    "REGISTRY_BAD_ROW_COUNT",
                    f"row count {name!r} must be a positive integer, got {value!r}")
            if isinstance(value, int):
                if value <= 0:
                    raise RegistryError(
                        "REGISTRY_BAD_ROW_COUNT",
                        f"row count {name!r} must be positive, got {value}")
                rows[str(name)] = value
            # Non-integer entries (None = unavailable, shape lists, "expected"
            # notes) are not row counts; None is honest unavailability, not zero.
    return rows


class DatastoreRegistry(BaseModel):
    """Parsed ``datastore_registry.json`` version-lock document."""

    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)

    schema_id: str = Field(alias="schema")
    status: str
    generated_at: str | None = None
    root: str | None = None
    git_commit: str | None = None
    components: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_id")
    @classmethod
    def _check_schema(cls, value: str) -> str:
        if value != REGISTRY_SCHEMA_ID:
            raise ValueError(
                f"unsupported registry schema {value!r}; expected "
                f"{REGISTRY_SCHEMA_ID!r}")
        return value

    @field_validator("status")
    @classmethod
    def _check_status(cls, value: str) -> str:
        if value not in REGISTRY_STATUSES:
            raise ValueError(
                f"status {value!r} not in {sorted(REGISTRY_STATUSES)}")
        return value

    @model_validator(mode="after")
    def _check_artifacts(self) -> "DatastoreRegistry":
        # Parse every artifact so a malformed sha256 is rejected at model level.
        iter_artifacts(self.components)
        return self

    def artifacts(self) -> dict[str, DatastoreEntry]:
        """Return ``dotted-name -> DatastoreEntry`` for all recorded artifacts."""
        return iter_artifacts(self.components)

    def row_counts(self) -> dict[str, int]:
        """Return the positive integer row counts recorded by the registry."""
        doc: dict[str, Any] = {"components": self.components}
        for key in ("row_counts", "channel_row_counts"):
            value = self.model_extra.get(key) if self.model_extra else None
            if isinstance(value, Mapping):
                doc[key] = value
        return _row_counts_from_doc(doc)


def _as_document(doc: Any) -> dict[str, Any]:
    """Normalise a registry document (raw mapping or model) to a plain dict."""
    if isinstance(doc, DatastoreRegistry):
        document = doc.model_dump(by_alias=True)
    elif isinstance(doc, Mapping):
        document = dict(doc)
    else:
        raise RegistryError(
            "REGISTRY_BAD_ARGUMENT",
            f"registry document must be a mapping, got {type(doc).__name__!r}")
    if "schema" not in document and "schema_id" in document:
        document["schema"] = document.pop("schema_id")
    return document


def validate_registry(doc: Any) -> dict[str, Any]:
    """Validate a registry document and return its normalised summary.

    Checks (each with a stable :class:`RegistryError` code):

    * the document is a mapping (``REGISTRY_BAD_ARGUMENT``);
    * ``schema`` equals :data:`REGISTRY_SCHEMA_ID` (``REGISTRY_SCHEMA_UNSUPPORTED``);
    * ``status`` is in :data:`REGISTRY_STATUSES` (``REGISTRY_BAD_STATUS``);
    * every recorded artifact sha256 is 64 hex chars
      (``REGISTRY_MALFORMED`` -- the pydantic validator raises it);
    * every recorded row count is a positive integer
      (``REGISTRY_BAD_ROW_COUNT``).

    Returns
    -------
    dict
        ``{"schema", "status", "generated_at", "root", "n_artifacts",
        "n_present", "n_missing_required", "row_counts", "artifacts"}``.
    """
    document = _as_document(doc)
    schema = document.get("schema")
    if schema != REGISTRY_SCHEMA_ID:
        raise RegistryError(
            "REGISTRY_SCHEMA_UNSUPPORTED",
            f"expected schema {REGISTRY_SCHEMA_ID!r}, got {schema!r}")
    status = document.get("status")
    if status not in REGISTRY_STATUSES:
        raise RegistryError(
            "REGISTRY_BAD_STATUS",
            f"status must be one of {sorted(REGISTRY_STATUSES)}, got {status!r}")

    try:
        registry = DatastoreRegistry.model_validate(document)
    except ValidationError as exc:
        message = str(exc)
        code = ("REGISTRY_BAD_SHA256" if "sha256" in message
                else "REGISTRY_MALFORMED")
        raise RegistryError(code, message) from exc

    artifacts = registry.artifacts()
    row_counts = _row_counts_from_doc(document)
    missing_required = sorted(
        name for name, entry in artifacts.items()
        if entry.required and not entry.exists)
    return {
        "schema": registry.schema_id,
        "status": registry.status,
        "generated_at": registry.generated_at,
        "root": registry.root,
        "git_commit": registry.git_commit,
        "n_artifacts": len(artifacts),
        "n_present": sum(1 for entry in artifacts.values() if entry.exists),
        "n_missing_required": missing_required,
        "row_counts": row_counts,
        "artifacts": {
            name: {
                "path": entry.path,
                "exists": entry.exists,
                "bytes": entry.bytes,
                "sha256": entry.sha256,
                "required": entry.required,
            }
            for name, entry in artifacts.items()
        },
    }


def load_registry(path: str | Path) -> DatastoreRegistry:
    """Load and validate ``datastore_registry.json`` from ``path``.

    Raises
    ------
    RegistryError
        ``REGISTRY_FILE_MISSING`` / ``REGISTRY_PARSE_ERROR`` /
        ``REGISTRY_SCHEMA_UNSUPPORTED`` / ``REGISTRY_BAD_STATUS`` /
        ``REGISTRY_MALFORMED``; never a bare ``FileNotFoundError`` /
        ``json.JSONDecodeError`` / ``pydantic.ValidationError``.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RegistryError(
            "REGISTRY_FILE_MISSING", f"registry file not found: {target}") from exc
    except OSError as exc:
        raise RegistryError(
            "REGISTRY_PARSE_ERROR", f"cannot read registry {target}: {exc}") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegistryError(
            "REGISTRY_PARSE_ERROR", f"invalid JSON in {target}: {exc}") from exc

    validate_registry(document)
    try:
        return DatastoreRegistry.model_validate(document)
    except ValidationError as exc:  # pragma: no cover - validate_registry covers
        raise RegistryError("REGISTRY_MALFORMED", str(exc)) from exc
