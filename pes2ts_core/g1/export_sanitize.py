"""Non-destructive migration of legacy G1 v2 exports to a truth-clean tree."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pes2ts_core.g1.reaction_case import ALLOWED_EXPORT_FIELDS
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.utils.hashing import sha256_bytes, stable_json_dumps
from pes2ts_core.utils.jsonio import write_json

SANITIZE_SCHEMA = "g1_v2_export_sanitize_v1"
SANITIZE_MANIFEST = "g1_v2_export_sanitize_manifest.json"


class ExportSanitizeError(ValueError):
    """A legacy export cannot be safely migrated without data loss."""


def _count_forbidden(value: Any, counts: Counter[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_EXPORT_KEYS:
                counts[key] += 1
            _count_forbidden(child, counts)
    elif isinstance(value, list):
        for child in value:
            _count_forbidden(child, counts)


def _remove_forbidden(value: Any, counts: Counter[str]) -> Any:
    if isinstance(value, dict):
        clean = {}
        for key, child in value.items():
            if key in FORBIDDEN_EXPORT_KEYS:
                counts[key] += 1
            else:
                clean[key] = _remove_forbidden(child, counts)
        return clean
    if isinstance(value, list):
        return [_remove_forbidden(child, counts) for child in value]
    return value


def sanitize_export_document(document: dict[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    """Apply a positive root whitelist and remove evaluation-only fields."""
    unknown = set(document) - ALLOWED_EXPORT_FIELDS
    if unknown:
        raise ExportSanitizeError(f"unknown export fields: {sorted(unknown)}")
    counts: Counter[str] = Counter()
    sanitized = _remove_forbidden(document, counts)
    return sanitized, counts


def migrate_export_tree(source_root: str | Path, output_root: str | Path, *,
                        expected_reaction_ids: set[str] | None = None,
                        resume_staging: bool = False) -> dict[str, Any]:
    """Create an immutable sibling export tree; never overwrites source/output.

    The output manifest records source/output tree digests and removed-key
    counts. The full per-reaction JSON documents remain the authoritative
    content; no TS/IRC geometry or labels are copied into the new tree.
    """
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"G1 export root not found: {source}")
    if source == output or source in output.parents:
        raise ExportSanitizeError("output must be separate from and outside the source export tree")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing sanitized export tree: {output}")
    paths = sorted(source.rglob("*.json"))
    if not paths:
        raise ExportSanitizeError("source export tree is empty")
    if expected_reaction_ids is not None:
        found = {path.stem for path in paths}
        missing, extra = expected_reaction_ids - found, found - expected_reaction_ids
        if missing or extra:
            raise ExportSanitizeError(f"scan-ready set mismatch: missing={len(missing)} extra={len(extra)}")

    staging = output.with_name(output.name + ".staging")
    if staging.exists() and not resume_staging:
        raise FileExistsError(f"staging path exists; pass resume_staging=True only after verifying it is this migration: {staging}")
    staging.mkdir(parents=True, exist_ok=True)
    removed: Counter[str] = Counter()
    source_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    ids_hash = hashlib.sha256()
    n_bytes = 0
    try:
        for path in paths:
            relative = path.relative_to(source)
            raw = path.read_bytes()
            source_hash.update(relative.as_posix().encode("utf-8") + b"\0" + hashlib.sha256(raw).digest())
            try:
                document = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ExportSanitizeError(f"invalid JSON in {relative}: {exc}") from exc
            if not isinstance(document, dict) or path.stem != document.get("reaction_id"):
                raise ExportSanitizeError(f"reaction ID/path mismatch in {relative}")
            sanitized, counts = sanitize_export_document(document)
            removed.update(counts)
            payload = stable_json_dumps(sanitized).encode("utf-8")
            destination = staging / relative
            if not destination.is_file() or destination.read_bytes() != payload:
                destination.parent.mkdir(parents=True, exist_ok=True)
                # The private staging tree is not published until its entire
                # manifest is complete, so one fsync+replace per tiny export
                # would add no useful crash-safety to the atomic tree publish.
                destination.write_bytes(payload)
            output_hash.update(relative.as_posix().encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
            ids_hash.update((document["reaction_id"] + "\n").encode("utf-8"))
            n_bytes += len(payload)
        manifest = {
            "schema_version": SANITIZE_SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source_root_name": source.name,
            "output_root_name": output.name,
            "n_exports": len(paths),
            "n_bytes": n_bytes,
            "reaction_ids_sha256": ids_hash.hexdigest(),
            "source_tree_sha256": source_hash.hexdigest(),
            "sanitized_tree_sha256": output_hash.hexdigest(),
            "removed_truth_key_occurrences": dict(sorted(removed.items())),
            "source_retained": True,
        }
        write_json(staging / SANITIZE_MANIFEST, manifest)
        os.replace(staging, output)
    except Exception:
        # Keep a named staging tree as recovery evidence if migration fails;
        # it is never presented as the published output.
        raise
    return manifest
