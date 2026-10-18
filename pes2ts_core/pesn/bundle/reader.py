"""Whitelist bundle reader with hash/schema/compatibility checks (B6).

A runtime bundle directory may contain exactly the whitelisted files (see
``BUNDLE_RUNTIME_FILES``).  The reader verifies, in order: the file set, the
manifest schema and capability table, and every file hash against
``checksums.json``.  Weight deserialization happens only after all checks
pass, and only when a backend is requested — a manifest-only read works
without torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping

from pes2ts_core.pesn.contracts.artifacts import (
    BUNDLE_RUNTIME_FILES,
    ArtifactError,
    validate_bundle_manifest,
)
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json

MANIFEST_NAME: Final[str] = "model_bundle.json"
CHECKSUMS_NAME: Final[str] = "checksums.json"
WEIGHTS_NAME: Final[str] = "weights.pt"

#: Bundle format identities the reader can load (v2 adds control heads).
SUPPORTED_BUNDLE_FORMAT: Final[str] = "pesn_bundle_format_v1"
SUPPORTED_BUNDLE_FORMAT_V2: Final[str] = "pesn_bundle_format_v2"
SUPPORTED_BUNDLE_FORMATS: Final[tuple[str, ...]] = (
    SUPPORTED_BUNDLE_FORMAT,
    SUPPORTED_BUNDLE_FORMAT_V2,
)


class BundleError(ArtifactError):
    """Typed bundle read/compat failure."""


def read_bundle_manifest(bundle_dir: str | Path) -> tuple[dict, Path]:
    """Validate file set + manifest + checksums; return (manifest, root)."""
    root = Path(bundle_dir)
    if not root.is_dir():
        raise BundleError("BUNDLE_MISSING", f"bundle directory missing: {root}")
    present = {entry.name for entry in root.iterdir() if entry.is_file()}
    extra = present - set(BUNDLE_RUNTIME_FILES)
    if extra:
        raise BundleError(
            "BUNDLE_FILE_NOT_WHITELISTED",
            f"bundle carries non-whitelisted files: {sorted(extra)}",
        )
    missing = {MANIFEST_NAME, CHECKSUMS_NAME, WEIGHTS_NAME} - present
    if missing:
        raise BundleError(
            "BUNDLE_FILE_MISSING", f"bundle lacks required files: {sorted(missing)}"
        )
    manifest = read_json(root / MANIFEST_NAME)
    if not isinstance(manifest, dict):
        raise BundleError("MANIFEST_MALFORMED", "bundle manifest is not an object")
    bundle_format = manifest.get("bundle_format")
    if bundle_format == SUPPORTED_BUNDLE_FORMAT:
        validate_bundle_manifest(manifest)
    elif bundle_format == SUPPORTED_BUNDLE_FORMAT_V2:
        from pes2ts_core.pesn.contracts.control import (
            validate_control_bundle_manifest,
        )

        validate_control_bundle_manifest(manifest)
    else:
        raise BundleError(
            "BUNDLE_FORMAT_UNKNOWN",
            f"bundle format {bundle_format!r} unsupported",
        )
    checksums = read_json(root / CHECKSUMS_NAME)
    if not isinstance(checksums, dict):
        raise BundleError("CHECKSUMS_MALFORMED", "checksums document is not an object")
    for filename in sorted(present - {CHECKSUMS_NAME}):
        expected = checksums.get(filename)
        if not isinstance(expected, str):
            raise BundleError(
                "CHECKSUM_MISSING", f"no checksum recorded for {filename}"
            )
        actual = sha256_file(root / filename)
        if actual != expected:
            raise BundleError(
                "CHECKSUM_MISMATCH", f"{filename}: hash {actual} != {expected}"
            )
    return manifest, root


def load_bundle_weights(bundle_dir: str | Path) -> dict:
    """Deserialize the whitelist weights (requires the ML backend)."""
    from pes2ts_core.pesn.models import BackendMissingError

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise BackendMissingError("torch", "bundle weight loading") from exc

    _, root = read_bundle_manifest(bundle_dir)
    return torch.load(root / WEIGHTS_NAME, map_location="cpu", weights_only=True)


def bundle_model_config(manifest: Mapping[str, object]) -> dict[str, int]:
    """Model constructor parameters recorded in the manifest."""
    section = manifest.get("model_config")
    if not isinstance(section, dict):
        raise BundleError("MANIFEST_MALFORMED", "bundle lacks model_config")
    return {
        "hidden": int(section["hidden"]),
        "n_layers": int(section["n_layers"]),
        "n_basis": int(section["n_basis"]),
    }


__all__ = [
    "CHECKSUMS_NAME",
    "MANIFEST_NAME",
    "SUPPORTED_BUNDLE_FORMAT",
    "SUPPORTED_BUNDLE_FORMATS",
    "SUPPORTED_BUNDLE_FORMAT_V2",
    "WEIGHTS_NAME",
    "BundleError",
    "bundle_model_config",
    "load_bundle_weights",
    "read_bundle_manifest",
]
