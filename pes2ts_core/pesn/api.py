"""Runtime public entry points for the PES-N predictor.

Only two functions are public: :func:`load_bundle` returns a ready
:class:`~pes2ts_core.pesn.inference.predictor.Predictor`, and
:func:`describe_bundle` reads a bundle's manifest and capabilities without
requiring any ML backend.  Both import their implementation lazily so that
importing this module stays cheap and backend-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time cycle guard
    from pes2ts_core.pesn.inference.predictor import Predictor


def load_bundle(
    path: str | Path,
    *,
    expected_feature_spec: str | None = None,
) -> "Predictor":
    """Load a validated runtime model bundle and return a :class:`Predictor`.

    The reader verifies the checksum manifest, schema versions, capability
    table, and feature-spec compatibility before any weight is deserialized.
    ``expected_feature_spec`` (a feature-spec content hash) must match the
    bundle's frozen spec when supplied.
    """
    from pes2ts_core.pesn.inference.predictor import Predictor

    return Predictor.from_bundle(Path(path), expected_feature_spec=expected_feature_spec)


def describe_bundle(path: str | Path) -> dict[str, Any]:
    """Return manifest/capability metadata of a bundle without loading weights."""
    from pes2ts_core.pesn.bundle.reader import read_bundle_manifest

    manifest, bundle_root = read_bundle_manifest(Path(path))
    return {
        "bundle_root": str(bundle_root),
        "model_id": manifest.get("model_id"),
        "model_hash": manifest.get("model_hash"),
        "capabilities": dict(manifest.get("capabilities", {})),
        "training_run_ids": list(manifest.get("training_run_ids", [])),
    }


__all__ = ["describe_bundle", "load_bundle"]
