"""Internal R1 gate machinery behind :mod:`pes2ts_data.truth_bridge`.

Not a public entry point: consumers must import
``pes2ts_data.truth_bridge`` (the only sanctioned module).  This helper keeps
the gate logic (frozen-split membership, name validation, audited accessor
call, typed refusal mapping) in one place while the public bridge stays a thin,
signature-stable facade.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0.ids import BadReactionId, normalize_reaction_id

__all__ = [
    "BRIDGE_CALLER",
    "DEFAULT_MANIFESTS_DIR",
    "DEFAULT_SPLIT_MANIFEST",
    "TruthBridgeError",
    "read_audited",
    "reaction_in_frozen_split",
]

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS_DIR: Final[Path] = REPO_ROOT / "data" / "manifests"
DEFAULT_SPLIT_MANIFEST: Final[Path] = (
    REPO_ROOT / "data" / "manifests" / "pesn_irc_split_ids_v1.json")
#: Recorded as the audit ``caller`` for every bridged read (R1 provenance).
BRIDGE_CALLER: Final[str] = "pes2ts_data.truth_bridge"

UNSET: Final[object] = object()
_ERRORS: Final[tuple[type[Exception], ...]] = (
    KeyError, PermissionError, OSError, ValueError)
_BAD_SUFFIXES: Final[tuple[str, ...]] = (
    ".parquet", ".h5", ".json", ".jsonl", ".csv", ".npz", ".pkl")


class TruthBridgeError(Exception):
    """Typed R1 refusal (P2): ``code`` is stable, ``detail`` explains it."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)


def check_override(allow_truth: object) -> None:
    """Refuse a caller-supplied ``allow_truth`` (the gate is not optional)."""
    if allow_truth is not UNSET:
        raise TruthBridgeError(
            "TRUTH_GATE_CLOSED",
            "the R1 bridge always sets allow_truth=True; passing allow_truth "
            "is a refused bypass attempt")


def normalize_name(reaction_id: object) -> str:
    """Normalize an id; refuse path-like / truth-file-like names by construction."""
    if isinstance(reaction_id, str):
        text = reaction_id.strip()
        if (not text or "/" in text or "\\" in text or ".." in text
                or text.lower().endswith(_BAD_SUFFIXES)):
            raise TruthBridgeError(
                "INVALID_REACTION_ID",
                f"{reaction_id!r} is not a reaction id; path-like/truth-file "
                "names are refused")
    try:
        return normalize_reaction_id(reaction_id)
    except BadReactionId as exc:
        raise TruthBridgeError("INVALID_REACTION_ID", str(exc)) from exc


@lru_cache(maxsize=8)
def _load_frozen_ids(path_text: str, mtime_ns: int) -> frozenset[str]:
    doc = json.loads(Path(path_text).read_text(encoding="utf-8"))
    ids: set[str] = set()
    for members in (doc.get("subsets") or {}).values():
        ids.update(str(member) for member in (members or ()))
    ids.update(str(member) for member in
               (doc.get("family_holdout") or {}).get("reaction_ids") or ())
    if not ids:
        raise TruthBridgeError("FROZEN_SPLIT_UNAVAILABLE",
                               f"{path_text}: no frozen reaction ids found")
    return frozenset(ids)


def frozen_ids(split_manifest_path: str | Path | None = None) -> frozenset[str]:
    """Frozen upstream reaction ids (typed refusal when the manifest is absent)."""
    path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST)
    if not path.is_file():
        raise TruthBridgeError("FROZEN_SPLIT_UNAVAILABLE",
                               f"frozen split manifest not found: {path}")
    return _load_frozen_ids(str(path), path.stat().st_mtime_ns)


def reaction_in_frozen_split(reaction_id: object, *,
                             split_manifest_path: str | Path | None = None) -> bool:
    """Whether ``reaction_id`` may be read (no truth access, no audit line)."""
    try:
        return normalize_name(reaction_id) in frozen_ids(split_manifest_path)
    except TruthBridgeError:
        return False


def _error_code(exc: Exception) -> str:
    if isinstance(exc, PermissionError):
        return "TRUTH_GATE_CLOSED"
    if isinstance(exc, KeyError):
        return "REACTION_NOT_IN_TRUTH"
    if isinstance(exc, OSError):
        return "TRUTH_ARTIFACT_MISSING"
    return "TRUTH_MANIFEST_INVALID"


def read_audited(
    accessor: Any,
    reaction_id: object,
    manifests_dir: str | Path | None,
    split_manifest_path: str | Path | None,
    caller: str | None,
) -> Any:
    """Gate one audited read: normalize -> frozen split -> accessor(caller)."""
    normalized = normalize_name(reaction_id)
    if normalized not in frozen_ids(split_manifest_path):
        raise TruthBridgeError("REACTION_NOT_IN_FROZEN_SPLIT",
                               f"{normalized} is not in the frozen upstream split")
    try:
        return accessor(normalized, True,
                        manifests_dir=Path(manifests_dir or DEFAULT_MANIFESTS_DIR),
                        caller=caller or BRIDGE_CALLER)
    except _ERRORS as exc:
        raise TruthBridgeError(_error_code(exc), str(exc)) from exc


def index_audited(accessor: Any, manifests_dir: str | Path | None) -> Any:
    """Gate one shape-only audited read (no reaction scope, still logged)."""
    try:
        return accessor(True, manifests_dir=Path(manifests_dir or DEFAULT_MANIFESTS_DIR))
    except _ERRORS as exc:
        raise TruthBridgeError(_error_code(exc), str(exc)) from exc
