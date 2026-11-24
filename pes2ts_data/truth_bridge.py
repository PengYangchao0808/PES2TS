"""THE single R1 truth bridge for Zones 2/3.

Only sanctioned entry to the quarantined TS/IRC ground truth; a thin wrapper
over ``pes2ts_core.g0.truth.truth_reader`` only (the gate machinery lives in
:mod:`pes2ts_data._truth_gate`).  Every access always passes
``allow_truth=True`` — an explicit ``allow_truth=`` argument is a typed refusal
— is written to ``truth_access_log.jsonl`` by the reader with
:data:`BRIDGE_CALLER`, and is limited to reactions of the frozen upstream split
(``data/manifests/pesn_irc_split_ids_v1.json``).  Refusals are
:class:`TruthBridgeError` with a stable ``.code``.

R1 boundary: never re-export this module from a package ``__init__`` and never
import it from an online/generation path; truth values are offline teacher
labels only, never model inputs or controller state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pes2ts_core.g0.truth import truth_reader

from ._truth_gate import (
    BRIDGE_CALLER,
    DEFAULT_MANIFESTS_DIR,
    DEFAULT_SPLIT_MANIFEST,
    UNSET,
    TruthBridgeError,
    check_override,
    index_audited,
    reaction_in_frozen_split,
    read_audited,
)

__all__ = [
    "BRIDGE_CALLER",
    "DEFAULT_MANIFESTS_DIR",
    "DEFAULT_SPLIT_MANIFEST",
    "TruthBridgeError",
    "load_irc_frames",
    "load_irc_index",
    "load_ts_geometry",
    "reaction_in_frozen_split",
]


def load_ts_geometry(
    reaction_id: str,
    *,
    manifests_dir: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    caller: str | None = None,
    allow_truth: object = UNSET,
) -> dict[str, Any]:
    """Read one quarantined TS geometry row through the audited accessor."""
    check_override(allow_truth)
    return read_audited(truth_reader.load_ts_geometry, reaction_id,
                        manifests_dir, split_manifest_path, caller)


def load_irc_frames(
    reaction_id: str,
    *,
    manifests_dir: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    caller: str | None = None,
    allow_truth: object = UNSET,
) -> dict[str, Any]:
    """Read one reaction's quarantined IRC frames through the audited accessor."""
    check_override(allow_truth)
    return read_audited(truth_reader.load_irc_frames, reaction_id,
                        manifests_dir, split_manifest_path, caller)


def load_irc_index(
    *,
    manifests_dir: str | Path | None = None,
    allow_truth: object = UNSET,
) -> list[dict[str, Any]]:
    """Read the shape-only IRC index (no trajectory data, still audited)."""
    check_override(allow_truth)
    return index_audited(truth_reader.load_irc_index, manifests_dir)
