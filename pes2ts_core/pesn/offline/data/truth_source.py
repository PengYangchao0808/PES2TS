"""The single audited truth bridge of the predictor product (B2).

This is the ONLY ``pesn`` module registered in the static truth guard's
allowlist, and the only place the predictor may touch the quarantined
TS/IRC truth.  It enforces scope-before-access exactly as the design
requires:

1. the caller must pass ``allow_truth=True`` (the audited accessors would
   refuse otherwise, but the bridge refuses first, before any file path is
   even resolved);
2. the requested reaction must be a member of the frozen predictor split
   subset the caller declared (train-side teacher building only — sealed
   valid/test reactions are structurally refused here);
3. the frozen split manifest must validate against the split contract;
4. only then is :func:`load_irc_frames` of the audited G0 accessor called
   with ``caller`` set to this module, appending to the shared audit log.

The bridge returns plain frame dictionaries (coordinates/EHG) — no file
paths, no source handles, and no iterators that could outlive the scope
check.  Every other ``pesn`` module receives teacher data exclusively as
plain values from the offline builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping

from pes2ts_core.g0.truth import truth_reader
from pes2ts_core.pesn.offline.splits.freeze import (
    TRAIN_CALIBRATION,
    TRAIN_DEV,
    TRAIN_FIT,
    load_split_manifest,
)
from pes2ts_core.utils.jsonio import read_json

TRUTH_BRIDGE_MODULE: Final[str] = "pes2ts_core.pesn.offline.data.truth_source"
#: Subsets whose members may be read through this bridge.
TRUTH_ALLOWED_SUBSETS: Final[frozenset[str]] = frozenset(
    {TRAIN_FIT, TRAIN_DEV, TRAIN_CALIBRATION}
)


class TruthBridgeError(PermissionError):
    """Refusal before any truth byte is read (typed, auditable)."""


def _allowed_reactions(split_dir: str | Path) -> frozenset[str]:
    manifest = load_split_manifest(split_dir)
    allowed: set[str] = set()
    subsets = manifest.get("subsets")
    if not isinstance(subsets, dict):
        raise TruthBridgeError("split manifest subsets malformed")
    for subset in TRUTH_ALLOWED_SUBSETS:
        members = subsets.get(subset)
        if isinstance(members, list):
            allowed.update(str(rid) for rid in members)
    return frozenset(allowed)


def load_scoped_irc_frames(
    reaction_id: str,
    *,
    allow_truth: bool,
    manifests_dir: str | Path,
    split_dir: str | Path,
) -> dict[str, object]:
    """Audited, scope-checked IRC frames for one train-side reaction.

    Raises
    ------
    TruthBridgeError
        When ``allow_truth`` is false or the reaction is outside the frozen
        train-side subsets — raised before any source path is resolved.
    KeyError
        From the audited accessor when the reaction has no IRC archive entry.
    """
    if not allow_truth:
        raise TruthBridgeError(
            "truth bridge requires allow_truth=True (audited teacher build)"
        )
    allowed = _allowed_reactions(split_dir)
    if reaction_id not in allowed:
        raise TruthBridgeError(
            f"reaction {reaction_id!r} is outside the frozen train-side subsets; "
            "sealed reactions are structurally refused by this bridge"
        )
    return truth_reader.load_irc_frames(
        reaction_id,
        allow_truth,
        manifests_dir=manifests_dir,
        caller=TRUTH_BRIDGE_MODULE,
    )


def load_scoped_ts_geometry(
    reaction_id: str,
    *,
    allow_truth: bool,
    manifests_dir: str | Path,
    split_dir: str | Path,
) -> dict[str, object]:
    """Audited, scope-checked TS geometry for one train-side reaction."""
    if not allow_truth:
        raise TruthBridgeError(
            "truth bridge requires allow_truth=True (audited teacher build)"
        )
    allowed = _allowed_reactions(split_dir)
    if reaction_id not in allowed:
        raise TruthBridgeError(
            f"reaction {reaction_id!r} is outside the frozen train-side subsets"
        )
    return truth_reader.load_ts_geometry(
        reaction_id,
        allow_truth,
        manifests_dir=manifests_dir,
        caller=TRUTH_BRIDGE_MODULE,
    )


def stream_scoped_irc_frames(
    selected_reaction_ids: frozenset[str],
    *,
    allow_truth: bool,
    manifests_dir: str | Path,
    split_dir: str | Path,
):
    """Single streaming pass that yields only in-scope selected reactions.

    Selection sizes above the per-reaction lookup threshold would make
    per-reaction reads O(N^2) in archive visits (each audited read rescans
    the group hierarchy), which is why G0 provides one audited streaming
    accessor for annotation-scale runs.  Scope discipline here is stronger
    than "read everything then mask": the allowed set is intersected with
    *selected_reaction_ids* BEFORE the archive is opened (every requested id
    must be a member of the frozen train-side subsets), and out-of-scope
    trajectories yielded by the stream are dropped inside this bridge —
    sealed frames never reach pesn code outside this module.

    Raises
    ------
    TruthBridgeError
        When ``allow_truth`` is false, or any requested reaction lies
        outside the frozen train-side subsets — before anything is read.
    """
    if not allow_truth:
        raise TruthBridgeError(
            "truth bridge requires allow_truth=True (audited teacher build)"
        )
    allowed = _allowed_reactions(split_dir)
    out_of_scope = sorted(set(selected_reaction_ids) - allowed)
    if out_of_scope:
        raise TruthBridgeError(
            f"{len(out_of_scope)} requested reactions are outside the frozen "
            f"train-side subsets (e.g. {out_of_scope[:3]}); refused before any read"
        )
    wanted = frozenset(selected_reaction_ids)
    for frames in truth_reader.iter_irc_trajectories(
        allow_truth,
        manifests_dir=manifests_dir,
        caller=TRUTH_BRIDGE_MODULE,
    ):
        if frames.reaction_id in wanted:
            yield {
                "reaction_id": frames.reaction_id,
                "n_atoms": frames.n_atoms,
                "n_frames": frames.n_frames,
                "has_forces": frames.has_forces,
                "coordinates": frames.coordinates,
                "EHG": frames.ehg,
                "atomic_numbers": frames.atomic_numbers,
            }


def peek_dataset_binding(dataset_manifest_path: str | Path) -> Mapping[str, object]:
    """Read a teacher dataset manifest for provenance binding (no truth)."""
    document = read_json(Path(dataset_manifest_path))
    if not isinstance(document, dict):
        raise TruthBridgeError(f"dataset manifest malformed: {dataset_manifest_path}")
    return document


__all__ = [
    "TRUTH_ALLOWED_SUBSETS",
    "TRUTH_BRIDGE_MODULE",
    "TruthBridgeError",
    "load_scoped_irc_frames",
    "load_scoped_ts_geometry",
    "peek_dataset_binding",
    "stream_scoped_irc_frames",
]
