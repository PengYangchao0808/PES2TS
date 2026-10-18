"""Direct free-path candidates and PathBundle adaptation (WP7a, §8/§9).

A ``DirectPathCandidate`` is the auditable record of one policy rollout
prefix: every evaluated geometry with its E/g receipt hashes, the logical
budget actually consumed, the termination classification, and the frame
hashes the strict chain keys on. ``build_direct_path_bundle`` adapts a
candidate to the existing ``g2_path_v1`` PathBundle frame structure so the
downstream validation chain consumes direct candidates through the same
contract as Classic paths — without touching the validation code itself.

This module never submits QC and never reads truth; it transforms already
committed transitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Iterable, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.direct import DirectTransition
from pes2ts_core.utils.hashing import sha256_bytes

CANDIDATE_SCHEMA: Final[str] = "pes2ts_direct_path_candidate_v1"
BUNDLE_FRAME_SCHEMA: Final[str] = "pes2ts_direct_bundle_frame_v1"


class DirectCandidateError(RuntimeError):
    """Typed candidate construction refusal."""


def _frame_hash(coordinates: np.ndarray) -> str:
    return sha256_bytes(np.asarray(coordinates, dtype=float).round(8).tobytes())


@dataclass(frozen=True, slots=True)
class DirectPathCandidate:
    schema_version: str = CANDIDATE_SCHEMA
    reaction_id: str = ""
    group_id: str = ""
    episode_id: str = ""
    subset: str = ""
    origin: str = "deployment_origin"
    frames_geometry: tuple[np.ndarray, ...] = ()
    frames_energy_hartree: tuple[float, ...] = ()
    frames_hash: tuple[str, ...] = ()
    termination_reason: str = ""
    reached_goal_proxy: bool = False
    best_potential: float = 0.0
    logical_steps_used: int = 0
    qc_request_hashes: tuple[str, ...] = ()
    policy_version: str = ""

    def validate(self) -> None:
        if self.schema_version != CANDIDATE_SCHEMA:
            raise DirectCandidateError("candidate schema mismatch")
        if not self.reaction_id or not self.episode_id:
            raise DirectCandidateError("candidate lacks reaction/episode identity")
        n_geo = len(self.frames_geometry)
        if n_geo == 0 or len(self.frames_energy_hartree) != n_geo:
            raise DirectCandidateError("frame geometry/energy length mismatch")
        if len(self.frames_hash) != n_geo:
            raise DirectCandidateError("frame hash length mismatch")
        if self.logical_steps_used <= 0:
            raise DirectCandidateError("candidate consumed no logical steps")
        if self.reached_goal_proxy and self.termination_reason != "goal_connected_proxy":
            raise DirectCandidateError("goal candidate must carry the goal terminal")


def candidate_from_transitions(
    transitions: Sequence[DirectTransition],
    *,
    reset_coordinates: np.ndarray,
    reset_energy_hartree: float,
    policy_version: str = "",
    origin: str = "deployment_origin",
    qc_request_hashes: Iterable[str] = (),
) -> DirectPathCandidate | None:
    """Wrap one rollout's committed transitions into a candidate.

    Returns ``None`` for empty rollouts. The candidate keeps every
    *evaluated* node (reset included); rejected proposals contribute no
    frame but do consume logical budget (R5: all attempts retained).
    """
    if not transitions:
        return None
    geometries: list[np.ndarray] = [np.asarray(reset_coordinates, dtype=float)]
    energies: list[float] = [float(reset_energy_hartree)]
    hashes: list[str] = [_frame_hash(geometries[0])]
    request_hashes: list[str] = list(qc_request_hashes)
    for tr in transitions:
        if tr.terminated or tr.td_eligible:
            if tr.next_state is not None and tr.next_state.observation_mask:
                coords = np.asarray(tr.next_state.coordinates, dtype=float)
                geometries.append(coords)
                energies.append(float(tr.next_state.energy_hartree or 0.0))
                hashes.append(_frame_hash(coords))
                if tr.qc_cost_evidence and tr.qc_cost_evidence.get("request_sha256"):
                    request_hashes.append(str(tr.qc_cost_evidence["request_sha256"]))
        elif tr.next_state is not None and tr.next_state.observation_mask:
            coords = np.asarray(tr.next_state.coordinates, dtype=float)
            geometries.append(coords)
            energies.append(float(tr.next_state.energy_hartree or 0.0))
            hashes.append(_frame_hash(coords))
    first = transitions[0]
    last = transitions[-1]
    best = max(
        (float(t.next_state.event_state.get("best_potential", 0.0))
         for t in transitions if t.next_state is not None),
        default=0.0,
    )
    candidate = DirectPathCandidate(
        reaction_id=first.reaction_id,
        group_id=first.group_id,
        episode_id=first.episode_id,
        subset=first.subset,
        origin=origin,
        frames_geometry=tuple(geometries),
        frames_energy_hartree=tuple(energies),
        frames_hash=tuple(hashes),
        termination_reason=last.termination_reason,
        reached_goal_proxy=last.termination_reason == "goal_connected_proxy",
        best_potential=best,
        logical_steps_used=len(transitions),
        qc_request_hashes=tuple(request_hashes),
        policy_version=policy_version,
    )
    candidate.validate()
    return candidate


def build_direct_path_bundle(
    candidate: DirectPathCandidate,
    *,
    elements: Sequence[str],
    atom_map_ids: Sequence[str],
    method_id: str = "GFN2-xTB",
    charge: int = 0,
) -> dict:
    """Adapt to the ``g2_path_v1`` frame structure consumed by validation."""
    frames = []
    for index, (geometry, energy, frame_hash) in enumerate(zip(
            candidate.frames_geometry, candidate.frames_energy_hartree,
            candidate.frames_hash)):
        frames.append({
            "frame_id": f"{candidate.episode_id}-f{index:04d}",
            "frame_index": index,
            "frame_sha256": frame_hash,
            "atom_map_ids": list(atom_map_ids),
            "geometry": np.asarray(geometry, dtype=float).round(8).tolist(),
            "elements": list(elements),
            "energies": {
                "xtp": {"unit": "hartree", "method_id": method_id,
                        "value": float(energy)},
            },
        })
    return {
        "schema_version": BUNDLE_FRAME_SCHEMA,
        "reaction_id": candidate.reaction_id,
        "group_id": candidate.group_id,
        "episode_id": candidate.episode_id,
        "subset": candidate.subset,
        "origin": candidate.origin,
        "policy_version": candidate.policy_version,
        "charge": int(charge),
        "termination_reason": candidate.termination_reason,
        "reached_goal_proxy": bool(candidate.reached_goal_proxy),
        "logical_steps_used": int(candidate.logical_steps_used),
        "qc_request_hashes": list(candidate.qc_request_hashes),
        "frames": frames,
        "n_frames": len(frames),
        "note": (
            "direct rollout candidate; strict validation still has to run the "
            "real OptTS/Freq/IRC chain — reaching the goal proxy is not a "
            "validated transition state"
        ),
    }


__all__ = [
    "BUNDLE_FRAME_SCHEMA",
    "CANDIDATE_SCHEMA",
    "DirectCandidateError",
    "DirectPathCandidate",
    "build_direct_path_bundle",
    "candidate_from_transitions",
]
