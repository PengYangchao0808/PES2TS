"""Energy-only candidate selection for the native scan (G2-AB3 WP-6).

Selects transition-state seed candidates from the physical-energy curve of
an accepted native-scan trajectory:

* a candidate needs NO gradient evidence — the rule is a local maximum of
  ``E_physical`` bracketed by accepted frames on both sides
  (``bracketed_physical_energy_peak``); missing-gradient observation masks
  are reported explicitly and never fabricated;
* the restraint-including ``E_effective`` channel is never used for peak
  selection — constraint-strength artifacts could manufacture fake peaks;
* when the interval is incomplete and the highest physical energy sits on
  the last accepted frame, that frame is reported separately as
  ``edge_unbracketed_prefix`` — it is a candidate, not a bracketed peak;
* proposals carry geometry/frame bindings (frame index + request sha +
  plan sha) so the existing strict-validation chain can hash-bind to them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

NATIVE_CANDIDATES_SCHEMA = "pes2ts_native_scan_seed_proposals_v1"

PEAK_RULE = "bracketed_physical_energy_peak_then_distinct_edge"
ENERGY_CHANNEL = "physical_unrestrained_single_point"


class NativeCandidateError(ValueError):
    """Raised when the native result document cannot be consumed."""


@dataclass(frozen=True)
class NativeCandidate:
    """One selected candidate frame."""

    frame_index: int
    lambda_value: float
    evidence_class: str
    energy_physical_hartree: float | None
    energy_effective_hartree: float | None
    neighbor_energies_hartree: tuple[float | None, ...]
    request_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "lambda_value": self.lambda_value,
            "evidence_class": self.evidence_class,
            "energy_physical_hartree": self.energy_physical_hartree,
            "energy_effective_hartree": self.energy_effective_hartree,
            "neighbor_energies_hartree": list(self.neighbor_energies_hartree),
            "request_sha256": self.request_sha256,
            "stationary_point_verified": False,
            "gradient_available": False,
        }


def _geometry_digest(geometry: Sequence[Sequence[float]]) -> str:
    rows = [[round(float(x), 6) for x in row] for row in geometry]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def select_native_energy_candidates(
    result: dict[str, Any],
    *,
    top_k: int = 3,
    energy_rmsd_dedup: float = 0.04,
) -> dict[str, Any]:
    """Select peak candidates from a ``pes2ts_native_scan_result_v1`` document.

    Args:
        result: The native scan result (frames with physical energies).
        top_k: Maximum candidates returned.
        energy_rmsd_dedup: Reserved for geometry-based deduplication; v1
            deduplicates by frame index/lambda monotonicity.

    Returns:
        ``pes2ts_native_scan_seed_proposals_v1`` document.
    """
    frames = result.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        return _empty(result, "no_accepted_frames")
    energies = [frame.get("energy_physical_hartree") for frame in frames]
    if not any(isinstance(value, (int, float)) for value in energies):
        return _empty(result, "physical_energy_channel_missing")

    peaks: list[NativeCandidate] = []
    for index in range(1, len(frames) - 1):
        e_prev, e_here, e_next = energies[index - 1], energies[index], energies[index + 1]
        if not all(isinstance(v, (int, float)) for v in (e_prev, e_here, e_next)):
            continue
        if e_here > e_prev and e_here >= e_next:
            peaks.append(
                NativeCandidate(
                    frame_index=index,
                    lambda_value=float(frames[index]["lambda_value"]),
                    evidence_class="bracketed_physical_energy_peak",
                    energy_physical_hartree=float(e_here),
                    energy_effective_hartree=frames[index].get("energy_effective_hartree"),
                    neighbor_energies_hartree=(float(e_prev), float(e_next)),
                    request_sha256=str(frames[index].get("request_sha256", "")),
                )
            )
    peaks.sort(key=lambda c: (-c.energy_physical_hartree, c.frame_index))

    selected: list[NativeCandidate] = []
    for candidate in peaks:
        if len(selected) >= top_k:
            break
        if any(candidate.frame_index == chosen.frame_index for chosen in selected):
            continue
        selected.append(candidate)

    edge: NativeCandidate | None = None
    if not result.get("completed_interval"):
        best_index = max(
            (i for i, value in enumerate(energies) if isinstance(value, (int, float))),
            key=lambda i: float(energies[i]),
        )
        if best_index == len(frames) - 1 and best_index not in (
            c.frame_index for c in selected
        ):
            edge = NativeCandidate(
                frame_index=best_index,
                lambda_value=float(frames[best_index]["lambda_value"]),
                evidence_class="edge_unbracketed_prefix",
                energy_physical_hartree=(
                    float(energies[best_index])
                    if isinstance(energies[best_index], (int, float)) else None
                ),
                energy_effective_hartree=frames[best_index].get("energy_effective_hartree"),
                neighbor_energies_hartree=(
                    (energies[best_index - 1],) if best_index >= 1 else ()
                ),
                request_sha256=str(frames[best_index].get("request_sha256", "")),
            )

    proposals: list[dict[str, Any]] = []
    for rank, candidate in enumerate(selected + ([edge] if edge else []), start=1):
        frame = frames[candidate.frame_index]
        proposals.append({
            "rank": rank,
            **candidate.to_payload(),
            "geometry": frame.get("geometry"),
            "geometry_sha256": _geometry_digest(frame.get("geometry") or []),
            "plan_sha256": result.get("plan_sha256"),
            "attempt_dir": frame.get("attempt_dir"),
            "observation_mask": {
                "physical_energy": True,
                "physical_gradient": False,
                "effective_energy": candidate.energy_effective_hartree is not None,
            },
        })

    return {
        "schema_version": NATIVE_CANDIDATES_SCHEMA,
        "engine_mode": "classic_native_xtb_scan",
        "reaction_id": result.get("reaction_id"),
        "rule": PEAK_RULE,
        "energy_channel": ENERGY_CHANNEL,
        "completed_interval": bool(result.get("completed_interval")),
        "n_accepted_frames": len(frames) - 1,
        "gradient_available": False,
        "n_candidates": len(proposals),
        "candidates": proposals,
        "validation_status": "blocked_on_human_review",
        "validation_blocked_needs": [
            "accepted ReviewRecord for this case (two-reviewer workbook + adjudication + import)"
        ],
    }


def _empty(result: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": NATIVE_CANDIDATES_SCHEMA,
        "engine_mode": "classic_native_xtb_scan",
        "reaction_id": result.get("reaction_id"),
        "rule": PEAK_RULE,
        "energy_channel": ENERGY_CHANNEL,
        "completed_interval": bool(result.get("completed_interval")),
        "n_accepted_frames": max(0, len(result.get("frames", [])) - 1),
        "gradient_available": False,
        "n_candidates": 0,
        "candidates": [],
        "no_candidate_reason": reason,
        "validation_status": "blocked_on_human_review",
        "validation_blocked_needs": [
            "accepted ReviewRecord for this case (two-reviewer workbook + adjudication + import)"
        ],
    }


__all__ = [
    "ENERGY_CHANNEL",
    "NATIVE_CANDIDATES_SCHEMA",
    "NativeCandidate",
    "NativeCandidateError",
    "PEAK_RULE",
    "select_native_energy_candidates",
]
