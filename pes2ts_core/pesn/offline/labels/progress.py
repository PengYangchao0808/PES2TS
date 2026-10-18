"""Signed progress and path-proximity labels from IRC trajectories (B2).

For each reaction the trajectory is split at the TS frame into the branches
the G1 P1 validation already froze (``ts_first_dual_branch`` /
``single_branch`` layouts only).  Adjacent frames are registered by
mass-weighted rigid (Kabsch) alignment, and the arc-length increments

    delta_l_i = sqrt(sum_a m_a * ||X_aligned[i+1,a] - X[i,a]||^2)

accumulate into ``d_arc`` from the TS frame; the branch totals ``L_R``/``L_P``
normalize it into signed progress ``s`` in ``[-1, +1]`` (negative on the
reactant side, positive on the product side, zero at the TS anchor) and path
proximity ``d = |s|``.

Semantics frozen in the LabelSpec:
* ``±1`` marks the recorded branch terminus only — ``endpoint_status`` records
  truncation; it is never read as "reaction complete";
* a branch with zero total arc masks its whole side (typed reason), never
  zero-filled;
* the TS anchor keeps ``s = 0`` but its direction label is masked upstream;
* frames with non-finite coordinates are masked with typed reasons;
* a single-sided trajectory labels only its present side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

#: Standard atomic weights (IUPAC 2021 abridged), Z = 1..86, in u.
ATOMIC_MASSES: Final[dict[int, float]] = {
    1: 1.008, 2: 4.0026, 3: 6.94, 4: 9.0122, 5: 10.81, 6: 12.011, 7: 14.007,
    8: 15.999, 9: 18.998, 10: 20.180, 11: 22.990, 12: 24.305, 13: 26.982,
    14: 28.085, 15: 30.974, 16: 32.06, 17: 35.45, 18: 39.948, 19: 39.098,
    20: 40.078, 21: 44.956, 22: 47.867, 23: 50.942, 24: 51.996, 25: 54.938,
    26: 55.845, 27: 58.933, 28: 58.693, 29: 63.546, 30: 65.38, 31: 69.723,
    32: 72.630, 33: 74.922, 34: 78.971, 35: 79.904, 36: 83.798, 37: 85.468,
    38: 87.62, 39: 88.906, 40: 91.224, 41: 92.906, 42: 95.95, 43: 98.0,
    44: 101.07, 45: 102.91, 46: 106.42, 47: 107.87, 48: 112.41, 49: 114.82,
    50: 118.71, 51: 121.76, 52: 127.60, 53: 126.90, 54: 131.29, 55: 132.91,
    56: 137.33, 57: 138.91, 58: 140.12, 59: 140.91, 60: 144.24, 61: 145.0,
    62: 150.36, 63: 151.96, 64: 157.25, 65: 158.93, 66: 162.50, 67: 164.93,
    68: 167.26, 69: 168.93, 70: 173.05, 71: 174.97, 72: 178.49, 73: 180.95,
    74: 183.84, 75: 186.21, 76: 190.23, 77: 192.22, 78: 195.08, 79: 196.97,
    80: 200.59, 81: 204.38, 82: 207.2, 83: 208.98, 84: 209.0, 85: 210.0,
    86: 222.0,
}
MASSES_VERSION: Final[str] = "iupac2021_abridged"

SIDE_R: Final[int] = -1
SIDE_TS: Final[int] = 0
SIDE_P: Final[int] = +1

REASON_OK: Final[str] = "ok"
REASON_ZERO_ARC: Final[str] = "zero_branch_arc"
REASON_NONFINITE_FRAME: Final[str] = "nonfinite_frame"


class ProgressLabelError(RuntimeError):
    """Typed failure of progress-label construction for one reaction."""


def masses_for(atomic_numbers: np.ndarray | list[int]) -> np.ndarray:
    try:
        return np.array([ATOMIC_MASSES[int(z)] for z in atomic_numbers], dtype=np.float64)
    except KeyError as exc:
        raise ProgressLabelError(f"no standard mass for Z={exc.args[0]}") from exc


def kabsch_align(mobile: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Mass-weighted rigid alignment of *mobile* onto *reference* (Kabsch)."""
    w = weights / weights.sum()
    mob_c = mobile - np.sum(w[:, None] * mobile, axis=0)
    ref_c = reference - np.sum(w[:, None] * reference, axis=0)
    covariance = (w[:, None] * mob_c).T @ ref_c
    u, _, vt = np.linalg.svd(covariance)
    d_sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d_sign if d_sign != 0 else 1.0])
    rotation = vt.T @ correction @ u.T
    return mob_c @ rotation.T + np.sum(w[:, None] * reference, axis=0)


def cumulative_arc(
    frames: np.ndarray, weights: np.ndarray, indices: list[int]
) -> np.ndarray:
    """Cumulative mass-weighted arc length along a frame-index path.

    ``indices`` is ordered from the TS frame outward; entry ``k`` is the
    accumulated arc from ``indices[0]`` to ``indices[k]``.
    """
    arc = np.zeros(len(indices), dtype=np.float64)
    for k in range(1, len(indices)):
        prev = frames[indices[k - 1]]
        aligned = kabsch_align(frames[indices[k]], prev, weights)
        arc[k] = arc[k - 1] + np.sqrt(
            float(np.sum(weights * np.sum((aligned - prev) ** 2, axis=1)))
        )
    return arc


@dataclass(frozen=True, slots=True)
class ProgressLabels:
    """Per-frame progress labels with masks and typed reasons."""

    signed_progress: np.ndarray  # (F,) float64, 0 where masked
    path_proximity: np.ndarray  # (F,) float64 = |s|
    progress_mask: np.ndarray  # (F,) bool
    side: np.ndarray  # (F,) int8: -1 R, 0 TS, +1 P
    d_arc: np.ndarray  # (F,) float64 arc distance to TS (masked side = 0)
    branch_total_r: float
    branch_total_p: float
    endpoint_status: dict[str, str]
    frame_reasons: dict[int, str]

    @property
    def n_frames(self) -> int:
        return len(self.signed_progress)


def _finite_frames(frames: np.ndarray) -> np.ndarray:
    return np.isfinite(frames).all(axis=(1, 2))


def compute_progress_labels(
    frames: np.ndarray,
    masses: np.ndarray,
    *,
    ts_frame_index: int,
    splice_index: int | None,
    layout: str,
    orientation: str,
) -> ProgressLabels:
    """Compute signed progress / proximity for one IRC trajectory.

    Branch semantics follow the frozen G1 P1 validation: for
    ``ts_first_dual_branch`` the frames after the TS split at ``splice_index``
    into branch_1 (``ts+1 .. splice``) and branch_2 (``splice+1 .. end``);
    ``orientation`` names which side branch_1 terminates on.  For
    ``single_branch`` only branch_1 exists.
    """
    n_frames = frames.shape[0]
    if not 0 <= ts_frame_index < n_frames:
        raise ProgressLabelError(f"ts_frame_index {ts_frame_index} out of range")
    finite = _finite_frames(frames)
    signed = np.zeros(n_frames, dtype=np.float64)
    mask = np.zeros(n_frames, dtype=bool)
    side = np.zeros(n_frames, dtype=np.int8)
    d_arc = np.zeros(n_frames, dtype=np.float64)
    reasons: dict[int, str] = {}
    endpoint_status: dict[str, str] = {
        "r_side": "absent",
        "p_side": "absent",
        "layout": layout,
        "orientation": orientation,
    }

    if layout == "single_branch":
        branches = [("b1", list(range(ts_frame_index + 1, n_frames)))]
    elif layout == "ts_first_dual_branch":
        if splice_index is None or not ts_frame_index < splice_index < n_frames - 1:
            raise ProgressLabelError(
                f"splice_index {splice_index} invalid for {n_frames} frames"
            )
        branches = [
            ("b1", list(range(ts_frame_index + 1, splice_index + 1))),
            ("b2", list(range(splice_index + 1, n_frames))),
        ]
    else:
        raise ProgressLabelError(f"unsupported layout {layout!r}")

    branch1_side = SIDE_P if orientation == "P_first" else SIDE_R
    totals: dict[int, float] = {}

    for branch_name, indices in branches:
        branch_side = branch1_side if branch_name == "b1" else -branch1_side
        usable = [i for i in indices if finite[i]]
        dropped = [i for i in indices if not finite[i]]
        for i in dropped:
            reasons[i] = REASON_NONFINITE_FRAME
        if len(usable) < 2:
            for i in indices:
                side[i] = branch_side
            endpoint_status["r_side" if branch_side == SIDE_R else "p_side"] = (
                "degenerate" if usable else "empty"
            )
            totals[branch_side] = float("nan")
            continue
        arc = cumulative_arc(frames, masses, [ts_frame_index, *usable])
        total = float(arc[-1])
        totals[branch_side] = total
        if total <= 1e-9:
            for i in usable:
                reasons[i] = REASON_ZERO_ARC
            endpoint_status["r_side" if branch_side == SIDE_R else "p_side"] = "zero_arc"
            continue
        for k, frame_idx in enumerate(usable):
            normalized = float(arc[k + 1]) / total
            signed[frame_idx] = normalized * branch_side
            mask[frame_idx] = True
            side[frame_idx] = branch_side
            d_arc[frame_idx] = float(arc[k + 1])
        endpoint_status["r_side" if branch_side == SIDE_R else "p_side"] = "recorded_terminus"

    signed[ts_frame_index] = 0.0
    mask[ts_frame_index] = True
    side[ts_frame_index] = SIDE_TS

    return ProgressLabels(
        signed_progress=signed,
        path_proximity=np.abs(signed),
        progress_mask=mask,
        side=side,
        d_arc=d_arc,
        branch_total_r=float(totals.get(SIDE_R, float("nan"))),
        branch_total_p=float(totals.get(SIDE_P, float("nan"))),
        endpoint_status=endpoint_status,
        frame_reasons=reasons,
    )


__all__ = [
    "ATOMIC_MASSES",
    "MASSES_VERSION",
    "ProgressLabelError",
    "ProgressLabels",
    "REASON_NONFINITE_FRAME",
    "REASON_OK",
    "REASON_ZERO_ARC",
    "SIDE_P",
    "SIDE_R",
    "SIDE_TS",
    "compute_progress_labels",
    "cumulative_arc",
    "kabsch_align",
    "masses_for",
]
