"""Graph-defined low-dim teacher direction and coordinate-activity labels (B2).

From the current graph-defined driver coordinates ``q = phi_G(X)`` (v1: F/B/O
edit-pair distances, scaled by ``q_tilde = (q - q_ref) / a``), the arc-length
parameter toward the TS is ``z`` (increasing toward the TS: ``z = L - d_arc``
on the frame's branch), and the local tangent

    v* = dq_tilde / dz   (one-sided difference toward the TS; the TS anchor
                          and zero-step neighbours are masked, never zeroed)

yields the unit supervision direction ``t*_q = v* / ||v*||`` and the per-
coordinate activity contribution ``|v*_j| / sum_j |v*_j|``.  A coordinate is
*active* when its contribution is at least the average contribution
(``>= 1/n_coords``), the v1 frozen rule recorded in the LabelSpec; threshold
calibration on train data is a later, separately-frozen revision.

These labels are an ideal local projection inside the chosen coordinates —
not commands, not Bellman transitions, and never written into any
``command_delta_q`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

from pes2ts_core.pesn.offline.labels.progress import ProgressLabels, SIDE_TS

ACTIVITY_RULE_VERSION: Final[str] = "above_average_v1"
DIFFERENCE_RULE_VERSION: Final[str] = "one_sided_toward_ts_v1"
REASON_OK: Final[str] = "ok"
REASON_TS_ANCHOR: Final[str] = "ts_anchor"
REASON_ZERO_ARC_STEP: Final[str] = "zero_arc_step"
REASON_ZERO_NORM: Final[str] = "zero_tangent_norm"
REASON_NONFINITE: Final[str] = "nonfinite_tangent"
EPS_ARC: Final[float] = 1e-9
EPS_NORM: Final[float] = 1e-12


class DirectionLabelError(RuntimeError):
    """Typed failure of direction-label construction."""


def evaluate_distance_coords(
    frames: np.ndarray, coord_atom_pairs: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Driver distance values ``q`` for every frame, shape ``(F, C)``."""
    if not coord_atom_pairs:
        raise DirectionLabelError("no driver coordinates declared")
    pairs = np.asarray(coord_atom_pairs, dtype=int)
    diffs = frames[:, pairs[:, 0], :] - frames[:, pairs[:, 1], :]
    return np.linalg.norm(diffs, axis=2)


@dataclass(frozen=True, slots=True)
class DirectionLabels:
    """Per-frame direction/activity labels with masks and typed reasons."""

    q: np.ndarray  # (F, C) raw driver values
    q_scaled: np.ndarray  # (F, C) (q - ref)/a
    direction: np.ndarray  # (F, C) unit t* (rows masked where invalid)
    activity: np.ndarray  # (F, C) 0/1 labels
    contribution: np.ndarray  # (F, C) normalized |v*| shares
    direction_mask: np.ndarray  # (F,) bool
    activity_mask: np.ndarray  # (F,) bool
    frame_reasons: dict[int, str]


def compute_direction_labels(
    q: np.ndarray,
    progress: ProgressLabels,
    *,
    scale_a: np.ndarray | float = 1.0,
    ref_q: np.ndarray | float = 0.0,
) -> DirectionLabels:
    """Compute t*_q and activity from driver values and progress labels.

    *q* has shape ``(F, C)`` in raw units; scaling by *scale_a*/*ref_q*
    (per coordinate) produces ``q_tilde``.  One-sided differences toward the
    TS define ``v*``; frames without a valid same-side neighbour closer to
    the TS (the TS anchor itself, branch termini adjacent to duplicate
    frames) are masked with typed reasons.
    """
    q = np.asarray(q, dtype=np.float64)
    n_frames, n_coords = q.shape
    scale = np.broadcast_to(np.asarray(scale_a, dtype=float), (n_coords,))
    ref = np.broadcast_to(np.asarray(ref_q, dtype=float), (n_coords,))
    q_scaled = (q - ref) / scale

    direction = np.zeros((n_frames, n_coords), dtype=np.float64)
    activity = np.zeros((n_frames, n_coords), dtype=np.float64)
    contribution = np.zeros((n_frames, n_coords), dtype=np.float64)
    direction_mask = np.zeros(n_frames, dtype=bool)
    activity_mask = np.zeros(n_frames, dtype=bool)
    reasons: dict[int, str] = {}

    d_arc = progress.d_arc
    side = progress.side
    branch_total = np.where(
        side < 0, progress.branch_total_r, progress.branch_total_p
    )
    z_toward_ts = branch_total - d_arc  # increases toward the TS on each side

    for i in range(n_frames):
        if side[i] == SIDE_TS:
            reasons[i] = REASON_TS_ANCHOR
            continue
        if not progress.progress_mask[i]:
            reasons.setdefault(i, "progress_masked")
            continue
        same_side = np.where((side == side[i]) & progress.progress_mask)[0]
        closer = [j for j in same_side if z_toward_ts[j] > z_toward_ts[i] + EPS_ARC]
        if not closer:
            nearest = same_side[np.argmax(z_toward_ts[same_side])]
            if z_toward_ts[nearest] <= z_toward_ts[i] + EPS_ARC:
                reasons[i] = REASON_ZERO_ARC_STEP
                continue
            closer = [int(nearest)]
        j = min(closer, key=lambda k: z_toward_ts[k])
        v = (q_scaled[i] - q_scaled[j]) / (z_toward_ts[i] - z_toward_ts[j])
        if not np.isfinite(v).all():
            reasons[i] = REASON_NONFINITE
            continue
        norm = float(np.linalg.norm(v))
        if norm < EPS_NORM:
            reasons[i] = REASON_ZERO_NORM
            continue
        direction[i] = v / norm
        contribution[i] = np.abs(v) / np.sum(np.abs(v))
        activity[i] = contribution[i] >= (1.0 / n_coords)
        direction_mask[i] = True
        activity_mask[i] = True
        reasons[i] = REASON_OK

    return DirectionLabels(
        q=q,
        q_scaled=q_scaled,
        direction=direction,
        activity=activity,
        contribution=contribution,
        direction_mask=direction_mask,
        activity_mask=activity_mask,
        frame_reasons=reasons,
    )


__all__ = [
    "ACTIVITY_RULE_VERSION",
    "DIFFERENCE_RULE_VERSION",
    "DirectionLabelError",
    "DirectionLabels",
    "REASON_NONFINITE",
    "REASON_OK",
    "REASON_TS_ANCHOR",
    "REASON_ZERO_ARC_STEP",
    "REASON_ZERO_NORM",
    "compute_direction_labels",
    "evaluate_distance_coords",
]
