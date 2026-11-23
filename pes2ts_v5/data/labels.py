"""Frozen V5 label contract (Move / Process / Trust) and loss masks.

Research plan v2.1 §1.2 / §2.5:

======  ==========================================  =============================
Head    legal V5 supervision                       forbidden pseudo-label
======  ==========================================  =============================
Move    dq/ds, reference geometric displacement,    invented "optimal QC action"
        Classic residual
Process path progress, reference TS-near region     unvalidated OptTS success state
Trust   large-error / OOD / geometry-anomaly proxy  invented xTB acceptance rate
======  ==========================================  =============================

All labels have explicit masks; ``no_future_step`` frames (the last frame of a
trajectory) are masked for Move.  Never fill a missing head label with a
constant to "make supervision complete".
"""

from __future__ import annotations

import numpy as np

PROCESS_CLASSES = ("pre_ts", "near_ts", "post_ts")

LABEL_SPEC = {
    "version": "v5_label_contract_v1_synthetic",
    "move": {
        "kind": "reference_geometric_displacement",
        "definition": "q_{t+1} - q_t (controlled reaction-coordinate displacement)",
        "mask": "last frame of each trajectory has no future step (no_future_step)",
        "forbidden": "invented optimal QC action",
    },
    "process": {
        "kind": "path_progress_and_ts_neighborhood",
        "definition": "progress in [-1, 1] with reference TS at 0; "
                      "class = pre/near/post by |progress| vs near_ts_window",
        "mask": "frames without a reference progress are masked",
        "forbidden": "unvalidated OptTS success state",
    },
    "trust": {
        "kind": "offline_risk_proxy",
        "definition": "normalized |second difference of q| (curvature anomaly proxy)",
        "mask": "frames without proxy are masked; never treat normal IRC frames as Trust=1",
        "forbidden": "invented xTB acceptance rate or success probability",
    },
}


def build_move_labels(
    q: np.ndarray,
    *,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (move_target [T,K], move_mask [T]) from a trajectory's q path.

    ``move_target[t] = q[t+1] - q[t]`` (reference geometric displacement);
    last frame is masked as ``no_future_step``.  ``smooth`` currently only
    documents intent: the reference displacement is already noise-free in the
    synthetic fixture; real data adapter may pass smoothed derivatives.
    """
    q = np.asarray(q, dtype=np.float32)
    n = q.shape[0]
    target = np.zeros_like(q)
    mask = np.zeros(n, dtype=bool)
    if n >= 2:
        target[:-1] = q[1:] - q[:-1]
        mask[:-1] = True
    return target, mask


def build_process_labels(
    progress: np.ndarray,
    *,
    near_ts_window: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (class [T], class_mask [T], progress_target [T], progress_mask [T])."""
    progress = np.asarray(progress, dtype=np.float32)
    valid = np.isfinite(progress)
    cls = np.full(progress.shape, -1, dtype=np.int64)
    cls[valid & (progress < -near_ts_window)] = 0
    cls[valid & (np.abs(progress) <= near_ts_window)] = 1
    cls[valid & (progress > near_ts_window)] = 2
    class_mask = cls >= 0
    progress_target = np.where(valid, progress, 0.0).astype(np.float32)
    return cls, class_mask, progress_target, class_mask.copy()


def build_trust_labels(
    q: np.ndarray,
    *,
    ood_flag: np.ndarray | None = None,
    clip: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (risk_target [T] in [0,1], trust_mask [T]).

    Risk proxy = normalized |second difference of q| plus an optional OOD flag.
    This is an **offline proxy**, explicitly not an xTB success probability.
    """
    q = np.asarray(q, dtype=np.float32)
    n, k = q.shape
    risk = np.zeros(n, dtype=np.float32)
    if n >= 3:
        second = np.abs(q[2:] - 2.0 * q[1:-1] + q[:-2]).sum(axis=1)
        scale = float(np.percentile(second, 95)) if second.size else 1.0
        risk[1:-1] = np.clip(second / max(scale, 1e-6), 0.0, 1.0)
        mask = np.zeros(n, dtype=bool)
        mask[1:-1] = True
    else:
        mask = np.zeros(n, dtype=bool)
    if ood_flag is not None:
        flag = np.asarray(ood_flag, dtype=np.float32)
        if flag.shape[0] == n:
            risk = np.clip(np.maximum(risk, flag), 0.0, clip)
            mask = mask | (flag > 0)
    return risk * clip, mask
