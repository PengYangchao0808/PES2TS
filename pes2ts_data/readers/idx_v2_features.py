"""Feature/spec primitives of the numpy ``idx_v2`` dataset view (Zone 2).

Split out of :mod:`pes2ts_data.readers.idx_v2_dataset` (which was ported from
``pes2ts_v5/data/idx_v2.py`` together with ``pes2ts_v5/data/trajectories.py``)
so both files stay within the project's 350-line rule.  Public surface is still
re-exported from :mod:`pes2ts_data.readers`; the dataset module also re-exports
these names for direct imports.

Nothing here imports torch: the frame features are pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "IdxV2Spec",
    "TrajectoryRef",
    "build_frame_features",
    "static_feats",
]


@dataclass(frozen=True)
class IdxV2Spec:
    """Minimal V5.5 dataset spec (subset of the frozen ``V55DatasetSpec``)."""

    dataset_version: str = "unknown"
    input_mode: str = "G0"
    frame_feat_dim: int = 7
    static_feat_dim: int = 8
    q_dim: int = 1
    history_len: int = 16
    max_trajectory_len: int = 512
    near_ts_window: float = 0.10
    notes: str = "numpy idx_v2 adapter (smoke split = curve hash; V5.5 contract pending)"


def build_frame_features(
    q: np.ndarray,
    ds: np.ndarray,
    energy: np.ndarray,
    grad_norm: np.ndarray,
    *,
    input_mode: str = "G0",
) -> np.ndarray:
    """Build the causal reaction-descriptor frame features (ported verbatim).

    Only current/past information is used (``dq`` is a backward difference), so
    the features are causal and can be regenerated during free-run rollout.
    ``q`` is ``[T, q_dim]``; the returned array is ``[T, 2*q_dim + 5]``.
    """
    q = np.asarray(q, dtype=np.float32)
    n, _ = q.shape
    ds = np.asarray(ds, dtype=np.float32)
    energy = np.asarray(energy, dtype=np.float32)
    grad_norm = np.asarray(grad_norm, dtype=np.float32)
    dq = np.zeros_like(q)
    if n >= 2:
        dq[1:] = q[1:] - q[:-1]
    has_qc = (np.ones(n, dtype=np.float32) if input_mode.upper() == "G1"
              else np.zeros(n, dtype=np.float32))
    energy = energy * has_qc
    grad_norm = grad_norm * has_qc
    step_norm = np.linalg.norm(dq, axis=1).astype(np.float32)
    feats = np.concatenate(
        [q, dq, ds[:, None], energy[:, None], grad_norm[:, None],
         has_qc[:, None], step_norm[:, None]],
        axis=1,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@dataclass(frozen=True)
class TrajectoryRef:
    """One (curve, intent) trajectory: sample rows of the idx_v2 sample table."""

    curve_idx: int
    intent_idx: int
    curve_id: str
    rows: np.ndarray


def static_feats(reader: Any, curve_idx: int) -> np.ndarray:
    """Per-reaction static feature vector (8 dims, frozen V5.6 adapter shape)."""
    meta = reader.curve_meta(curve_idx)
    n_nodes = max(int(meta.get("n_nodes", 1)), 1)
    n_atoms = len(meta.get("atomic_numbers", ()) or ())
    ts_index = float(meta.get("ts_index", 0))
    return np.asarray([
        n_atoms / 32.0,
        n_nodes / 120.0,
        ts_index / n_nodes,
        1.0 if meta.get("has_energy") else 0.0,
        abs(int(meta.get("endpoint_b", 0)) - int(meta.get("endpoint_a", 0))) / n_nodes,
        float(meta.get("weight_total", 1.0)),
        0.0,
        1.0,
    ], dtype=np.float32)
