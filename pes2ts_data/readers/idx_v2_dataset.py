"""V5.5-shaped numpy dataset view over ``v53mp_idx_v2`` (Zone 2, torch-free).

Split out of the ported ``pes2ts_v5/data/idx_v2.py`` so the memmap reader
(:mod:`pes2ts_data.readers.idx_v2`) stays a pure index and the V5.5-shaped view
stays free of torch: :class:`IdxV2Dataset` builds the same trajectory sequences
and Move/Process/Trust targets as the frozen V5.6 adapter, but yields plain
``dict[str, numpy.ndarray]`` samples (and lists of them) instead of torch
batches.  The v5 engine owns collation now.

Label mapping (V5.5-shaped, unchanged from the frozen adapter):

* **Move**: ``Δq_t = q_canonical[t+1] - q_canonical[t]`` along the canonical IRC
  coordinate, masked to contiguous ``label_status == ok`` pairs.
* **Process**: canonical coordinate ``q`` (TS at 0) -> pre/near/post with
  ``near_ts_window``.
* **Trust**: ``|d_key_ts_ep|`` offline proxy clipped to [0, 1], masked for
  degenerate curves / non-ok status.  Never an xTB success probability.

Split note: idx_v2 ``fold`` is sample-level (``position % 3``) and cannot be a
leak-free split; :meth:`IdxV2Dataset` therefore uses the frozen curve-level hash
split for smoke runs and marks it as *not* the V5.5 frozen ID/OOD protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ._common import ReaderError
from .idx_v2 import SPLIT_NAMES, STATUS_OK, IdxV2Error, IdxV2Reader

__all__ = [
    "IdxV2Dataset",
    "IdxV2Spec",
    "build_frame_features",
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
class _TrajectoryRef:
    """One (curve, intent) trajectory: sample rows of the idx_v2 sample table."""

    curve_idx: int
    intent_idx: int
    curve_id: str
    rows: np.ndarray


@dataclass
class IdxV2Dataset:
    """V5.5-shaped numpy dataset view over :class:`IdxV2Reader` (smoke stage).

    Parameters
    ----------
    reader:
        An open :class:`IdxV2Reader`.
    mode:
        Feature input mode (``G0`` / ``G1``).
    seed:
        Curve-hash split seed.
    max_curves:
        Optional per-split curve caps.
    history_len, near_ts_window:
        Sequence/feature parameters carried by the spec.
    """

    reader: IdxV2Reader
    mode: str = "G0"
    seed: int = 17
    max_curves: dict[str, int] | None = None
    history_len: int = 16
    near_ts_window: float = 0.10
    spec: IdxV2Spec = field(init=False)
    _splits: dict[str, list[int]] = field(default_factory=dict, init=False)
    _trajs: dict[str, list[_TrajectoryRef]] = field(default_factory=dict, init=False)
    _counts: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.max_curves = self.max_curves or {}
        self.spec = IdxV2Spec(
            dataset_version=self.reader.dataset_version,
            input_mode=str(self.mode).upper(),
            frame_feat_dim=7,
            static_feat_dim=8,
            q_dim=1,
            history_len=int(self.history_len),
            max_trajectory_len=512,
            near_ts_window=float(self.near_ts_window),
        )
        for split in ("train", "valid", "id"):
            self._build_split(split)
        self._counts = self._compute_counts()

    # ---------------------------------------------------------------- splits
    def _build_split(self, split: str) -> None:
        curves = self.reader.split_curves(
            split, seed=self.seed, max_curves=self.max_curves.get(split))
        intent_index = self.reader["samples/intent_idx"]
        trajectories: list[_TrajectoryRef] = []
        for curve_idx in curves:
            lo, hi = self.reader.sample_rows(curve_idx)
            if hi <= lo:
                continue
            intent_ids = np.unique(np.asarray(intent_index[lo:hi]))
            curve_id = (self.reader.curve_ids[curve_idx]
                        if curve_idx < len(self.reader.curve_ids) else str(curve_idx))
            for intent in intent_ids:
                rows = lo + np.flatnonzero(np.asarray(intent_index[lo:hi]) == intent)
                trajectories.append(_TrajectoryRef(curve_idx, int(intent), curve_id,
                                                   rows.astype(np.int64)))
        self._splits[split] = curves
        self._trajs[split] = trajectories

    def _compute_counts(self) -> dict[str, int]:
        n_frames = sum(int(ref.rows.shape[0])
                       for trajs in self._trajs.values() for ref in trajs)
        return {"n_curves": sum(len(v) for v in self._splits.values()),
                "n_trajectories": sum(len(v) for v in self._trajs.values()),
                "n_frames": n_frames, "n_decisions": n_frames}

    # ------------------------------------------------------------ properties
    @property
    def dataset_hash(self) -> str:
        return self.reader.dataset_hash

    @property
    def dataset_version(self) -> str:
        return self.reader.dataset_version

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def frame_feat_dim(self) -> int:
        return int(self.spec.frame_feat_dim)

    @property
    def static_feat_dim(self) -> int:
        return int(self.spec.static_feat_dim)

    @property
    def q_dim(self) -> int:
        return int(self.spec.q_dim)

    def reaction_ids_for_split(self, split: str) -> list[str]:
        return [self.reader.curve_ids[i] if i < len(self.reader.curve_ids) else str(i)
                for i in self._splits.get(split, [])]

    def split_counts(self) -> dict[str, int]:
        return {split: len(self._splits.get(split, [])) for split in SPLIT_NAMES}

    def trajectory_indices(self, split: str, *, fraction: float = 1.0,
                           seed: int = 17, min_reactions: int = 1) -> list[int]:
        n = len(self._trajs.get(split, []))
        if fraction >= 1.0 or n == 0:
            return list(range(n))
        rng = np.random.default_rng(seed)
        take = max(min_reactions, int(round(n * float(fraction))))
        return sorted(rng.choice(n, size=min(take, n), replace=False).tolist())

    def _static_feats(self, curve_idx: int) -> np.ndarray:
        meta = self.reader.curve_meta(curve_idx)
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

    # ------------------------------------------------------------- sequences
    def build_sequence(self, traj_idx: int) -> dict:
        """Build one V5.5-shaped numpy sequence dict (no torch)."""
        split = next((name for name, trajs in self._trajs.items()
                      if traj_idx < len(trajs)), None)
        if split is None:
            raise IdxV2Error("TRAJ_RANGE",
                             f"trajectory index {traj_idx} out of range")
        ref = self._trajs[split][traj_idx]
        nodes = self.reader.curve_node_arrays(ref.curve_idx)
        rows = ref.rows
        node_index = np.asarray(self.reader["samples/node_index"][rows], dtype=np.int64)
        status = np.asarray(self.reader["samples/label_status"][rows], dtype=np.int64)
        q = np.asarray(nodes["q"][node_index], dtype=np.float32)
        s = np.asarray(nodes["s"][node_index], dtype=np.float32)
        if "energy" in nodes:
            energy = np.asarray(nodes["energy"][node_index], dtype=np.float32)
        else:
            energy = np.zeros_like(q, dtype=np.float32)
        ds = np.zeros_like(s)
        if s.shape[0] > 1:
            ds[1:] = np.diff(s)
        move_target = np.zeros_like(q)
        move_mask = np.zeros(q.shape, dtype=bool)
        if node_index.shape[0] > 1:
            contiguous = np.ones(q.shape, dtype=bool)
            contiguous[1:] = np.diff(node_index) == 1
            ok = (status[:-1] == STATUS_OK) & (status[1:] == STATUS_OK) & contiguous[1:]
            move_target[:-1] = q[1:] - q[:-1]
            move_mask[:-1] = ok
        valid = status == STATUS_OK
        process_label = np.full(q.shape, -1, dtype=np.int64)
        process_label[valid & (q < -self.near_ts_window)] = 0
        process_label[valid & (np.abs(q) <= self.near_ts_window)] = 1
        process_label[valid & (q > self.near_ts_window)] = 2
        process_mask = process_label >= 0
        process_progress = np.where(valid, q, 0.0).astype(np.float32)
        d_key = np.asarray(self.reader["labels_ep/d_key_ts_ep"][rows], dtype=np.float32)
        degenerate = np.asarray(self.reader["labels_ep/degenerate_curve"][rows],
                                dtype=bool)
        trust_mask = valid & np.isfinite(d_key) & (~degenerate)
        trust_label = np.where(trust_mask, np.clip(np.abs(d_key), 0.0, 1.0),
                               0.0).astype(np.float32)
        q_feat = q.reshape(-1, 1) if q.ndim == 1 else q
        grad_norm = np.zeros(q.shape[0], dtype=np.float32)
        feats = build_frame_features(q_feat, ds, energy, grad_norm,
                                     input_mode=self.spec.input_mode)
        return {
            "traj_idx": int(traj_idx),
            "trajectory_id": f"{ref.curve_id}:{ref.intent_idx}",
            "reaction_id": ref.curve_id,
            "reaction_row": int(ref.curve_idx),
            "direction": 1,
            "frame_feats": feats,
            "static_feats": self._static_feats(ref.curve_idx),
            "delta_s": ds,
            "frame_mask": valid,
            "q": q,
            "s": s,
            "length": int(q.shape[0]),
            "move_target": move_target,
            "move_mask": move_mask,
            "process_label": process_label,
            "process_mask": process_mask,
            "process_progress": process_progress,
            "progress_mask": process_mask.copy(),
            "trust_label": trust_label,
            "trust_mask": trust_mask,
        }

    def iter_sequences(self, split: str, *, fraction: float = 1.0,
                       shuffle: bool = False, seed: int = 17,
                       max_samples: int | None = None
                       ) -> Iterator[tuple[list[dict], list[dict]]]:
        """Yield ``(samples, metas)`` chunks; collation belongs to the caller."""
        indices = self.trajectory_indices(split, fraction=fraction, seed=seed)
        if shuffle:
            rng = np.random.default_rng(seed)
            indices = [indices[i] for i in rng.permutation(len(indices))]
        if max_samples is not None:
            indices = indices[: int(max_samples)]
        samples = [self.build_sequence(index) for index in indices]
        metas = [{key: sample[key] for key in ("traj_idx", "trajectory_id",
                                               "reaction_id", "reaction_row",
                                               "direction", "length")}
                 for sample in samples]
        yield samples, metas

    # ------------------------------------------------------- not implemented
    def median_ds(self, traj_idx: int) -> float:
        sequence = self.build_sequence(traj_idx)
        ds = np.asarray(sequence["delta_s"][1:], dtype=np.float32)
        return float(np.median(ds)) if ds.size else 1.0

    def reference_q(self, traj_idx: int) -> np.ndarray:
        return np.asarray(self.build_sequence(traj_idx)["q"], dtype=np.float32)

    def geom_for(self, reaction_row: int):
        """idx_v2 has no per-reaction kinematic basis (honest refusal, R1)."""
        raise ReaderError(
            "GEOMETRY_UNAVAILABLE",
            "idx_v2 has no per-reaction kinematic basis; R1 rollout on real data "
            "needs the XYZ geometry adapter (staging_chunks) — reported as a "
            "limitation, not fabricated")
