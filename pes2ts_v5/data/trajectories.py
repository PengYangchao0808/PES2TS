"""Trajectory-level V5 dataset: loading, batching, and the synthetic fixture.

Implements the V5.5-shaped artifact layout from :mod:`pes2ts_v5.data.manifest`
and a deterministic **synthetic fixture** used until the real V5.5 contract and
data-land are ready (task-10 hard constraint: real 50k tables are read-only
reference and are not directly depended on yet).

Key discipline (plan v2.1 §2.4/§3.4):

* sequences are **variable-length trajectories**, padded only at collate time;
* all frames/branches/resamples of one reaction live in the same Group;
* labels and masks come from :mod:`pes2ts_v5.data.labels` (never filled);

Frame feature vector (``frame_feat_dim = 2 * q_dim + 5``):

    q[k], dq[k] (q_t - q_{t-1}), ds (path interval), energy, grad_norm,
    has_qc (0 in G0, 1 in G1), step_norm (|dq|)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from pes2ts_v5.data.labels import (
    LABEL_SPEC,
    build_move_labels,
    build_process_labels,
    build_trust_labels,
)
from pes2ts_v5.data.manifest import (
    ARTIFACT_FILES,
    DatasetManifest,
    V55DatasetSpec,
    compute_dataset_hash,
    sha256_file,
)
from pes2ts_v5.data.split import (
    SPLIT_NAMES,
    check_group_leak,
    compute_normalization_stats,
)

FRAME_FEATURE_NAMES = (
    "q0", "q1", "q2", "dq0", "dq1", "dq2",
    "ds", "energy", "grad_norm", "has_qc", "step_norm",
)
STATIC_FEATURE_NAMES = (
    "n_atoms_norm", "n_bonds_norm", "n_formed_norm", "n_broken_norm",
    "n_key_norm", "family_norm", "size_bucket_norm", "bias",
)


def build_frame_features(
    q: np.ndarray,
    ds: np.ndarray,
    energy: np.ndarray,
    grad_norm: np.ndarray,
    *,
    input_mode: str = "G0",
) -> np.ndarray:
    """Build the low-dimensional reaction-descriptor frame features.

    Only current/past information is used (``dq`` is a backward difference), so
    features are causal and can be regenerated during free-run rollout.
    """
    q = np.asarray(q, dtype=np.float32)
    n, k = q.shape
    ds = np.asarray(ds, dtype=np.float32)
    energy = np.asarray(energy, dtype=np.float32)
    grad_norm = np.asarray(grad_norm, dtype=np.float32)
    dq = np.zeros_like(q)
    if n >= 2:
        dq[1:] = q[1:] - q[:-1]
    has_qc = np.ones(n, dtype=np.float32) if input_mode.upper() == "G1" else np.zeros(n, dtype=np.float32)
    energy = energy * has_qc
    grad_norm = grad_norm * has_qc
    step_norm = np.linalg.norm(dq, axis=1).astype(np.float32)
    feats = np.concatenate(
        [q, dq, ds[:, None], energy[:, None], grad_norm[:, None],
         has_qc[:, None], step_norm[:, None]],
        axis=1,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class V5TrajectoryDataset:
    """Read a V5.5-shaped dataset artifact directory."""

    def __init__(self, data_dir: str | Path, *, mmap: bool = False):
        self.data_dir = Path(data_dir)
        self.manifest = DatasetManifest.load(self.data_dir)
        self.spec = self.manifest.spec
        if not self.data_dir.joinpath("splits.json").exists():
            raise FileNotFoundError(f"missing splits.json under {self.data_dir}")
        self.splits: dict[str, list[str]] = json.loads(
            (self.data_dir / "splits.json").read_text(encoding="utf-8"))
        self._load_arrays(mmap=mmap)

    # ------------------------------------------------------------------ load
    def _load_arrays(self, *, mmap: bool) -> None:
        reac = np.load(self.data_dir / "reactions.npz", allow_pickle=False)
        self.reaction_ids: list[str] = [str(x) for x in reac["reaction_ids"]]
        self.family_ids = np.asarray(reac["family_ids"])
        self.scaffold_ids = np.asarray(reac["scaffold_ids"])
        self.n_atoms = np.asarray(reac["n_atoms"], dtype=np.int64)
        self.static_feats = np.asarray(reac["static_feats"], dtype=np.float32)
        traj = np.load(self.data_dir / "trajectories.npz", allow_pickle=False)
        self.traj_ids: list[str] = [str(x) for x in traj["traj_ids"]]
        self.traj_reaction_idx = np.asarray(traj["reaction_idx"], dtype=np.int64)
        self.traj_direction = np.asarray(traj["direction"], dtype=np.int64)
        self.traj_lengths = np.asarray(traj["length"], dtype=np.int64)
        self.traj_offsets = np.asarray(traj["frame_offset"], dtype=np.int64)
        frames = np.load(self.data_dir / "frames.npz", allow_pickle=False)
        # frames are lazily sliced: keep the npz file open-backed arrays
        self._frames: dict[str, np.ndarray] = {
            k: np.asarray(frames[k]) for k in frames.files
        }
        geom_path = self.data_dir / "geom.npz"
        self._geom: dict[str, np.ndarray] | None = None
        if geom_path.exists():
            geom = np.load(geom_path, allow_pickle=False)
            self._geom = {k: np.asarray(geom[k]) for k in geom.files}
        self._reaction_index = {rid: i for i, rid in enumerate(self.reaction_ids)}

    # ------------------------------------------------------------- properties
    @property
    def dataset_hash(self) -> str:
        return self.manifest.dataset_hash

    @property
    def dataset_version(self) -> str:
        return self.manifest.dataset_version

    @property
    def counts(self) -> dict[str, int]:
        return dict(self.manifest.counts)

    @property
    def frame_feat_dim(self) -> int:
        return int(self.spec.frame_feat_dim)

    @property
    def static_feat_dim(self) -> int:
        return int(self.spec.static_feat_dim)

    @property
    def q_dim(self) -> int:
        return int(self.spec.q_dim)

    # ---------------------------------------------------------------- splits
    def reaction_ids_for_split(self, split: str) -> list[str]:
        if split not in self.splits:
            raise KeyError(f"split {split!r} not in dataset {list(self.splits)}")
        return list(self.splits[split])

    def split_counts(self) -> dict[str, int]:
        return {s: len(self.splits.get(s, [])) for s in SPLIT_NAMES}

    def trajectory_indices(
        self,
        split: str,
        *,
        fraction: float = 1.0,
        seed: int = 17,
        min_reactions: int = 1,
    ) -> list[int]:
        """Group-level subsample: pick reactions first, then all their trajectories."""
        rids = self.reaction_ids_for_split(split)
        if not rids:
            return []
        if fraction < 1.0:
            rng = np.random.default_rng(seed)
            n_take = max(min_reactions, int(round(len(rids) * float(fraction))))
            n_take = min(n_take, len(rids))
            chosen = set(rng.choice(len(rids), size=n_take, replace=False).tolist())
            rids = [r for i, r in enumerate(rids) if i in chosen]
        want = {self._reaction_index[r] for r in rids}
        out = [i for i, ri in enumerate(self.traj_reaction_idx) if int(ri) in want]
        return out

    # --------------------------------------------------------------- access
    def frame_slice(self, t: int) -> tuple[int, int]:
        return int(self.traj_offsets[t]), int(self.traj_offsets[t] + self.traj_lengths[t])

    def reference_q(self, traj_idx: int) -> np.ndarray:
        lo, hi = self.frame_slice(traj_idx)
        return np.asarray(self._frames["q"][lo:hi], dtype=np.float32).copy()

    def median_ds(self, traj_idx: int) -> float:
        lo, hi = self.frame_slice(traj_idx)
        ds = np.asarray(self._frames["ds"][lo:hi], dtype=np.float32)
        return float(np.median(ds[1:])) if ds.size > 1 else 1.0

    def reaction_static(self, reaction_row: int) -> np.ndarray:
        return np.asarray(self.static_feats[reaction_row], dtype=np.float32)

    def geom_for(self, reaction_row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (basis [3A,K], x0 [3A], n_atoms) for R1 kinematic rollout."""
        if self._geom is None:
            raise NotImplementedError(
                "geom.npz absent: this dataset cannot run R1 kinematic rollout")
        off = np.asarray(self._geom["geom_offset"], dtype=np.int64)
        lo, hi = int(off[reaction_row]), int(off[reaction_row + 1])
        basis = np.asarray(self._geom["basis"][lo:hi], dtype=np.float32)
        x0 = np.asarray(self._geom["x0"][lo:hi], dtype=np.float32)
        return basis, x0, int(self.n_atoms[reaction_row])

    # ------------------------------------------------------------- sequence
    def build_sequence(self, traj_idx: int) -> dict:
        lo, hi = self.frame_slice(traj_idx)
        f = {k: np.asarray(v[lo:hi]) for k, v in self._frames.items()}
        q = f["q"].astype(np.float32)
        ds = f["ds"].astype(np.float32)
        energy = f["energy"].astype(np.float32)
        grad_norm = f["grad_norm"].astype(np.float32)
        move_target = f["move_target"].astype(np.float32)
        move_mask = f["move_mask"].astype(bool)
        process_label = f["process_label"].astype(np.int64)
        process_mask = f["process_mask"].astype(bool)
        process_progress = f["process_progress"].astype(np.float32)
        progress_mask = f["progress_mask"].astype(bool)
        trust_label = f["trust_label"].astype(np.float32)
        trust_mask = f["trust_mask"].astype(bool)
        feats = build_frame_features(
            q, ds, energy, grad_norm, input_mode=self.spec.input_mode)
        frame_mask = np.ones(q.shape[0], dtype=bool)
        ri = int(self.traj_reaction_idx[traj_idx])
        return {
            "traj_idx": int(traj_idx),
            "trajectory_id": self.traj_ids[traj_idx],
            "reaction_id": self.reaction_ids[ri],
            "reaction_row": ri,
            "direction": int(self.traj_direction[traj_idx]),
            "frame_feats": feats,
            "static_feats": self.reaction_static(ri),
            "delta_s": ds,
            "frame_mask": frame_mask,
            "q": q,
            "s": np.cumsum(ds).astype(np.float32),
            "length": int(q.shape[0]),
            "move_target": move_target,
            "move_mask": move_mask,
            "process_label": process_label,
            "process_mask": process_mask,
            "process_progress": process_progress,
            "progress_mask": progress_mask,
            "trust_label": trust_label,
            "trust_mask": trust_mask,
        }

    def iter_batches(
        self,
        split: str,
        *,
        batch_size: int = 4,
        shuffle: bool = False,
        seed: int = 17,
        fraction: float = 1.0,
        max_batches: int | None = None,
        drop_last: bool = False,
    ) -> Iterator[tuple[dict, list[dict]]]:
        idx = self.trajectory_indices(split, fraction=fraction, seed=seed)
        if shuffle:
            rng = np.random.default_rng(seed)
            idx = [idx[i] for i in rng.permutation(len(idx))]
        meta: list[dict] = []
        for start in range(0, len(idx), batch_size):
            chunk = idx[start:start + batch_size]
            if drop_last and len(chunk) < batch_size:
                break
            samples = [self.build_sequence(i) for i in chunk]
            batch = collate_samples(samples)
            metas = [{k: s[k] for k in ("traj_idx", "trajectory_id", "reaction_id",
                                        "reaction_row", "direction", "length")}
                     for s in samples]
            meta.extend(metas)
            yield batch, metas
            if max_batches is not None and (start // batch_size + 1) >= max_batches:
                break


# ---------------------------------------------------------------------------
# Collate (variable-length trajectories -> padded tensors)
# ---------------------------------------------------------------------------

def _pad_stack(arrays: list[np.ndarray], fill: float = 0.0) -> torch.Tensor:
    max_len = max(a.shape[0] for a in arrays)
    out = np.full((len(arrays), max_len) + arrays[0].shape[1:], fill,
                  dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, :a.shape[0]] = a
    return torch.from_numpy(out)


def _pad_bool(arrays: list[np.ndarray]) -> torch.Tensor:
    max_len = max(a.shape[0] for a in arrays)
    out = np.zeros((len(arrays), max_len), dtype=bool)
    for i, a in enumerate(arrays):
        out[i, :a.shape[0]] = a
    return torch.from_numpy(out)


def _pad_int(arrays: list[np.ndarray], fill: int = -1) -> torch.Tensor:
    max_len = max(a.shape[0] for a in arrays)
    out = np.full((len(arrays), max_len), fill, dtype=np.int64)
    for i, a in enumerate(arrays):
        out[i, :a.shape[0]] = a
    return torch.from_numpy(out)


def collate_samples(samples: list[dict]) -> dict:
    """Pad variable-length trajectory samples into a CPU tensor batch.

    Masks are AND-ed with the padded frame mask so padding never enters a loss.
    """
    lengths = torch.tensor([s["length"] for s in samples], dtype=torch.int64)
    valid = _pad_bool([np.ones(s["length"], dtype=bool) for s in samples])
    move_mask = _pad_bool([s["move_mask"] for s in samples]) & valid
    process_mask = _pad_bool([s["process_mask"] for s in samples]) & valid
    progress_mask = _pad_bool([s["progress_mask"] for s in samples]) & valid
    trust_mask = _pad_bool([s["trust_mask"] for s in samples]) & valid
    batch = {
        "frame_feats": _pad_stack([s["frame_feats"] for s in samples]),
        "static_feats": torch.from_numpy(
            np.stack([s["static_feats"] for s in samples]).astype(np.float32)),
        "delta_s": _pad_stack([s["delta_s"] for s in samples]),
        "lengths": lengths,
        "frame_mask": valid,
        "move_target": _pad_stack([s["move_target"] for s in samples]),
        "move_mask": move_mask,
        "process_label": _pad_int([s["process_label"] for s in samples]),
        "process_mask": process_mask,
        "process_progress": _pad_stack([s["process_progress"] for s in samples]),
        "progress_mask": progress_mask,
        "trust_label": _pad_stack([s["trust_label"] for s in samples]),
        "trust_mask": trust_mask,
        "q": _pad_stack([s["q"] for s in samples]),
        "s": _pad_stack([s["s"] for s in samples]),
    }
    return batch


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

#: split plan for the default 24-reaction synthetic fixture:
#: (split, n_reactions, families, scaffolds, n_atoms_range)
_SYNTH_PLAN = (
    ("train", 12, (0, 1), (0, 1, 2), (4, 8)),
    ("valid", 4, (0, 1), (0, 1), (4, 8)),
    ("id", 3, (0, 1), (0, 1), (4, 8)),
    ("scaffold_ood", 2, (0, 1), (4, 5), (4, 8)),
    ("family_ood", 2, (2, 3), (2, 3), (4, 8)),
    ("size_ood", 1, (0,), (0,), (12, 14)),
)


def build_synthetic_dataset(
    out_dir: str | Path,
    *,
    seed: int = 17,
    n_trajectories_per_reaction: int = 2,
    spec: V55DatasetSpec | None = None,
) -> Path:
    """Write a deterministic synthetic V5.5-shaped dataset; return its directory.

    The fixture exercises the full contract: variable-length trajectories,
    irregular ``ds``, all four splits, Move/Process/Trust labels, masks including
    ``no_future_step``, and per-reaction kinematic geometry (R1 rollout).
    """
    spec = spec or V55DatasetSpec()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    k = int(spec.q_dim)

    reaction_ids: list[str] = []
    family_ids: list[int] = []
    scaffold_ids: list[int] = []
    n_atoms_list: list[int] = []
    static_rows: list[np.ndarray] = []
    splits: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}
    geom_basis_rows: list[np.ndarray] = []
    geom_x0_rows: list[np.ndarray] = []
    geom_offset: list[int] = [0]

    traj_ids: list[str] = []
    traj_reaction_idx: list[int] = []
    traj_direction: list[int] = []
    traj_length: list[int] = []
    traj_offset: list[int] = [0]

    frame_cols = {name: [] for name in (
        "q", "ds", "energy", "grad_norm", "move_target", "move_mask",
        "process_label", "process_mask", "process_progress", "progress_mask",
        "trust_label", "trust_mask")}

    for split, n_react, families, scaffolds, atom_range in _SYNTH_PLAN:
        for j in range(n_react):
            rid = f"R{len(reaction_ids):04d}_{split}"
            family = int(families[j % len(families)])
            scaffold = int(scaffolds[j % len(scaffolds)])
            n_atoms = int(rng.integers(atom_range[0], atom_range[1] + 1))
            reaction_ids.append(rid)
            family_ids.append(family)
            scaffold_ids.append(scaffold)
            n_atoms_list.append(n_atoms)
            splits[split].append(rid)
            n_bonds = max(2, n_atoms - 1)
            static_rows.append(np.array([
                n_atoms / 12.0, n_bonds / 12.0,
                rng.integers(0, 3) / 4.0, rng.integers(0, 3) / 4.0,
                rng.integers(1, 4) / 4.0, family / 4.0,
                0.0 if split != "size_ood" else 1.0, 1.0,
            ], dtype=np.float32))
            # kinematic geometry basis: orthonormal K columns in R^{3A}
            dim = 3 * n_atoms
            mat = rng.normal(0.0, 1.0, (dim, k))
            basis, _ = np.linalg.qr(mat)
            basis = basis[:, :k].astype(np.float32)
            x0 = rng.normal(0.0, 0.5, dim).astype(np.float32)
            geom_basis_rows.append(basis)
            geom_x0_rows.append(x0)
            geom_offset.append(geom_offset[-1] + dim)
            reaction_row = len(reaction_ids) - 1
            direction_vec = rng.normal(0.0, 1.0, k)
            direction_vec /= max(np.linalg.norm(direction_vec), 1e-6)
            bend_vec = rng.normal(0.0, 1.0, k)
            bend_vec /= max(np.linalg.norm(bend_vec), 1e-6)
            for t_i in range(n_trajectories_per_reaction):
                direction = 1 if t_i % 2 == 0 else -1
                length = int(rng.integers(8, 25))
                progress = np.linspace(-1.0, 1.0, length).astype(np.float32)
                q = (progress[:, None] * direction_vec
                     + 0.20 * np.sin(np.pi * progress)[:, None] * bend_vec)
                q += 0.01 * rng.normal(0.0, 1.0, q.shape).astype(np.float32)
                ds = rng.uniform(0.5, 1.5, length).astype(np.float32)
                ds[0] = 0.0
                energy = (0.5 * progress ** 2
                          + 0.05 * rng.normal(0.0, 1.0, length)).astype(np.float32)
                grad_norm = (np.abs(progress) + 0.05
                             * rng.normal(0.0, 1.0, length)).astype(np.float32)
                move_target, move_mask = build_move_labels(q)
                p_cls, p_mask, p_prog, prog_mask = build_process_labels(
                    progress, near_ts_window=spec.near_ts_window)
                trust, trust_mask = build_trust_labels(
                    q, ood_flag=(progress > 0.95).astype(np.float32))
                if split == "size_ood":
                    trust = np.clip(trust + 0.15, 0.0, 1.0).astype(np.float32)
                traj_ids.append(f"T{len(traj_ids):05d}")
                traj_reaction_idx.append(reaction_row)
                traj_direction.append(direction)
                traj_length.append(length)
                traj_offset.append(traj_offset[-1] + length)
                frame_cols["q"].append(q)
                frame_cols["ds"].append(ds)
                frame_cols["energy"].append(energy)
                frame_cols["grad_norm"].append(grad_norm)
                frame_cols["move_target"].append(move_target)
                frame_cols["move_mask"].append(move_mask)
                frame_cols["process_label"].append(p_cls)
                frame_cols["process_mask"].append(p_mask)
                frame_cols["process_progress"].append(p_prog)
                frame_cols["progress_mask"].append(prog_mask)
                frame_cols["trust_label"].append(trust)
                frame_cols["trust_mask"].append(trust_mask)

    def _cat(key: str, dtype=np.float32) -> np.ndarray:
        return np.concatenate(frame_cols[key]).astype(dtype)

    q_all = _cat("q")
    ds_all = _cat("ds")
    frames = {
        "q": q_all,
        "ds": ds_all,
        "energy": _cat("energy"),
        "grad_norm": _cat("grad_norm"),
        "move_target": _cat("move_target"),
        "move_mask": _cat("move_mask", bool),
        "process_label": _cat("process_label", np.int64),
        "process_mask": _cat("process_mask", bool),
        "process_progress": _cat("process_progress"),
        "progress_mask": _cat("progress_mask", bool),
        "trust_label": _cat("trust_label"),
        "trust_mask": _cat("trust_mask", bool),
    }
    np.savez(out / "reactions.npz",
             reaction_ids=np.asarray(reaction_ids, dtype="U64"),
             family_ids=np.asarray(family_ids, dtype=np.int64),
             scaffold_ids=np.asarray(scaffold_ids, dtype=np.int64),
             n_atoms=np.asarray(n_atoms_list, dtype=np.int64),
             static_feats=np.stack(static_rows).astype(np.float32))
    np.savez(out / "trajectories.npz",
             traj_ids=np.asarray(traj_ids, dtype="U64"),
             reaction_idx=np.asarray(traj_reaction_idx, dtype=np.int64),
             direction=np.asarray(traj_direction, dtype=np.int64),
             length=np.asarray(traj_length, dtype=np.int64),
             frame_offset=np.asarray(traj_offset, dtype=np.int64))
    np.savez(out / "frames.npz", **frames)
    np.savez(out / "geom.npz",
             geom_offset=np.asarray(geom_offset, dtype=np.int64),
             basis=np.concatenate(geom_basis_rows, axis=0).astype(np.float32),
             x0=np.concatenate(geom_x0_rows).astype(np.float32))
    (out / "splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    # normalization stats: training frames only (plan §2.4)
    train_reactions = set(splits["train"])
    train_mask = np.zeros(q_all.shape[0], dtype=bool)
    for t_i, ri in enumerate(traj_reaction_idx):
        if reaction_ids[ri] in train_reactions:
            lo, hi = traj_offset[t_i], traj_offset[t_i + 1]
            train_mask[lo:hi] = True
    feats_all = build_frame_features(q_all, ds_all, frames["energy"],
                                     frames["grad_norm"], input_mode=spec.input_mode)
    norm = compute_normalization_stats(feats_all, train_mask)

    leak = check_group_leak(reaction_ids, splits)
    if not leak["ok"]:
        raise RuntimeError(f"synthetic split leak: {leak['violations']}")

    files = {}
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    n_decisions = int(np.asarray(frames["move_mask"]).sum())
    manifest = DatasetManifest(
        spec=spec,
        dataset_version=f"synthetic-v{seed}",
        dataset_hash=compute_dataset_hash(out),
        files=files,
        counts={
            "n_reactions": len(reaction_ids),
            "n_trajectories": len(traj_ids),
            "n_frames": int(q_all.shape[0]),
            "n_decisions": n_decisions,
        },
        splits_counts={s: len(v) for s, v in splits.items()},
        normalization=norm,
        label_spec=dict(LABEL_SPEC),
        notes="deterministic synthetic fixture for V5.6 smoke/tests; not real data",
    )
    manifest.save(out)
    return out


__all__ = [
    "FRAME_FEATURE_NAMES",
    "STATIC_FEATURE_NAMES",
    "V5TrajectoryDataset",
    "build_frame_features",
    "build_synthetic_dataset",
    "collate_samples",
]
