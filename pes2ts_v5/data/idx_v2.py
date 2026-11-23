"""Real-data adapter: ``IdxV2Reader`` over the frozen ``v53mp_idx_v2`` index.

This is the V5.6 real-data path recommended by the data owner: read the
pre-built trajectory index (numpy memmaps) instead of parsing the 24 GB
``sample_view.csv``.  Layout (``/pes2ts_data/v53mp_50k_data/idx_v2``)::

    curves.json                  # schema v53mp_idx_v2: binding/n_curves/n_intents/
                                 # n_samples + per-curve metadata + file hashes
    samples/*.npy                # per-decision-sample columns (14,685,162)
    label_curves/*.npy           # per-node: q_canonical, arc_s, node_order, offsets
    labels_ep/*.npy              # per-sample: ts_row, start_row, terminal_row,
                                 # d_key_ts_ep, degenerate_curve, ...
    labels_v52/*.npy             # per-sample: q progress, q_mask
    curve_node_energies.npy      # per-node energies (G1 cached-QC only)
    staging_chunks/chunk_*.npz   # real XYZ geometry shards (used by geometry smoke)

Label mapping (V5.5-shaped, see ``pes2ts_v5.data.labels``):

* **Move**: displacement ``Δq_t = q_canonical[t+1] - q_canonical[t]`` along the
  canonical IRC coordinate (equivalent to ``tangent * Δs`` with tangent
  ``dq/ds = Δq / Δs``).  No ``v_C`` (Classic reference direction) is present in
  idx_v2 -> ``classic_dir=None`` branch, residual BC needs a later adapter.
* **Process**: canonical coordinate ``q`` (TS at 0) -> pre/near/post class with
  ``near_ts_window``; per-node energies are **not** used in G0.
* **Trust**: ``|d_key_ts_ep|`` large-error proxy clipped to [0,1]; masked where
  ``degenerate_curve`` or ``label_status != ok``.  This is an offline proxy,
  never an xTB success probability.

Split note: idx_v2 ``fold`` is a **sample-level** column (``fold = position % 3``)
and cannot be used for a leak-free split.  This adapter therefore uses a
curve-level hash split (``curve_idx``-grouped) for smoke/training runs and
marks it explicitly as *not* the V5.5 frozen ID/OOD protocol; the V5.5 contract
adapter will replace it once the contract lands.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pes2ts_v5.data.manifest import V55DatasetSpec
from pes2ts_v5.data.split import SPLIT_NAMES
from pes2ts_v5.data.trajectories import build_frame_features, collate_samples

DEFAULT_IDX_ROOT = "/pes2ts_data/v53mp_50k_data/idx_v2"
DEFAULT_STAGING_ROOT = "/pes2ts_data/v53mp_50k_data/staging_chunks"
SCHEMA = "v53mp_idx_v2"
# label_status codes from curves.json["label_status_table"]
STATUS_CONFLICT, STATUS_MASKED, STATUS_OK, STATUS_TERMINAL = 0, 1, 2, 3


def idx_v2_available(root: str | Path | None = None) -> bool:
    base = Path(root or os.environ.get("PES2TS_V5_IDX_V2_DIR", DEFAULT_IDX_ROOT))
    return (base / "curves.json").exists()


class IdxV2Error(RuntimeError):
    """Typed refusal for missing/incompatible idx_v2 artifacts."""


def read_staging_chunk(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Read one real-geometry staging chunk: ``{reaction_id: {coordinates, ...}}``."""
    data = np.load(path, allow_pickle=False)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        reaction_id, _, field = key.partition("::")
        out.setdefault(reaction_id, {})[field] = np.asarray(data[key])
    return out


class IdxV2Reader:
    """Memmap reader over ``v53mp_idx_v2`` (read-only)."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.environ.get("PES2TS_V5_IDX_V2_DIR",
                                                DEFAULT_IDX_ROOT))
        curves_path = self.root / "curves.json"
        if not curves_path.exists():
            raise IdxV2Error(
                f"idx_v2 not found at {self.root} (set PES2TS_V5_IDX_V2_DIR); "
                "real-data smoke requires the WSL 50k index, synthetic fixture "
                "is used otherwise")
        self.curves_doc = json.loads(curves_path.read_text(encoding="utf-8"))
        if self.curves_doc.get("schema") != SCHEMA:
            raise IdxV2Error(f"unexpected idx schema {self.curves_doc.get('schema')!r}")
        self.binding: dict = self.curves_doc.get("binding", {})
        self.curve_ids: list[str] = list(self.curves_doc.get("curve_ids", []))
        self.curves: list[dict] = list(self.curves_doc.get("curves", []))
        self.label_status_table: list[str] = list(
            self.curves_doc.get("label_status_table", []))
        self._arrays: dict[str, np.ndarray] = {}
        self._load_arrays()

    # ------------------------------------------------------------------ load
    def _load_arrays(self) -> None:
        groups = {
            "samples": ("curve_idx", "intent_idx", "node_index", "label_status",
                        "fold", "d0", "action_off", "action_len", "sample_weight",
                        "label_weight", "curve_weight"),
            "label_curves": ("q_canonical", "arc_s", "node_order", "offsets"),
            "labels_ep": ("ts_row", "start_row", "terminal_row", "d_key_ts_ep",
                          "degenerate_curve", "ep_label", "geometry_legal"),
            "labels_v52": ("q", "q_mask"),
        }
        for sub, names in groups.items():
            for name in names:
                path = self.root / sub / f"{name}.npy"
                if path.exists():
                    self._arrays[f"{sub}/{name}"] = np.load(path, mmap_mode="r")
        energies = self.root / "curve_node_energies.npy"
        if energies.exists():
            self._arrays["curve_node_energies"] = np.load(energies, mmap_mode="r")
        for required in ("label_curves/q_canonical", "label_curves/arc_s",
                         "label_curves/offsets", "samples/curve_idx",
                         "labels_ep/ts_row", "labels_ep/d_key_ts_ep"):
            if required not in self._arrays:
                raise IdxV2Error(f"missing required idx_v2 array: {required}")

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            raise IdxV2Error(f"array {name!r} not available in idx_v2")
        return self._arrays[name]

    # ------------------------------------------------------------ properties
    @property
    def n_curves(self) -> int:
        return int(self.binding.get("n_curves", len(self.curves)))

    @property
    def n_intents(self) -> int:
        return int(self.binding.get("n_intents", 0))

    @property
    def n_samples(self) -> int:
        return int(self.binding.get("n_samples", self.__getitem__("samples/curve_idx").shape[0]))

    @property
    def n_nodes(self) -> int:
        return int(self.__getitem__("label_curves/q_canonical").shape[0])

    @property
    def data_manifest_sha256(self) -> str | None:
        return self.binding.get("data_manifest_sha256")

    @property
    def dataset_hash(self) -> str:
        """Hash recorded from the frozen file registry in ``curves.json``."""
        files = self.curves_doc.get("files") or {}
        lines = []
        for name in sorted(files):
            rec = files[name]
            sha = rec.get("sha256") if isinstance(rec, dict) else str(rec)
            lines.append(f"{name}:{sha}")
        payload = "\n".join(lines) + ("\n" + str(self.binding.get("data_manifest_sha256")))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def dataset_version(self) -> str:
        created = str(self.curves_doc.get("created_at", "unknown"))[:10]
        commit = str(self.curves_doc.get("builder_git_commit", ""))[:8]
        return f"idx_v2-{created}-{commit}"

    def counts(self) -> dict[str, int]:
        return {"n_curves": self.n_curves, "n_intents": self.n_intents,
                "n_frames": self.n_nodes, "n_samples": self.n_samples}

    def manifest_doc(self) -> dict:
        return {
            "schema": SCHEMA,
            "source": str(self.root),
            "binding": self.binding,
            "builder_git_commit": self.curves_doc.get("builder_git_commit"),
            "created_at": self.curves_doc.get("created_at"),
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "counts": self.counts(),
            "note": "idx_v2 is an index over the frozen 50k tables; frame-level "
                    "counts are nodes, decision-level counts are samples",
        }

    # -------------------------------------------------------------- per curve
    def curve_meta(self, curve_idx: int) -> dict:
        return self.curves[int(curve_idx)]

    def curve_node_arrays(self, curve_idx: int) -> dict[str, np.ndarray]:
        """Return per-node arrays in node-index order for one curve."""
        offs = self.__getitem__("label_curves/offsets")
        lo, hi = int(offs[curve_idx]), int(offs[curve_idx + 1])
        order = self.__getitem__("label_curves/node_order")[lo:hi]
        n = hi - lo
        row_for_node = np.empty(n, dtype=np.int64)
        row_for_node[np.asarray(order, dtype=np.int64)] = np.arange(n)
        out = {
            "q": np.asarray(self.__getitem__("label_curves/q_canonical")[lo:hi],
                            dtype=np.float32)[row_for_node],
            "s": np.asarray(self.__getitem__("label_curves/arc_s")[lo:hi],
                            dtype=np.float32)[row_for_node],
        }
        if "curve_node_energies" in self._arrays:
            out["energy"] = np.asarray(
                self._arrays["curve_node_energies"][lo:hi], dtype=np.float32)[row_for_node]
        return out

    def sample_rows(self, curve_idx: int) -> tuple[int, int]:
        ci = self.__getitem__("samples/curve_idx")
        return (int(np.searchsorted(ci, curve_idx, side="left")),
                int(np.searchsorted(ci, curve_idx, side="right")))

    def split_curves(self, split: str, *, seed: int = 17,
                     max_curves: int | None = None) -> list[int]:
        """Curve-level hash split for smoke runs (NOT the frozen V5.5 OOD split)."""
        rng_hash = (np.arange(self.n_curves, dtype=np.int64) * 2654435761
                    + int(seed)) % 100
        if split == "train":
            keep = rng_hash < 80
        elif split == "valid":
            keep = (rng_hash >= 80) & (rng_hash < 90)
        elif split == "id":
            keep = rng_hash >= 90
        else:
            raise KeyError(f"idx_v2 smoke split {split!r} unsupported; "
                           f"use train/valid/id (frozen V5.5 OOD splits await contract)")
        idx = np.flatnonzero(keep).tolist()
        if max_curves is not None:
            idx = idx[: int(max_curves)]
        return idx


@dataclass
class _TrajectoryRef:
    curve_idx: int
    intent_idx: int
    curve_id: str
    rows: np.ndarray


class IdxV2Dataset:
    """V5.5-shaped dataset view over ``v53mp_idx_v2`` (smoke / adapter stage)."""

    def __init__(
        self,
        reader: IdxV2Reader,
        *,
        mode: str = "G0",
        seed: int = 17,
        max_curves: dict[str, int] | None = None,
        history_len: int = 16,
        near_ts_window: float = 0.10,
    ):
        self.reader = reader
        self.seed = int(seed)
        self.max_curves = max_curves or {}
        self.near_ts_window = float(near_ts_window)
        self.spec = V55DatasetSpec(
            dataset_version=reader.dataset_version,
            input_mode=str(mode).upper(),
            frame_feat_dim=7,      # 2*q_dim + 5 with q_dim=1
            static_feat_dim=8,
            q_dim=1,
            history_len=int(history_len),
            max_trajectory_len=512,
            near_ts_window=self.near_ts_window,
            notes="real idx_v2 adapter (smoke split = curve hash; V5.5 contract pending)",
        )
        self._splits: dict[str, list[int]] = {}
        self._trajs: dict[str, list[_TrajectoryRef]] = {}
        for split in SPLIT_NAMES:
            if split in ("train", "valid", "id"):
                self._build_split(split)
        self._counts = self._compute_counts()

    # ---------------------------------------------------------------- splits
    def _build_split(self, split: str) -> None:
        curves = self.reader.split_curves(split, seed=self.seed,
                                          max_curves=self.max_curves.get(split))
        ii = self.reader["samples/intent_idx"]
        trajs: list[_TrajectoryRef] = []
        for curve_idx in curves:
            lo, hi = self.reader.sample_rows(curve_idx)
            if hi <= lo:
                continue
            intent_ids = np.unique(np.asarray(ii[lo:hi]))
            curve_id = (self.reader.curve_ids[curve_idx]
                        if curve_idx < len(self.reader.curve_ids) else str(curve_idx))
            for intent in intent_ids:
                rows = lo + np.flatnonzero(np.asarray(ii[lo:hi]) == intent)
                trajs.append(_TrajectoryRef(curve_idx, int(intent), curve_id,
                                            rows.astype(np.int64)))
        self._splits[split] = curves
        self._trajs[split] = trajs

    def _compute_counts(self) -> dict[str, int]:
        n_traj = sum(len(v) for v in self._trajs.values())
        n_frames = 0
        for trajs in self._trajs.values():
            for ref in trajs:
                n_frames += int(ref.rows.shape[0])
        return {"n_curves": sum(len(v) for v in self._splits.values()),
                "n_trajectories": n_traj, "n_frames": n_frames,
                "n_decisions": n_frames}

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
        return {s: len(self._splits.get(s, [])) for s in SPLIT_NAMES}

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

    # ------------------------------------------------------------- sequence
    def build_sequence(self, traj_idx: int) -> dict:
        split = next((s for s, trajs in self._trajs.items() if traj_idx < len(trajs)), None)
        if split is None:
            raise IndexError(f"trajectory index {traj_idx} out of range")
        ref = self._trajs[split][traj_idx]
        curve_idx = ref.curve_idx
        nodes = self.reader.curve_node_arrays(curve_idx)
        rows = ref.rows
        node_index = np.asarray(self.reader["samples/node_index"][rows], dtype=np.int64)
        status = np.asarray(self.reader["samples/label_status"][rows], dtype=np.int64)
        q = np.asarray(nodes["q"][node_index], dtype=np.float32)
        s = np.asarray(nodes["s"][node_index], dtype=np.float32)
        energy = (np.asarray(nodes.get("energy", np.zeros_like(q))[node_index],
                             dtype=np.float32) if "energy" in nodes
                  else np.zeros(q.shape, dtype=np.float32))
        ds = np.zeros_like(s)
        if s.shape[0] > 1:
            ds[1:] = np.diff(s)
        # Move: displacement Δq (= tangent·Δs with tangent dq/ds); mask ok pairs
        move_target = np.zeros_like(q)
        move_mask = np.zeros(q.shape, dtype=bool)
        contiguous = np.ones(q.shape, dtype=bool)
        if node_index.shape[0] > 1:
            contiguous[1:] = np.diff(node_index) == 1
            ok = (status[:-1] == STATUS_OK) & (status[1:] == STATUS_OK) & contiguous[1:]
            move_target[:-1] = q[1:] - q[:-1]
            move_mask[:-1] = ok
        # Process: canonical coordinate q (TS at 0) -> pre/near/post
        valid = status == STATUS_OK
        process_label = np.full(q.shape, -1, dtype=np.int64)
        process_label[valid & (q < -self.near_ts_window)] = 0
        process_label[valid & (np.abs(q) <= self.near_ts_window)] = 1
        process_label[valid & (q > self.near_ts_window)] = 2
        process_mask = process_label >= 0
        process_progress = np.where(valid, q, 0.0).astype(np.float32)
        progress_mask = process_mask.copy()
        # Trust: |d_key_ts_ep| large-error proxy, masked for degenerate curves
        d_key = np.asarray(self.reader["labels_ep/d_key_ts_ep"][rows], dtype=np.float32)
        degenerate = np.asarray(self.reader["labels_ep/degenerate_curve"][rows], dtype=bool)
        trust_mask = valid & np.isfinite(d_key) & (~degenerate)
        trust_label = np.where(trust_mask, np.clip(np.abs(d_key), 0.0, 1.0), 0.0).astype(np.float32)
        feats = build_frame_features(q, ds, energy, np.zeros_like(q),
                                     input_mode=self.spec.input_mode)
        return {
            "traj_idx": int(traj_idx),
            "trajectory_id": f"{ref.curve_id}:{ref.intent_idx}",
            "reaction_id": ref.curve_id,
            "reaction_row": int(curve_idx),
            "direction": 1,
            "frame_feats": feats,
            "static_feats": self._static_feats(curve_idx),
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
            "progress_mask": progress_mask,
            "trust_label": trust_label,
            "trust_mask": trust_mask,
        }

    def iter_batches(self, split: str, *, batch_size: int = 4, shuffle: bool = False,
                     seed: int = 17, fraction: float = 1.0,
                     max_batches: int | None = None, drop_last: bool = False):
        idx = self.trajectory_indices(split, fraction=fraction, seed=seed)
        if shuffle:
            rng = np.random.default_rng(seed)
            idx = [idx[i] for i in rng.permutation(len(idx))]
        for start in range(0, len(idx), batch_size):
            chunk = idx[start:start + batch_size]
            if drop_last and len(chunk) < batch_size:
                break
            samples = [self.build_sequence(i) for i in chunk]
            batch = collate_samples(samples)
            metas = [{k: s[k] for k in ("traj_idx", "trajectory_id", "reaction_id",
                                        "reaction_row", "direction", "length")}
                     for s in samples]
            yield batch, metas
            if max_batches is not None and (start // batch_size + 1) >= max_batches:
                break

    # ------------------------------------------------------- not implemented
    def median_ds(self, traj_idx: int) -> float:
        seq = self.build_sequence(traj_idx)
        ds = np.asarray(seq["delta_s"][1:], dtype=np.float32)
        return float(np.median(ds)) if ds.size else 1.0

    def reference_q(self, traj_idx: int) -> np.ndarray:
        return np.asarray(self.build_sequence(traj_idx)["q"], dtype=np.float32)

    def geom_for(self, reaction_row: int):
        raise NotImplementedError(
            "idx_v2 has no per-reaction kinematic basis; R1 rollout on real data "
            "needs a XYZ geometry adapter (staging_chunks). Not implemented in "
            "V5.6 skeleton — reported as a limitation, not fabricated.")
