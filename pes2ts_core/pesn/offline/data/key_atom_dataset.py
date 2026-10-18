"""Offline precompute + lazy dataset for V5 key-atom features
(ADR-0026 §2.3/§2.6, sprint plan §2/§5.2).

Storage layout — ONE npz per ``(RXN, branch, d0)`` under
``outputs/pesn/v5/features/d0_<v>/<RXN>.branch{k}.npz``:

* flattened node/edge/key/pair arrays concatenated across the shard's
  samples, with int64 pointer tables ``sample_ptr / edge_ptr / key_ptr /
  pair_ptr``;
* per-sample scalars: ``start_row``, ``q_weight`` (= ``q_frame`` × conflict
  weight), ``eligible``, ``near_ts``, ``terminal``, ``capped``, ``d0``,
  ``method_id_hash``, ``solvent_id_hash``, ``charge``, ``multiplicity``,
  ``rho_known``, ``solvent_known``, ``key_progress`` (sampler bookkeeping —
  never a model input);
* per-key-atom supervision ``y_teacher [Σm, 3] float32`` (aligned with
  ``key_ptr``).

Features are stored ANCHOR-ON only; anchor-off is applied at LOAD time via
the fixed slot mask (:func:`pes2ts_core.pesn.features.key_atom_graph.collate`
→ ``KeyAtomBatch.anchor_off_``), so the six arms share one preprocessing
geometry (plan §5.2, ADR-0026 §2.6).

Lane A integration: the task loader and the weighted Kabsch alignment live
in ``pes2ts_core.pesn.offline.data.key_atom_teacher`` / Lane A's task
module; they are imported LAZILY inside :func:`precompute_features` with
injectable seams (``task_loader`` / ``align_fn``) so tests and the campaign
dry-run can bind them independently.  No EHG / force arrays are ever read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from pes2ts_core.pesn.contracts.key_atom import KeyAtomBatch
from pes2ts_core.pesn.features.key_atom_graph import collate
from pes2ts_core.pesn.features.key_atom_spec import (
    METHOD_MAX_KNOWN,
    SOLVENT_MAX_KNOWN,
    build_vocab,
)

FEATURE_SHARD_SCHEMA: Final[str] = "pesn_v5_feature_shard_v1"
D0_VALUES: Final[tuple[float, ...]] = (0.05, 0.08, 0.12)
DEFAULT_FEATURE_ROOT: Final[str] = "outputs/pesn/v5/features"


class KeyAtomDatasetError(RuntimeError):
    """Typed refusal of the offline feature precompute/dataset."""


def _require(cond: bool, code: str, msg: str) -> None:
    if not cond:
        raise KeyAtomDatasetError(f"{code}: {msg}")


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Lane A lazy bindings
# ---------------------------------------------------------------------------
def _lane_a_align(pos0: np.ndarray, pos_t: np.ndarray,
                  ref_rows: np.ndarray, ref_weights: np.ndarray) -> np.ndarray:
    """Weighted Kabsch of the branch X0 frame into the X_t frame.

    Uses Lane A's ``pesn_v5_teacher`` helpers (ADR-0026 §2.2): the rigid
    transform is fit on the reference block S (``ref_rows``/``ref_weights``
    from the native meta json) with the row-vector proper-rotation
    convention, then applied to the FULL X0 frame.  Only X0 and X_t enter —
    never a future frame.
    """
    try:
        from pes2ts_core.pesn.offline.data.key_atom_teacher import (
            apply_rigid_transform,
            weighted_kabsch,
        )
    except Exception as exc:  # pragma: no cover - campaign-bound seam
        raise KeyAtomDatasetError(
            "LANE_A_UNAVAILABLE: pes2ts_core.pesn.offline.data.key_atom_teacher."
            "weighted_kabsch not importable; pass align_fn= explicitly"
        ) from exc
    _aligned_ref, transform = weighted_kabsch(
        np.asarray(pos0, dtype=np.float64)[ref_rows],
        np.asarray(pos_t, dtype=np.float64)[ref_rows],
        np.asarray(ref_weights, dtype=np.float64))
    return apply_rigid_transform(np.asarray(pos0, dtype=np.float64), transform)


def _lane_a_load_task(task_path: str | Path) -> Any:
    """Load one task JSON via the seeded Lane A loader (lazy seam)."""
    try:
        from pes2ts_core.pesn.contracts.key_atom import load_key_atom_task
        return load_key_atom_task(task_path)
    except Exception as exc:  # pragma: no cover - campaign-bound seam
        raise KeyAtomDatasetError(
            "LANE_A_UNAVAILABLE: contracts.key_atom.load_key_atom_task failed; "
            "pass task_loader= explicitly"
        ) from exc


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrecomputeResult:
    """Summary of one precompute run (paths + counts, no feature payload)."""

    shard_paths: tuple[str, ...]
    n_samples: int
    n_eligible: int
    feature_spec_path: str | None = None


@dataclass(frozen=True)
class _ShardInputs:
    """Everything one (reaction, branch) precompute iteration consumes."""

    task: Any
    pos: np.ndarray              # (S, N, 3) float64 native frames
    key_progress: np.ndarray     # (S,) float64 arc-length progress per frame
    q_frame: np.ndarray          # (S,) float64 per-frame sampling weight
    key_rows: np.ndarray         # (m,) int64
    start_row: np.ndarray        # (Sm,) int64 teacher-selected start frames
    u_raw: np.ndarray            # (Sm, m, 3) float32 teacher displacements
    eligible: np.ndarray         # (Sm,) bool
    reason_id: np.ndarray        # (Sm,) uint8
    near_ts: np.ndarray          # (Sm,) float32
    terminal: np.ndarray         # (Sm,) bool
    capped: np.ndarray           # (Sm,) bool
    d_eff: np.ndarray            # (Sm,) float64
    conflict_weight: float
    ref_rows: np.ndarray | None = None
    ref_weights: np.ndarray | None = None


def _resolve_task_path(task_root: Path, reaction_id: str) -> Path | None:
    for candidate in (task_root / f"{reaction_id}.task.json",
                      task_root / f"{reaction_id}.json"):
        if candidate.is_file():
            return candidate
    return None


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    _require(path.is_file(), "SHARD_MISSING", f"required npz not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def precompute_features(
    config: Mapping[str, Any],
    fold_groups: Mapping[str, Sequence[str]] | None = None,
    *,
    task_loader: Callable[[str | Path], Any] | None = None,
    align_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                       np.ndarray] | None = None,
    conflict_weights: Mapping[str, float] | None = None,
    write_feature_spec: bool = True,
) -> PrecomputeResult:
    """Build and write the per-(RXN, branch, d0) feature shards.

    ``config`` keys (paths relative to ``config["root"]``, default cwd):
    ``native_root`` (npz with pos/key_progress/q_frame/key_rows + a sidecar
    ``<RXN>.branch{k}.meta.json`` carrying the branch reference block),
    ``teacher_root`` (``d0_<v>/<RXN>.branch{k}.npz``), ``task_root``,
    ``feature_root``, optional ``conflict_weights`` mapping
    reaction-or-group id → weight.

    ``fold_groups`` restricts the run to the given groups' reaction ids.
    ``task_loader`` / ``align_fn`` default to the Lane A lazy bindings.
    ``conflict_weights`` overrides the config-embedded mapping (tests).
    """
    from pes2ts_core.pesn.features.key_atom_spec import stable_id_hash

    root = Path(config.get("root", "."))
    native_root = root / config["native_root"]
    teacher_root = root / config["teacher_root"]
    task_root = root / config["task_root"]
    feature_root = root / config.get("feature_root", DEFAULT_FEATURE_ROOT)
    d0_values = tuple(float(d) for d in config.get("d0_values", D0_VALUES))
    load_task = task_loader or _lane_a_load_task
    align = align_fn or _lane_a_align
    conflict_map = dict(conflict_weights or config.get("conflict_weights", {}))

    reactions: set[tuple[str, int]] = set()
    for path in teacher_root.glob("d0_*/*.branch*.npz"):
        rid, rest = path.name.split(".branch", 1)
        reactions.add((rid, int(rest.split(".", 1)[0])))
    if fold_groups is not None:
        allowed = {r for group in fold_groups.values() for r in group}
        reactions = {(r, b) for (r, b) in reactions if r in allowed}
    _require(bool(reactions), "NO_REACTIONS",
             f"no teacher shards found under {teacher_root}")

    shard_paths: list[str] = []
    n_samples_total = 0
    n_eligible_total = 0
    spec_hashes: dict[str, str] = {}

    tasks: dict[str, Any] = {}
    for reaction_id, branch in sorted(reactions):
        if reaction_id not in tasks:
            task_path = _resolve_task_path(task_root, reaction_id)
            _require(task_path is not None, "TASK_MISSING",
                     f"no task JSON for {reaction_id} under {task_root}")
            spec_hashes[f"task:{reaction_id}"] = _sha256_file(task_path)
            tasks[reaction_id] = load_task(task_path)
        task = tasks[reaction_id]
        native = _load_npz(native_root / f"{reaction_id}.branch{branch}.npz")
        meta_path = native_root / f"{reaction_id}.branch{branch}.meta.json"
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            spec_hashes[f"native_meta:{reaction_id}.b{branch}"] = \
                _sha256_file(meta_path)
        # Lane A native meta nests the reference block under "reference"
        # (weights as {row: w}); older flat keys stay supported.
        ref_rows: np.ndarray
        ref_weights: np.ndarray
        ref_block = meta.get("reference")
        if isinstance(ref_block, Mapping):
            ref_rows = np.asarray(ref_block.get("rows", ()), dtype=np.int64)
            w_map = ref_block.get("weights", {})
            if isinstance(w_map, Mapping):
                ref_weights = np.array(
                    [float(w_map.get(str(int(r)), 1.0)) for r in ref_rows],
                    dtype=np.float64)
            else:
                ref_weights = np.asarray(w_map, dtype=np.float64)
        else:
            ref_rows = np.asarray(meta.get("reference_rows", ()), dtype=np.int64)
            ref_weights = np.asarray(meta.get("reference_weights", ()),
                                     dtype=np.float64)
        if ref_rows.size and not bool(meta.get("reference", {}).get("gate_ok", True)):
            # gauge_unresolved branches keep X0 unaligned (ADR-0026 §2.2);
            # the builder then sees pos0 = raw X0 and anchor_valid marks it.
            ref_rows = np.asarray((), dtype=np.int64)
            ref_weights = np.asarray((), dtype=np.float64)
        weight = float(conflict_map.get(reaction_id,
                                        conflict_map.get(f"{reaction_id}.b{branch}",
                                                         1.0)))
        for d0 in d0_values:
            d0_tag = f"{d0:g}"
            teacher_path = (teacher_root / f"d0_{d0_tag}"
                            / f"{reaction_id}.branch{branch}.npz")
            teacher = _load_npz(teacher_path)
            spec_hashes[f"teacher:{reaction_id}.b{branch}.d0{d0_tag}"] = \
                _sha256_file(teacher_path)
            inputs = _ShardInputs(
                task=task,
                pos=np.asarray(native["pos"], dtype=np.float64),
                key_progress=np.asarray(native["key_progress"], dtype=np.float64),
                q_frame=np.asarray(native["q_frame"], dtype=np.float64),
                key_rows=np.asarray(native["key_rows"], dtype=np.int64),
                start_row=np.asarray(teacher["start_row"], dtype=np.int64),
                u_raw=np.asarray(teacher["u_raw"], dtype=np.float32),
                eligible=np.asarray(teacher["eligible"], dtype=bool),
                reason_id=np.asarray(teacher["reason_id"], dtype=np.uint8),
                near_ts=np.asarray(teacher["near_ts"], dtype=np.float32),
                terminal=np.asarray(teacher["terminal"], dtype=bool),
                capped=np.asarray(teacher["capped"], dtype=bool),
                d_eff=np.asarray(teacher["d_eff"], dtype=np.float64),
                conflict_weight=weight,
                ref_rows=ref_rows if ref_rows.size else None,
                ref_weights=ref_weights if ref_weights.size else None,
            )
            out = feature_root / f"d0_{d0_tag}" / f"{reaction_id}.branch{branch}.npz"
            n_s, n_e = _write_shard(out, inputs, task, branch, d0,
                                    align, stable_id_hash)
            shard_paths.append(str(out))
            n_samples_total += n_s
            n_eligible_total += n_e

    spec_path: str | None = None
    if write_feature_spec:
        from pes2ts_core.pesn.features.key_atom_spec import write_feature_spec as _ws
        spec_path = str(feature_root / "feature_spec.json")
        _ws(spec_path, spec_hashes)
    return PrecomputeResult(tuple(shard_paths), n_samples_total,
                            n_eligible_total, spec_path)


def _write_shard(
    out_path: Path,
    inputs: _ShardInputs,
    task: Any,
    branch: int,
    d0: float,
    align: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    stable_id_hash: Callable[[str], int],
) -> tuple[int, int]:
    """Build every teacher start row of one (reaction, branch, d0) and write
    the flattened shard npz.  Returns ``(n_samples, n_eligible)``."""
    from pes2ts_core.pesn.features.key_atom_graph import build_sample_features

    _require(inputs.pos.ndim == 3 and inputs.pos.shape[2] == 3,
             "NATIVE_POS_INVALID", "native pos must be (S, N, 3)")
    n_frames = inputs.pos.shape[0]
    n_rows = int(inputs.start_row.shape[0])
    _require(inputs.u_raw.shape[0] == n_rows,
             "TEACHER_SHAPE_INVALID", "u_raw rows must match start_row")
    _require(bool(np.all((inputs.start_row >= 0) & (inputs.start_row < n_frames))),
             "START_ROW_INVALID", "start_row outside native frame range")
    # native key_rows and the task's key rows must agree (ADR-0026 §2.1/§2.3)
    task_key_rows = np.asarray(sorted(set(
        int(a) for a in (getattr(task, "key_rows", None)
                         or getattr(task, "key_atom_rows", ())))),
        dtype=np.int64)
    _require(np.array_equal(inputs.key_rows, task_key_rows),
             "KEY_ROWS_MISMATCH",
             f"native key_rows {inputs.key_rows.tolist()} != task key_rows "
             f"{task_key_rows.tolist()}")
    _require(inputs.u_raw.shape[1] == task_key_rows.size,
             "TEACHER_WIDTH_INVALID",
             "u_raw key-atom width must match the task key set")

    x0 = inputs.pos[0]                                   # causal X0 = row 0
    method_hash = stable_id_hash(str(getattr(task, "method_id", "")))
    solvent_raw = getattr(task, "solvent", None)
    if solvent_raw is None:
        solvent_raw = getattr(task, "solvent_id", None)
    solvent_known = solvent_raw is not None and str(solvent_raw) not in ("", "unknown")
    solvent_hash = stable_id_hash(str(solvent_raw)) if solvent_known else 0
    rho_known = bool(getattr(task, "rho_known", solvent_known))

    method_hashes = np.full(n_rows, method_hash, dtype=np.int64)
    solvent_hashes = np.full(n_rows, solvent_hash, dtype=np.int64)
    charges = np.full(n_rows, float(getattr(task, "charge", 0)), dtype=np.float64)
    mults = np.full(n_rows, float(getattr(task, "multiplicity", 1)), dtype=np.float64)
    rho_known_arr = np.full(n_rows, rho_known, dtype=bool)
    solvent_known_arr = np.full(n_rows, solvent_known, dtype=bool)

    samples: list[dict[str, np.ndarray]] = []
    n_eligible = 0
    for s in range(n_rows):
        row = int(inputs.start_row[s])
        pos_t = inputs.pos[row]
        pos0 = (align(x0, pos_t, inputs.ref_rows, inputs.ref_weights)
                if inputs.ref_rows is not None and inputs.ref_weights is not None
                else x0.copy())
        sample = build_sample_features(
            task, pos_t, pos0, branch, d0, anchor_mode=True,
            method_hash=method_hash, solvent_hash=solvent_hash,
            rho_known=rho_known, solvent_known=solvent_known)
        sample["start_row"] = np.int64(row)
        sample["q_weight"] = np.float64(
            inputs.q_frame[row] * inputs.conflict_weight)
        sample["eligible"] = np.bool_(bool(inputs.eligible[s]))
        sample["y_teacher"] = inputs.u_raw[s].astype(np.float32)
        sample["near_ts"] = np.float32(inputs.near_ts[s])
        sample["terminal"] = np.bool_(bool(inputs.terminal[s]))
        sample["capped"] = np.bool_(bool(inputs.capped[s]))
        sample["d_eff"] = np.float64(inputs.d_eff[s])
        sample["reason_id"] = np.uint8(inputs.reason_id[s])
        sample["key_progress"] = np.float64(inputs.key_progress[row])
        sample["reaction_id"] = str(getattr(task, "reaction_id", ""))
        sample["group_id"] = str(getattr(task, "reaction_group_id",
                                         getattr(task, "group_id", "")))
        samples.append(sample)
        n_eligible += int(sample["eligible"])

    _flatten_and_save(out_path, samples,
                      {"method_id_hash": method_hashes,
                       "solvent_id_hash": solvent_hashes,
                       "charge": charges, "multiplicity": mults,
                       "rho_known": rho_known_arr,
                       "solvent_known": solvent_known_arr})
    return len(samples), n_eligible


_SAMPLE_SCALAR_KEYS: Final[tuple[str, ...]] = (
    "start_row", "q_weight", "eligible", "near_ts", "terminal", "capped",
    "d_eff", "reason_id", "key_progress")


def _flatten_and_save(
    out_path: Path,
    samples: Sequence[Mapping[str, np.ndarray]],
    per_sample: Mapping[str, np.ndarray],
) -> None:
    """Flatten concatenated samples + pointer tables into one npz."""
    n = len(samples)
    sample_ptr = np.zeros(n + 1, dtype=np.int64)
    edge_ptr = np.zeros(n + 1, dtype=np.int64)
    key_ptr = np.zeros(n + 1, dtype=np.int64)
    pair_ptr = np.zeros(n + 1, dtype=np.int64)
    for i, s in enumerate(samples):
        sample_ptr[i + 1] = sample_ptr[i] + int(np.asarray(s["h_node_inv"]).shape[0])
        edge_ptr[i + 1] = edge_ptr[i] + int(np.asarray(s["edge_index"]).shape[1])
        key_ptr[i + 1] = key_ptr[i] + int(np.asarray(s["key_index"]).shape[0])
        pair_ptr[i + 1] = pair_ptr[i] + int(np.asarray(s["pair_index"]).shape[1])

    arrays: dict[str, np.ndarray] = {
        "sample_ptr": sample_ptr, "edge_ptr": edge_ptr,
        "key_ptr": key_ptr, "pair_ptr": pair_ptr,
        "z_atomic": np.concatenate([np.asarray(s["z_atomic"]) for s in samples]),
        "h_node_inv": np.concatenate(
            [np.asarray(s["h_node_inv"], dtype=np.float32) for s in samples], axis=0),
        "pos": np.concatenate([np.asarray(s["pos"], dtype=np.float32)
                               for s in samples], axis=0),
        "pos0": np.concatenate([np.asarray(s["pos0"], dtype=np.float32)
                                for s in samples], axis=0),
        "anchor_vec": np.concatenate([np.asarray(s["anchor_vec"], dtype=np.float32)
                                      for s in samples], axis=0),
        # edge_index stored SAMPLE-LOCAL; collate re-offsets via ptrs
        "edge_index": np.concatenate(
            [np.asarray(s["edge_index"], dtype=np.int64).reshape(2, -1)
             for s in samples], axis=1),
        "edge_attr": np.concatenate(
            [np.asarray(s["edge_attr"], dtype=np.float32) for s in samples], axis=0),
        "r_ij": np.concatenate([np.asarray(s["r_ij"], dtype=np.float32)
                                for s in samples], axis=0),
        "r0_ij": np.concatenate([np.asarray(s["r0_ij"], dtype=np.float32)
                                 for s in samples], axis=0),
        "key_index": np.concatenate(
            [np.asarray(s["key_index"], dtype=np.int64) for s in samples]),
        "pair_index": np.concatenate(
            [np.asarray(s["pair_index"], dtype=np.int64).reshape(2, -1)
             for s in samples], axis=1),
        "pair_edit": np.concatenate(
            [np.asarray(s["pair_edit"], dtype=np.float32) for s in samples]),
        "degK": np.concatenate([np.asarray(s["degK"], dtype=np.float32)
                                for s in samples]),
    }
    for key in _SAMPLE_SCALAR_KEYS:
        if all(key in s for s in samples):
            first = samples[0][key]
            arrays[key] = np.array([s[key] for s in samples], dtype=first.dtype)
    # per-sample identity/bookkeeping arrays consumed by build_vocabs/pools
    arrays["sample_branch"] = np.array(
        [int(s["branch"]) for s in samples], dtype=np.int64)
    arrays["sample_d0"] = np.array(
        [float(s["d0"]) for s in samples], dtype=np.float64)
    arrays["sample_method_hash"] = np.array(
        [int(s["method_hash"]) for s in samples], dtype=np.int64)
    arrays["sample_solvent_hash"] = np.array(
        [int(s["solvent_hash"]) for s in samples], dtype=np.int64)
    max_rid = max((len(str(s.get("reaction_id", ""))) for s in samples), default=1)
    arrays["sample_reaction_id"] = np.array(
        [str(s.get("reaction_id", "")) for s in samples],
        dtype=f"U{max(1, max_rid)}")
    max_gid = max((len(str(s.get("group_id", ""))) for s in samples), default=1)
    arrays["sample_group_id"] = np.array(
        [str(s.get("group_id", "")) for s in samples],
        dtype=f"U{max(1, max_gid)}")
    for key, arr in per_sample.items():
        arrays[key] = np.asarray(arr)
    if all("y_teacher" in s for s in samples):
        arrays["y_teacher"] = np.concatenate(
            [np.asarray(s["y_teacher"], dtype=np.float32) for s in samples], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        np.savez(fh, **arrays)


# ---------------------------------------------------------------------------
# Lazy dataset
# ---------------------------------------------------------------------------
class V5FeatureDataset(Dataset):
    """Lazy reader over one or more ``(RXN, branch, d0)`` feature shards.

    npz members are opened with ``mmap_mode="r"`` where the format allows;
    per-sample slices materialize on ``__getitem__`` as plain numpy dicts
    consumed by :func:`key_atom_graph.collate`.
    """

    def __init__(self, shard_paths: Sequence[str | Path]) -> None:
        _require(len(shard_paths) > 0, "EMPTY_DATASET",
                 "V5FeatureDataset needs at least one shard path")
        self.shard_paths = tuple(str(p) for p in shard_paths)
        self._doc: dict[str, Any] | None = None
        self._handles: list[Any] = []
        self._n_total = 0
        self._shard_of_sample: np.ndarray = np.zeros(0, dtype=np.int64)
        self._local_of_sample: np.ndarray = np.zeros(0, dtype=np.int64)
        self._doc = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        shard_index = np.zeros(len(self.shard_paths) + 1, dtype=np.int64)
        for i, path in enumerate(self.shard_paths):
            with np.load(path, allow_pickle=False) as data:
                shard_index[i + 1] = shard_index[i] + int(data["sample_ptr"].size - 1)
        self._shard_index = shard_index
        self._n_total = int(shard_index[-1])
        return {"n_samples": self._n_total}

    def __len__(self) -> int:
        return self._n_total

    def _resolve(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self._n_total:
            raise IndexError(index)
        shard = int(np.searchsorted(self._shard_index, index, side="right") - 1)
        return shard, int(index - self._shard_index[shard])

    def _shard_arrays(self, shard: int) -> dict[str, np.ndarray]:
        if not hasattr(self, "_cache"):
            self._cache: dict[int, dict[str, np.ndarray]] = {}
        if shard not in self._cache:
            # small LRU-ish cache: keep at most 4 open shards
            if len(self._cache) >= 4:
                self._cache.pop(next(iter(self._cache)))
            with np.load(self.shard_paths[shard], allow_pickle=False,
                         mmap_mode="r") as data:
                self._cache[shard] = {k: data[k] for k in data.files}
        return self._cache[shard]

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard, local = self._resolve(index)
        a = self._shard_arrays(shard)
        ns, ne, nk, npair = (int(a["sample_ptr"][local + 1] - a["sample_ptr"][local]),
                             int(a["edge_ptr"][local + 1] - a["edge_ptr"][local]),
                             int(a["key_ptr"][local + 1] - a["key_ptr"][local]),
                             int(a["pair_ptr"][local + 1] - a["pair_ptr"][local]))
        ns0, ne0, nk0, npair0 = (int(a["sample_ptr"][local]),
                                 int(a["edge_ptr"][local]),
                                 int(a["key_ptr"][local]),
                                 int(a["pair_ptr"][local]))
        sample: dict[str, Any] = {
            "z_atomic": np.asarray(a["z_atomic"][ns0:ns0 + ns], dtype=np.int64),
            "h_node_inv": np.asarray(a["h_node_inv"][ns0:ns0 + ns], dtype=np.float32),
            "pos": np.asarray(a["pos"][ns0:ns0 + ns], dtype=np.float32),
            "pos0": np.asarray(a["pos0"][ns0:ns0 + ns], dtype=np.float32),
            "anchor_vec": np.asarray(a["anchor_vec"][ns0:ns0 + ns], dtype=np.float32),
            "edge_index": np.asarray(a["edge_index"][:, ne0:ne0 + ne], dtype=np.int64),
            "edge_attr": np.asarray(a["edge_attr"][ne0:ne0 + ne], dtype=np.float32),
            "r_ij": np.asarray(a["r_ij"][ne0:ne0 + ne], dtype=np.float32),
            "r0_ij": np.asarray(a["r0_ij"][ne0:ne0 + ne], dtype=np.float32),
            "key_index": np.asarray(a["key_index"][nk0:nk0 + nk], dtype=np.int64),
            "pair_index": np.asarray(a["pair_index"][:, npair0:npair0 + npair],
                                     dtype=np.int64),
            "pair_edit": np.asarray(a["pair_edit"][npair0:npair0 + npair],
                                    dtype=np.float32),
            "degK": np.asarray(a["degK"][nk0:nk0 + nk], dtype=np.float32),
        }
        for key in _SAMPLE_SCALAR_KEYS:
            if key in a:
                sample[key] = a[key][local].item() if np.ndim(a[key][local]) == 0 \
                    else a[key][local]
        for key in ("branch", "d0", "method_hash", "solvent_hash"):
            src = f"sample_{key}"
            if src in a:
                sample[key] = a[src][local].item()
        for key, dtype in (("method_id_hash", np.int64), ("solvent_id_hash", np.int64),
                           ("charge", np.float64), ("multiplicity", np.float64),
                           ("rho_known", np.bool_), ("solvent_known", np.bool_)):
            if key in a:
                sample[key] = np.asarray(a[key][local], dtype=dtype)
        if "y_teacher" in a:
            sample["y_teacher"] = np.asarray(a["y_teacher"][nk0:nk0 + nk],
                                             dtype=np.float32)
        if "sample_reaction_id" in a:
            sample["reaction_id"] = str(a["sample_reaction_id"][local])
        else:
            sample["reaction_id"] = str(self.shard_paths[shard].split("/")[-1]
                                        .split(".branch")[0])
        if "sample_group_id" in a:
            sample["group_id"] = str(a["sample_group_id"][local])
        else:
            sample["group_id"] = sample["reaction_id"]
        return sample

    def get_collate_fn(
        self,
        anchor_mode: bool,
        method_vocab: Mapping[int, int],
        solvent_vocab: Mapping[int, int],
        device: torch.device | str = "cpu",
    ) -> Callable[[Sequence[dict[str, Any]]], KeyAtomBatch]:
        """DataLoader-style collate closure over the frozen vocab/anchor mode."""

        def _collate(batch: Sequence[dict[str, Any]]) -> KeyAtomBatch:
            return collate(batch, device, anchor_mode, method_vocab, solvent_vocab)

        return _collate


# ---------------------------------------------------------------------------
# Vocabularies + start-frame pools
# ---------------------------------------------------------------------------
def build_vocabs(
    feature_root: str | Path,
    train_reaction_ids: Sequence[str],
) -> tuple[dict[int, int], dict[int, int]]:
    """Method/solvent vocabs from TRAIN groups only (plan §2; UNK = 0)."""
    root = Path(feature_root)
    train_set = set(train_reaction_ids)
    method_ids: set[int] = set()
    solvent_ids: set[int] = set()
    for path in sorted(root.glob("d0_*/*.branch*.npz")):
        reaction_id = path.name.split(".branch")[0]
        if reaction_id not in train_set:
            continue
        with np.load(path, allow_pickle=False) as data:
            if "method_id_hash" in data.files:
                method_ids.update(int(v) for v in data["method_id_hash"])
            if "solvent_id_hash" in data.files:
                solvent_ids.update(int(v) for v in data["solvent_id_hash"]
                                   if int(v) != 0)
    return (build_vocab(method_ids, max_known=METHOD_MAX_KNOWN),
            build_vocab(solvent_ids, max_known=SOLVENT_MAX_KNOWN))


#: One pool entry: ``(shard_idx, sample_idx, q_weight)``.
PoolEntry = tuple[int, int, float]


def _pool_shards(feature_root: Path, d0_tag: str) -> list[Path]:
    return sorted((feature_root / f"d0_{d0_tag}").glob("*.branch*.npz"))


def build_pools(
    feature_root: str | Path,
    groups: Mapping[str, Sequence[str]],
    max_per_branch_cycle: int = 16,
    d0_tag: str = "0.08",
) -> dict[str, dict[int, list[list[PoolEntry]]]]:
    """Sparse start-frame pools (plan §5.2): per ``(group, branch)`` a list
    of CYCLES; each cycle holds up to ``max_per_branch_cycle`` entries
    ``(shard_idx, sample_idx, q_weight)``.

    Selection: eligible rows of the group's shards sorted by
    ``key_progress``, split into 4 quartiles, 4 rows per quartile per cycle
    (fewer when a quartile is small); the starting offset within each
    quartile rotates deterministically by ``cycle_index`` seeded from the
    (group, branch) key.  The final cycle always contains the last eligible
    row of the branch (the near-TS short-tail step, §5.2).  ``shard_idx``
    indexes the sorted per-d0 shard list returned by :func:`pool_shard_paths`.
    """
    root = Path(feature_root)
    shards = _pool_shards(root, d0_tag)
    _require(bool(shards), "NO_SHARDS", f"no feature shards under {root}/d0_{d0_tag}")
    per_shard: list[dict[str, np.ndarray]] = []
    for path in shards:
        with np.load(path, allow_pickle=False) as data:
            per_shard.append({k: np.asarray(data[k]) for k in data.files
                              if k in ("sample_reaction_id", "sample_branch",
                                       "key_progress", "q_weight", "eligible")})

    pools: dict[str, dict[int, list[list[PoolEntry]]]] = {}
    for group_id, reaction_ids in groups.items():
        rid_set = set(reaction_ids)
        pools[group_id] = {}
        for branch in (0, 1):
            rows: list[tuple[float, int, int, float]] = []  # (progress, shard, local, q)
            for si, doc in enumerate(per_shard):
                if "sample_reaction_id" not in doc:
                    continue
                for li in range(len(doc["key_progress"])):
                    rid = str(doc["sample_reaction_id"][li])
                    if rid not in rid_set:
                        continue
                    if int(doc["sample_branch"][li]) != branch:
                        continue
                    if not bool(doc["eligible"][li]):
                        continue
                    rows.append((float(doc["key_progress"][li]), si, li,
                                 float(doc["q_weight"][li])))
            pools[group_id][branch] = _cycles_for_branch(
                rows, group_id, branch, max_per_branch_cycle)
    return pools


def pool_shard_paths(feature_root: str | Path, d0_tag: str = "0.08") -> list[str]:
    """The shard list ``build_pools`` indexes with its ``shard_idx``."""
    return [str(p) for p in _pool_shards(Path(feature_root), d0_tag)]


def _cycles_for_branch(
    rows: Sequence[tuple[float, int, int, float]],
    group_id: str,
    branch: int,
    max_per_branch_cycle: int,
) -> list[list[PoolEntry]]:
    if not rows:
        return []
    per_quartile = max(1, max_per_branch_cycle // 4)
    ordered = sorted(rows, key=lambda t: (t[0], t[1], t[2]))
    n = len(ordered)
    quartile_bounds = [(n * q) // 4 for q in range(1, 4)]
    quartiles: list[list[tuple[float, int, int, float]]] = []
    start = 0
    for bound in [*quartile_bounds, n]:
        quartiles.append(list(ordered[start:bound]))
        start = bound

    seed = int(hashlib.sha256(f"{group_id}|{branch}".encode()).hexdigest()[:8], 16)
    n_cycles = max(1, max(len(q) for q in quartiles))
    cycles: list[list[PoolEntry]] = []
    for c in range(n_cycles):
        cycle: list[PoolEntry] = []
        for qi, quartile in enumerate(quartiles):
            if not quartile:
                continue
            offset = (seed + c * per_quartile + qi) % len(quartile)
            for t in range(per_quartile):
                entry = quartile[(offset + t) % len(quartile)]
                pool_entry = (entry[1], entry[2], entry[3])
                if pool_entry not in cycle:
                    cycle.append(pool_entry)
        cycles.append(cycle)

    # near-TS retention: the final cycle always holds the branch's last row
    last = ordered[-1]
    final = (last[1], last[2], last[3])
    if final not in cycles[-1]:
        cycles[-1].append(final)
    return cycles


def iter_sample_weights(pools: Mapping[str, Mapping[int, Sequence[Sequence[PoolEntry]]]]
                        ) -> Iterator[float]:
    """Yield the q weight of every pooled entry (sampler exposure)."""
    for branches in pools.values():
        for cycles in branches.values():
            for cycle in cycles:
                for _, _, q in cycle:
                    yield q


# ---------------------------------------------------------------------------
# Module main
# ---------------------------------------------------------------------------
def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pes2ts_core.pesn.offline.data.key_atom_dataset",
        description="Precompute V5 key-atom feature shards (ADR-0026).")
    parser.add_argument("--config", required=True,
                        help="JSON file with the precompute config mapping")
    parser.add_argument("--fold-groups", default=None,
                        help="Optional JSON {group_id: [reaction ids]} filter")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    fold_groups = (json.loads(Path(args.fold_groups).read_text(encoding="utf-8"))
                   if args.fold_groups else None)
    result = precompute_features(config, fold_groups)
    print(json.dumps({
        "n_shards": len(result.shard_paths),
        "n_samples": result.n_samples,
        "n_eligible": result.n_eligible,
        "feature_spec": result.feature_spec_path,
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
