"""V5.3-MP loader v2 (plan §4 / S4 infra lane I1): columnar + memmap dataset.

``ColumnarKeyAtomDatasetV3`` exposes the SAME constructor surface and public
API as :class:`ShardedKeyAtomDatasetV3` (v1), but serves the per-sample light
index and the label tables from flat ``.npy`` column files opened with
``np.load(..., mmap_mode="r")`` instead of a 2 GB pickle + a 4 GB eagerly
expanded label npz.  Sample order is BYTE-IDENTICAL to v1's ``samples``
tuple (the idx is built from the v1 light pickle in order; certified by
``scripts/agent_ops/data_gate/r3_v53mp_loader_v2_certificate.py``).

Files consumed (built by ``r3_v53mp_build_idx_v2.py``; ``allow_pickle=False``
everywhere, write AND read):

```text
<tables_dir>/idx_v2/
  curves.json              # schema/binding block + curve/intent string tables
  actions.npy              # float32 [sum(K), 3] teacher displacements
  samples/<field>.npy      # one .npy per sample column (~45 B/sample core)
  intents/<field>.npy      # columnar IntentRecordV3 (first-appearance order)
  labels_v52/<field>.npy   # q/q_mask/stop/d_key_ts, SAME row order as samples
  labels_ep/<field>.npy    # endpoint_label_spec_v1 columns, SAME row order
  label_curves/<field>.npy # per-curve node_order/arc_s/q_canonical (lazy)
  sid_lookup_keys.npy      # int64 packed (curve_idx, node, intent_idx) hash
  sid_lookup_vals.npy      # int32 sample row per hash slot (−1 = empty)
  curve_node_energies.npy  # float64 [sum(n_nodes)] NaN-padded per-node energies
```

Design notes / documented deviations from plan §4.1:

* the plan sketches ``samples.npz`` "one array per field … mmap_mode='r'" —
  a zip ``.npz`` CANNOT be memory-mapped, so the idx is emitted as one flat
  ``.npy`` per field under ``samples/`` (the plan's own per-field intent);
  the real 20k build is 1.1 GB total, **43 B/sample core index** (44 B plan
  target) + intent/label/energy columns;
* ``intent_idx`` is **int32**, not the plan's int16: the frozen 20k cohort
  already has 52,076 distinct intents (> 32,767);
* weights are stored float32 per plan; the record exposes them widened to
  float64, so ``float32(rec)`` is bit-identical to v1 (the trainer consumes
  them only through an ``np.float32`` cast — certified);
* ``sample_id`` is not stored; it is reconstructed as
  ``f"{curve_id}:{node_index}:{intent_id}:{d0}"`` and the BUILDER refuses
  (fail-closed) unless every v1 row satisfies the format.

Fail-closed binding (mirrors v1's light-cache discipline): init verifies the
idx schema, every column file's sha256, the binding block against the LIVE
``data_manifest.json`` / split manifest shas, and (size, mtime_ns) of the
source light pickle + label cache.  ``verify_sources=True`` additionally
re-hashes the multi-GB sources.  Any mismatch raises
:class:`KeyAtomDatasetV3Error` — never a silent stale serve.

Geometry stays on the v1 shard-CSV LRU (inherited; lane I2 replaces it —
:meth:`ColumnarKeyAtomDatasetV3.frame_xyz` is the override point).
Torch-free, like v1.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    D0_PRIMARY_ANGSTROM,
    IntentRecordV3,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
    KeyAtomDatasetV3Error,
    SampleRecordV3Lite,
    ShardedKeyAtomDatasetV3,
    _load_folds_at,
    _require,
    _sha256,
)

#: idx directory inside the tables dir.
IDX2_DIRNAME: Final[str] = "idx_v2"
#: idx schema id (curves.json ``schema``; also the binding schema).
IDX2_SCHEMA: Final[str] = "v53mp_idx_v2"

#: Geometry memmap pack (S4 plan §5.2 / lane I2): one flat float64 file of
#: every curve's frames + an offset table, replacing the shard-LRU path.
GEOM_DIRNAME: Final[str] = "geom"
GEOM_COORDS_FILENAME: Final[str] = "coords.f64.npy"
GEOM_OFFSETS_FILENAME: Final[str] = "curve_offsets.json"
GEOM_SCHEMA: Final[str] = "v53mp_geom_pack_v1"

#: Per-sample core columns (plan §4.1; intent_idx widened to int32 — see
#: module docstring).  ``{name: (dtype str, required)}``.
SAMPLE_FIELDS: Final[dict[str, str]] = {
    "curve_idx": "int32",
    "node_index": "int32",
    "intent_idx": "int32",
    "fold": "int8",
    "label_status": "int8",
    "d0": "float64",
    "label_weight": "float32",
    "curve_weight": "float32",
    "sample_weight": "float32",
    "target_node": "int32",
    "action_off": "int32",
    "action_len": "int8",
}
#: Label-cache v5.2 sample columns carried columnar (identity fields are
#: served from the sample index itself, exactly like the v1 dataclasses do).
LABELS_V52_FIELDS: Final[dict[str, str]] = {
    "q": "float64",
    "q_mask": "bool",
    "stop": "int8",
    "d_key_ts": "float64",
}
#: ``endpoint_label_spec_v1`` sample columns carried columnar.
LABELS_EP_FIELDS: Final[dict[str, str]] = {
    "ep_label": "int8",
    "d_end": "float64",
    "d_key_ts_ep": "float64",
    "bond_edit_complete": "bool",
    "geometry_legal": "bool",
    "ts_row": "bool",
    "start_row": "bool",
    "terminal_row": "bool",
    "degenerate_curve": "bool",
    "energy_ev_within_window": "int8",
}
#: Per-curve v5.2 label backbone columns (lazy CurveV52Labels views).
LABEL_CURVE_FIELDS: Final[dict[str, str]] = {
    "node_order": "int32",
    "arc_s": "float64",
    "q_canonical": "float64",
}
#: Intent columnar schema (IntentRecordV3 enumeration; see builder).
INTENT_FIELDS: Final[dict[str, str]] = {
    "intent_ids": "str",
    "curve_idx": "int32",
    "direction": "int8",
    "reverse_of": "int32",
    "pair_ids": "str",
    "g_src_off": "int32",
    "g_tgt_off": "int32",
    "bond_i": "int32",
    "bond_j": "int32",
    "bond_order": "float64",
    "cv_off": "int32",
    "cv_deltas": "float64",
    "cv_coords": "int32",
    "cv_coords_off": "int32",
    "g_atoms": "int32",
    "g_atoms_off": "int32",
    "version_ids": "str",
    "underdetermined": "bool",
    "cv_kind_ids": "str",
}

#: Hash packing base: (curve_idx << 42) | (node_index << 21) | intent_idx —
#: injective for curve < 2^21, node < 2^21, intent < 2^21 (builder asserts).
_SID_SHIFT: Final[int] = 21
_SID_MASK: Final[int] = (1 << _SID_SHIFT) - 1
_U64: Final[int] = (1 << 64) - 1


def _pack_sid_key(curve_idx: int, node_index: int, intent_idx: int) -> int:
    return ((int(curve_idx) << (2 * _SID_SHIFT))
            | (int(node_index) << _SID_SHIFT) | int(intent_idx))


def _slot_of(key: int, mask: int) -> int:
    """Multiplicative-hash slot (xor-folded, then odd multiply, pow-2 mask)."""
    z = key ^ (key >> _SID_SHIFT) ^ (key >> (2 * _SID_SHIFT))
    return ((z * 0x9E3779B97F4A7C15) & _U64) & int(mask)


# ---------------------------------------------------------------------------
# lazy sample sequence + sid lookup
# ---------------------------------------------------------------------------
class _LazySampleSequence(Sequence[SampleRecordV3Lite]):
    """Single-pass-safe lazy view over the columnar sample index.

    Constructs :class:`SampleRecordV3Lite` ON DEMAND from memmap columns;
    nothing is cached or accumulated, so ``len()`` / repeated iteration /
    random access never materialise the full record set (tests pin this
    with a construction counter).
    """

    __slots__ = ("_ds",)

    def __init__(self, dataset: "ColumnarKeyAtomDatasetV3") -> None:
        self._ds = dataset

    def __len__(self) -> int:
        return self._ds._n_rows

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self._ds._record_at(row)
                    for row in range(*index.indices(len(self)))]
        row = int(index)
        if row < 0:
            row += len(self)
        if not 0 <= row < len(self):
            raise IndexError(index)
        return self._ds._record_at(row)

    def __iter__(self) -> Iterator[SampleRecordV3Lite]:
        for row in range(len(self)):
            yield self._ds._record_at(row)


class _SidLookup:
    """sample_id → row via open-addressing probe over packed int64 keys.

    The builder emits a 1.3×-load linear-probe table keyed by the injective
    pack ``(curve_idx, node_index, intent_idx)``; parsing one sid costs two
    dict lookups (curve/intent string tables) + O(1) probes — no 6M-entry
    Python dict is ever built.
    """

    __slots__ = ("_keys", "_vals", "_mask", "_curves", "_intents")

    def __init__(self, idx_dir: Path, curve_row: Mapping[str, int],
                 intent_row: Mapping[str, int]) -> None:
        self._keys = np.load(idx_dir / "sid_lookup_keys.npy",
                             mmap_mode="r", allow_pickle=False)
        self._vals = np.load(idx_dir / "sid_lookup_vals.npy",
                             mmap_mode="r", allow_pickle=False)
        _require(self._keys.shape == self._vals.shape and self._keys.ndim == 1,
                 "IDX_SHAPE", "sid_lookup key/val arrays disagree")
        _require((int(self._keys.shape[0])
                  & int(self._keys.shape[0]) - 1) == 0,
                 "IDX_SHAPE", "sid_lookup table length must be a power of 2")
        self._mask = int(self._keys.shape[0]) - 1
        self._curves = curve_row
        self._intents = intent_row

    def row_of(self, sample_id: str) -> int | None:
        try:
            curve_part, node_part, rest = sample_id.split(":", 2)
            intent_part = rest.rsplit(":", 1)[0]
            curve_idx = self._curves[curve_part]
            intent_idx = self._intents[intent_part]
            node_index = int(node_part)
        except (ValueError, KeyError):
            return None
        key = _pack_sid_key(curve_idx, node_index, intent_idx)
        slot = _slot_of(key, self._mask)
        keys = self._keys
        while True:
            stored = int(keys[slot])
            if stored == -1:
                return None
            if stored == key:
                return int(self._vals[slot])
            slot = (slot + 1) & self._mask


# ---------------------------------------------------------------------------
# lazy label tables (columnar views; row objects built per access)
# ---------------------------------------------------------------------------
class _LazyBySample:
    """``by_sample[sid]`` view constructing row dataclasses on access."""

    __slots__ = ("_table",)

    def __init__(self, table: "_ColumnarLabelTables") -> None:
        self._table = table

    def __contains__(self, sample_id: object) -> bool:
        if not isinstance(sample_id, str):
            return False
        return self._table._row_of(str(sample_id)) is not None

    def __getitem__(self, sample_id: str) -> Any:
        row = self._table._row_of(str(sample_id))
        if row is None:
            raise KeyError(sample_id)
        return self._table._row_object(row, str(sample_id))

    def get(self, sample_id: str, default: Any = None) -> Any:
        row = self._table._row_of(str(sample_id))
        if row is None:
            return default
        return self._table._row_object(row, str(sample_id))

    def __len__(self) -> int:
        return self._table._n_rows

    def __iter__(self) -> Iterator[str]:
        for row in range(self._table._n_rows):
            yield self._table._sid_at(row)


class _LazyCurves:
    """``curves[cid]`` view building :class:`CurveV52Labels` per access."""

    __slots__ = ("_table",)

    def __init__(self, table: "_ColumnarLabelTables") -> None:
        self._table = table

    def __contains__(self, cid: object) -> bool:
        return isinstance(cid, str) and cid in self._table._curve_row

    def __getitem__(self, cid: str) -> Any:
        row = self._table._curve_row.get(str(cid))
        if row is None:
            raise KeyError(cid)
        return self._table._curve_object(row)

    def get(self, cid: str, default: Any = None) -> Any:
        row = self._table._curve_row.get(str(cid))
        if row is None:
            return default
        return self._table._curve_object(row)

    def __len__(self) -> int:
        return self._table._n_curves

    def __iter__(self) -> Iterator[str]:
        for cid in self._table._curve_ids:
            yield cid


class _ColumnarLabelTables:
    """Shared columnar backbone of the lazy v52 + endpoint label tables."""

    def __init__(self, dataset: "ColumnarKeyAtomDatasetV3") -> None:
        ds = dataset
        self._ds = ds
        self._n_rows = ds._n_rows
        self._n_curves = ds._n_curves
        self._curve_ids = ds._curve_ids
        self._curve_row = ds._curve_row
        lab = ds._idx_dir / "labels_v52"
        ep = ds._idx_dir / "labels_ep"
        lc = ds._idx_dir / "label_curves"
        self._q = np.load(lab / "q.npy", mmap_mode="r", allow_pickle=False)
        self._q_mask = np.load(lab / "q_mask.npy",
                               mmap_mode="r", allow_pickle=False)
        self._stop = np.load(lab / "stop.npy", mmap_mode="r", allow_pickle=False)
        self._d_key_ts = np.load(lab / "d_key_ts.npy",
                                 mmap_mode="r", allow_pickle=False)
        self._ep_label = np.load(ep / "ep_label.npy",
                                 mmap_mode="r", allow_pickle=False)
        self._d_end = np.load(ep / "d_end.npy", mmap_mode="r", allow_pickle=False)
        self._d_key_ts_ep = np.load(ep / "d_key_ts_ep.npy",
                                    mmap_mode="r", allow_pickle=False)
        self._bond_edit = np.load(ep / "bond_edit_complete.npy",
                                  mmap_mode="r", allow_pickle=False)
        self._geom_legal = np.load(ep / "geometry_legal.npy",
                                   mmap_mode="r", allow_pickle=False)
        self._ts_row = np.load(ep / "ts_row.npy", mmap_mode="r", allow_pickle=False)
        self._start_row = np.load(ep / "start_row.npy",
                                  mmap_mode="r", allow_pickle=False)
        self._terminal_row = np.load(ep / "terminal_row.npy",
                                     mmap_mode="r", allow_pickle=False)
        self._degenerate = np.load(ep / "degenerate_curve.npy",
                                   mmap_mode="r", allow_pickle=False)
        self._energy_ev = np.load(ep / "energy_ev_within_window.npy",
                                  mmap_mode="r", allow_pickle=False)
        self._node_order = np.load(lc / "node_order.npy",
                                   mmap_mode="r", allow_pickle=False)
        self._arc_s = np.load(lc / "arc_s.npy", mmap_mode="r", allow_pickle=False)
        self._q_canon = np.load(lc / "q_canonical.npy",
                                mmap_mode="r", allow_pickle=False)
        # label-curve arrays are packed by label-cache n (irc_curve.n_nodes),
        # which can differ from the light-index node count on real tables —
        # dedicated offsets, never the sample-side node offsets
        self._lab_off = np.load(lc / "offsets.npy", mmap_mode="r",
                                allow_pickle=False)
        self._lab_q_off = np.load(lc / "q_offsets.npy", mmap_mode="r",
                                  allow_pickle=False)
        meta = ds._doc["label_curves"]
        self._lab_n = np.asarray([int(m["lab_n"]) for m in meta], dtype=np.int64)
        self._lab_ts = np.asarray([int(m["ts_index"]) for m in meta],
                                  dtype=np.int64)
        self._lab_has_q = [bool(m["has_q"]) for m in meta]
        self._lab_reason = [str(m["q_zero_arc_reason"]) for m in meta]
        windows = ds._doc["labels"]["endpoint_windows"]
        self.positive_window = float(windows["positive_window_angstrom"])
        self.negative_window = float(windows["negative_window_angstrom"])
        self._sid_lookup = ds._sid_lookup

    # -- shared helpers -------------------------------------------------------
    def _row_of(self, sample_id: str) -> int | None:
        return self._sid_lookup.row_of(sample_id)

    # -- I4 startup fast-path columns (same row order as the dataset) --------
    @property
    def q_col(self) -> Any:
        """float64 q per row (NaN = None) — v52 table, idx row order."""
        return self._q

    @property
    def q_mask_col(self) -> Any:
        """bool q_mask per row — v52 table, idx row order."""
        return self._q_mask

    @property
    def ep_label_col(self) -> Any:
        """int8 endpoint_label per row — endpoint table, idx row order."""
        return self._ep_label

    def covers_all_positions(self, n_expected: int) -> bool:
        """Structural coverage: the pack builder wrote one label row per
        sample IN SAMPLE ORDER, so len == n_samples means every position
        has a row (the builder asserted full coverage fail-closed)."""
        return int(self._n_rows) == int(n_expected)

    def _sid_at(self, row: int) -> str:
        return self._ds._record_at(int(row)).sample_id

    def _identity(self, row: int) -> dict[str, Any]:
        # column reads only — no SampleRecordV3Lite construction (this runs
        # per label-row access, i.e. on the trainer's per-batch hot path)
        ds = self._ds
        cols = ds._columns
        rec_fold = int(cols["fold"][row])
        intent = ds._intent_by_row[int(cols["intent_idx"][row])]
        return {
            "curve_id": ds._curve_ids[int(cols["curve_idx"][row])],
            "node_index": int(cols["node_index"][row]),
            "intent_id": intent.intent_id,
            "direction": str(intent.direction),
            "fold": (None if rec_fold < 0 else rec_fold),
            "label_status": ds._status_of(int(cols["label_status"][row])),
        }

    # -- row builders (override point per table) ------------------------------
    def _row_object(self, row: int, sample_id: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    def _curve_object(self, row: int) -> Any:  # pragma: no cover
        raise NotImplementedError


class _LazyV52LabelTables(_ColumnarLabelTables):
    """V52LabelTable-compatible lazy view (by_sample + curves)."""

    def __init__(self, dataset: "ColumnarKeyAtomDatasetV3") -> None:
        super().__init__(dataset)
        self.by_sample = _LazyBySample(self)
        self.curves = _LazyCurves(self)

    def _row_object(self, row: int, sample_id: str) -> Any:
        from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
            V52SampleLabel,
        )
        q = float(self._q[row])
        return V52SampleLabel(
            sample_id=sample_id,
            **self._identity(row),
            q=(None if np.isnan(q) else q),
            q_mask=bool(self._q_mask[row]),
            stop=int(self._stop[row]),
            d_key_ts=float(self._d_key_ts[row]),
        )

    def _curve_object(self, row: int) -> Any:
        from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
            CurveV52Labels,
        )
        n = int(self._lab_n[row])
        start = int(self._lab_off[row])
        no = tuple(int(v) for v in
                   self._node_order[start:start + n])
        arc = tuple(float(v) for v in self._arc_s[start:start + n])
        qc: tuple[float, ...] | None = None
        if self._lab_has_q[row]:
            q_start = int(self._lab_q_off[row])
            qc = tuple(float(v) for v in self._q_canon[q_start:q_start + n])
        return CurveV52Labels(
            curve_id=str(self._curve_ids[row]),
            ts_index=int(self._lab_ts[row]),
            arc_s=arc,
            q_canonical=qc,
            q_zero_arc_reason=str(self._lab_reason[row]),
            node_order=no,
        )

    @property
    def zero_arc_curves(self) -> tuple[str, ...]:
        return tuple(str(self._curve_ids[row])
                     for row in range(self._n_curves)
                     if not self._lab_has_q[row])

    def q_masked_sample_ids(self) -> tuple[str, ...]:
        rows = np.nonzero(np.asarray(self._q_mask) == False)[0]  # noqa: E712
        return tuple(self._sid_at(int(r)) for r in rows)


class _LazyEndpointLabelTables(_ColumnarLabelTables):
    """EndpointLabelTableV1-compatible lazy view."""

    def __init__(self, dataset: "ColumnarKeyAtomDatasetV3") -> None:
        super().__init__(dataset)
        self.by_sample = _LazyBySample(self)
        self.curve_families = None

    @property
    def n_rows(self) -> int:
        return self._n_rows

    def _row_object(self, row: int, sample_id: str) -> Any:
        from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
            EndpointSampleLabelV1,
        )
        ev = int(self._energy_ev[row])
        return EndpointSampleLabelV1(
            sample_id=sample_id,
            **self._identity(row),
            endpoint_label=int(self._ep_label[row]),
            d_end_angstrom=float(self._d_end[row]),
            d_key_ts_angstrom=float(self._d_key_ts_ep[row]),
            bond_edit_complete=bool(self._bond_edit[row]),
            geometry_legal=bool(self._geom_legal[row]),
            ts_row=bool(self._ts_row[row]),
            start_row=bool(self._start_row[row]),
            terminal_row=bool(self._terminal_row[row]),
            degenerate_curve=bool(self._degenerate[row]),
            energy_ev_within_window=(None if ev < 0 else bool(ev)),
        )

    def _curve_object(self, row: int) -> Any:  # pragma: no cover
        raise KeyError("endpoint label table has no per-curve rows")

    def count_label(self, label: int) -> int:
        return int((np.asarray(self._ep_label) == int(label)).sum())


# ---------------------------------------------------------------------------
# columnar dataset
# ---------------------------------------------------------------------------
class ColumnarStartup:
    """Vectorized per-position columns for the startup passes (I4 / B2).

    The trainer/driver setup makes ~7 full passes over the 5.96M-sample
    lazy sequence (pool split, confirm checks, curve-dir index, curve-id
    sets, label coverage, endpoint pos-weight, constant-q mean) — each
    constructing a record per position (~740 s of pure construction at the
    20k tier).  This object exposes the SAME per-position information as
    numpy columns WITHOUT constructing a single record, so those passes
    become vectorized gathers.  Columns are read-only memmap (or a small
    materialized gather), built lazily on first access.

    Equivalence discipline: the trainer's fast paths consume these columns
    and must produce results IDENTICAL to the per-item path — pinned by
    ``tests/pesn/test_columnar_startup_fast.py`` and re-verified on the
    real tables by ``r3_v53mp_startup_certificate.py``.  The per-item
    fallback paths are kept VERBATIM; ``V53MP_DISABLE_COLUMNAR_FAST=1``
    forces them (the certificate's reference arm).
    """

    __slots__ = ("_ds", "_ci")

    def __init__(self, dataset: "ColumnarKeyAtomDatasetV3") -> None:
        self._ds = dataset
        self._ci: Any = None          # materialized curve_idx (cached)

    # -- raw columns (memmap; read-only) ------------------------------------
    @property
    def curve_idx(self) -> Any:
        return self._ds._columns["curve_idx"]

    @property
    def fold(self) -> Any:
        return self._ds._columns["fold"]

    @property
    def intent_idx(self) -> Any:
        return self._ds._columns["intent_idx"]

    @property
    def d0(self) -> Any:
        return self._ds._columns["d0"]

    @property
    def curve_ids(self) -> list:
        return self._ds._curve_ids

    @property
    def curve_row(self) -> dict:
        return self._ds._curve_row

    @property
    def n(self) -> int:
        return self._ds._n_rows

    @property
    def intent_dir(self) -> Any:
        """int8 [n_intents]: 0 = forward, 1 = reverse (idx encoding)."""
        return self._ds._intent_dir

    def _ci_arr(self) -> Any:
        if self._ci is None:
            self._ci = np.asarray(self.curve_idx)
        return self._ci

    # -- vectorized helpers (each mirrors one per-item comprehension) --------
    def positions_in_curves(self, curve_id_iter: Iterable[str]) -> Any:
        """Ascending positions whose curve_id is in the given set.

        Mirrors ``[i for i, s in enumerate(samples) if s.curve_id in ids]``.
        """
        wanted = sorted({self.curve_row[str(c)] for c in curve_id_iter
                         if str(c) in self.curve_row})
        if not wanted:
            return np.zeros(0, dtype=np.int64)
        ci = self._ci_arr()
        mask = np.isin(ci, np.asarray(wanted, dtype=np.int64))
        return np.flatnonzero(mask).astype(np.int64)

    def unique_sorted_curve_ids(self, positions: Any) -> list:
        """Sorted (by string) unique curve_ids of ``positions``.

        Mirrors ``sorted({str(samples[i].curve_id) for i in positions})`` —
        STRING sort (the seeded rng draws depend on this exact order).
        """
        pos = np.asarray(positions, dtype=np.int64)
        seen = np.unique(self._ci_arr()[pos])
        return sorted(self.curve_ids[int(c)] for c in seen)

    def curve_id_set(self, positions: Any) -> set:
        """Set of curve_ids of ``positions`` (mirrors the per-item set)."""
        pos = np.asarray(positions, dtype=np.int64)
        seen = np.unique(self._ci_arr()[pos])
        return {self.curve_ids[int(c)] for c in seen}

    def fold_array(self, positions: Any) -> Any:
        """int8 fold per position (-1 = None), gathered ascending."""
        pos = np.asarray(positions, dtype=np.int64)
        return np.asarray(self.fold)[pos]

    def direction_int(self, positions: Any) -> Any:
        """int8 direction per position (0=forward,1=reverse), gathered."""
        pos = np.asarray(positions, dtype=np.int64)
        return self.intent_dir[np.asarray(self.intent_idx)[pos]]

    def sample_id_at(self, position: int) -> str:
        """One reconstructed sample_id (probes only — NOT a bulk path)."""
        return self._ds._record_at(int(position)).sample_id


class ColumnarKeyAtomDatasetV3(ShardedKeyAtomDatasetV3):
    """Loader v2: v1 API + shard geometry LRU, columnar memmap light index.

    Same constructor surface as :class:`ShardedKeyAtomDatasetV3` (``light_cache``
    is accepted and ignored — v2 IS the light index).  ``samples`` is a LAZY
    sequence (single-pass consumers only; the trainer/driver access pattern
    was audited: enumerate / ``samples[i]`` / ``records(pos)`` — nothing
    materialises or double-buffers the record set).  Label tables come from
    :meth:`label_tables` as lazy per-row views over the same memmaps (same
    access surface the trainer/``evaluate_v53mp`` use: ``by_sample[sid]``,
    ``.q/.q_mask/.curve_id/.direction``, ``.endpoint_label/.ts_row/
    .start_row``, table windows).
    """

    def __init__(
        self,
        tables_dir: str | Path,
        *,
        d0: float = D0_PRIMARY_ANGSTROM,
        label_statuses: Iterable[str] = ("ok", "terminal", "masked",
                                         "conflict"),
        cache_curves: int = 512,
        verify_manifest: bool = True,
        split_manifest: str | Path | None = None,
        fold: int | None = None,
        directions: Iterable[str] = ("forward", "reverse"),
        light_cache: bool = False,          # accepted, ignored (v2 IS light)
        verify_sources: bool = False,
        use_geometry_pack: bool = True,     # I2: geom/ memmap when present
    ) -> None:
        t_init = time.perf_counter()
        del light_cache
        self.tables_dir = Path(tables_dir)
        self.d0 = float(d0)
        self.fold = fold
        self.label_statuses = frozenset(label_statuses)
        self.directions = frozenset(directions)
        self.cache_curves = int(cache_curves)
        self.verify_manifest = bool(verify_manifest)
        self.verify_sources = bool(verify_sources)
        self._split_manifest_input = split_manifest
        _require(self.tables_dir.is_dir(), "TABLES_DIR",
                 f"{self.tables_dir} is not a directory")
        _require(self.cache_curves >= 1, "CACHE_CURVES",
                 f"cache_curves must be >= 1, got {self.cache_curves}")

        # --- presence of all six tables (v1 parity; geometry CSV fallback) ---
        from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
            TABLE_FILENAMES,
        )
        for table, filename in TABLE_FILENAMES.items():
            _require((self.tables_dir / filename).is_file(),
                     "TABLE_MISSING", f"{table} -> {filename} not found")

        self._idx_dir = self.tables_dir / IDX2_DIRNAME
        _require(self._idx_dir.is_dir(), "IDX2_MISSING",
                 f"{self._idx_dir} not found — run r3_v53mp_build_idx_v2.py")

        # --- binding block (fail-closed) -------------------------------------
        curves_path = self._idx_dir / "curves.json"
        _require(curves_path.is_file(), "IDX2_MISSING",
                 f"{curves_path} not found")
        self._doc = json.loads(curves_path.read_text(encoding="utf-8"))
        _require(self._doc.get("schema") == IDX2_SCHEMA, "IDX2_SCHEMA",
                 f"curves.json schema {self._doc.get('schema')!r} != "
                 f"{IDX2_SCHEMA!r}")
        binding = self._doc.get("binding")
        _require(isinstance(binding, Mapping), "IDX2_SCHEMA",
                 "curves.json has no binding block")
        self.idx_binding = dict(binding)
        self._verify_binding(binding)

        # --- columnar files: presence + integrity + memmap open --------------
        self._verify_idx_files()
        self._columns: dict[str, Any] = {}
        for field in SAMPLE_FIELDS:
            self._columns[field] = np.load(
                self._idx_dir / "samples" / f"{field}.npy",
                mmap_mode="r", allow_pickle=False)
        n_rows = int(self._columns["curve_idx"].shape[0])
        _require(all(int(self._columns[f].shape[0]) == n_rows
                     for f in SAMPLE_FIELDS),
                 "IDX2_SHAPE", "sample columns disagree on length")
        self._n_rows = n_rows
        _require(self._n_rows > 0, "IDX2_SHAPE", "idx_v2 has no samples")
        _require(int(binding["n_samples"]) == self._n_rows, "IDX2_BINDING",
                 f"binding n_samples {binding['n_samples']} != "
                 f"{self._n_rows} rows on disk")
        self._actions = np.load(self._idx_dir / "actions.npy",
                                mmap_mode="r", allow_pickle=False)
        _require(self._actions.ndim == 2 and self._actions.shape[1] == 3
                 and self._actions.dtype == np.float32,
                 "IDX2_SHAPE", f"actions.npy bad shape/dtype "
                 f"{self._actions.shape} {self._actions.dtype}")

        # --- curve + intent string tables ------------------------------------
        self._curve_ids: list[str] = [str(c) for c in self._doc["curve_ids"]]
        self._n_curves = len(self._curve_ids)
        _require(int(binding["n_curves"]) == self._n_curves, "IDX2_BINDING",
                 "binding n_curves disagrees with curve table")
        self._curve_row = {cid: i for i, cid in enumerate(self._curve_ids)}
        _require(len(self._curve_row) == self._n_curves, "IDX2_SCHEMA",
                 "duplicate curve_id in idx curve table")
        self._curves: dict[str, dict[str, Any]] = {}
        self._curve_atomic: dict[str, tuple[int, ...]] = {}
        self._node_offsets = np.zeros(self._n_curves + 1, dtype=np.int64)
        self._has_energy = np.zeros(self._n_curves, dtype=bool)
        for row, meta in enumerate(self._doc["curves"]):
            cid = self._curve_ids[row]
            self._curves[cid] = {
                "ts_index": int(meta["ts_index"]),
                "endpoint_a_index": int(meta["endpoint_a"]),
                "endpoint_b_index": int(meta["endpoint_b"]),
                "n_nodes": int(meta["n_nodes"]),
            }
            self._curve_atomic[cid] = tuple(
                int(a) for a in meta["atomic_numbers"])
            self._node_offsets[row + 1] = (
                self._node_offsets[row] + int(meta["n_nodes"]))
            self._has_energy[row] = bool(meta["has_energy"])
        self._node_index = _RangeNodeIndex(
            {cid: int(meta["n_nodes"])
             for cid, meta in zip(self._curve_ids, self._doc["curves"])})

        self._intents, self._intent_by_row, self._intent_row = \
            self._load_intents()
        self._sid_lookup = _SidLookup(self._idx_dir, self._curve_row,
                                      self._intent_row)

        # --- folds + shard fast path (v1 code paths, same attributes) --------
        _require(self.fold is None or self.fold == binding.get("fold"),
                 "IDX2_BINDING",
                 f"constructor fold {self.fold} != idx binding fold "
                 f"{binding.get('fold')!r} (idx_v2 is fold-fixed; rebuild "
                 "the idx for fold filtering)")
        self._folds = _load_folds_at(self.tables_dir, split_manifest, fold)
        self._shard_curve_path: dict[str, str] = {}
        self._shard_sha: dict[str, str] = {}
        self._shard_verified: set[str] = set()
        self.shard_verify_seconds = 0.0
        self._load_shard_index()
        self._preverify_shards()

        # --- geometry memmap pack (plan §5.2 / lane I2; optional) ------------
        # Active only when geom/curve_offsets.json exists AND its binding
        # matches the LIVE shard_index + data_manifest shas.  When active,
        # frame_xyz / to_sample_v3 read single-node slices straight from the
        # coords memmap — the shard LRU never loads (cold-start and anon win);
        # when absent, the v1 shard-LRU path above serves everything.
        self._geom_coords: Any = None
        self.geom_pack_binding: dict | None = None
        self._geom_offset: dict[str, int] = {}
        self._geom_natoms: dict[str, int] = {}
        self.geom_verify_seconds = 0.0
        if use_geometry_pack:
            self._load_geometry_pack(verify=self.verify_manifest)

        # --- v52_labels() fallback cache + geometry LRU (empty at init) ------
        self._frames: "OrderedDict[str, dict[int, np.ndarray]]" = OrderedDict()
        self._v52_cache: dict[tuple[float, float, float], Any] = {}

        # I4 startup fast-path columns (lazy; no records constructed)
        self._startup = ColumnarStartup(self)
        self.manifest_sha256 = str(binding["data_manifest_sha256"])
        self.init_seconds = time.perf_counter() - t_init

    @property
    def startup(self) -> "ColumnarStartup":
        """Vectorized startup columns (I4); consumed by the trainer fast
        paths.  Constructed eagerly (cheap), gathers lazily."""
        return self._startup

    # -- binding verification -------------------------------------------------
    def _verify_binding(self, binding: Mapping[str, Any]) -> None:
        """Refuse a stale/drifted idx (typed errors only; never serve)."""
        _require(float(binding["d0"]) == self.d0, "IDX2_BINDING",
                 f"idx built for d0={binding['d0']}, requested {self.d0}")
        _require(sorted(binding["label_statuses"])
                 == sorted(self.label_statuses), "IDX2_BINDING",
                 f"idx label_statuses {sorted(binding['label_statuses'])} != "
                 f"{sorted(self.label_statuses)}")
        _require(sorted(binding["directions"]) == sorted(self.directions),
                 "IDX2_BINDING",
                 f"idx directions {sorted(binding['directions'])} != "
                 f"{sorted(self.directions)}")
        _require(binding.get("fold") is None, "IDX2_BINDING",
                 f"idx binding fold {binding.get('fold')!r} is not the "
                 "unfiltered build")
        # live manifest / split shas (cheap; fail-closed on ANY drift)
        manifest_path = self.tables_dir / "data_manifest.json"
        live_manifest = (_sha256(manifest_path)
                         if manifest_path.is_file() else "")
        _require(live_manifest == str(binding["data_manifest_sha256"]),
                 "IDX2_BINDING",
                 "data_manifest.json sha drifted from idx binding "
                 f"({live_manifest[:12]}… vs "
                 f"{str(binding['data_manifest_sha256'])[:12]}…)")
        split_input = self._split_manifest_input
        split_path = (Path(split_input) if split_input is not None
                      else self.tables_dir / "split_manifest_v53mp.json")
        live_split = _sha256(split_path) if split_path.is_file() else ""
        _require(live_split == str(binding["split_manifest_sha256"]),
                 "IDX2_BINDING",
                 "split manifest sha drifted from idx binding "
                 f"({live_split[:12]}… vs "
                 f"{str(binding['split_manifest_sha256'])[:12]}…)")
        # source artifacts: size+mtime always, full sha on demand
        for key in ("light_pickle", "labels_npz", "labels_json"):
            entry = binding.get(f"source_{key}")
            if not isinstance(entry, Mapping):
                continue
            path = self.tables_dir / str(entry["path"])
            _require(path.is_file(), "IDX2_BINDING",
                     f"source {entry['path']} listed in idx binding is gone")
            stat = path.stat()
            _require(int(entry["size"]) == int(stat.st_size)
                     and int(entry["mtime_ns"]) == int(stat.st_mtime_ns),
                     "IDX2_BINDING",
                     f"source {entry['path']} size/mtime_ns drifted from idx "
                     "binding (rebuild idx_v2)")
            if self.verify_sources and str(entry["sha256"]):
                # full re-hash of the multi-GB sources costs ~23 s on drvfs;
                # opt-in only (the builder bound them at build time; any
                # rewrite changes size/mtime_ns and is refused above)
                actual = _sha256(path)
                _require(actual == str(entry["sha256"]), "IDX2_BINDING",
                         f"source {entry['path']}: sha256 "
                         f"{actual[:12]}… != binding "
                         f"{str(entry['sha256'])[:12]}…")
        created = str(self._doc.get("created_at", ""))
        _require(bool(created), "IDX2_SCHEMA", "curves.json has no created_at")

    def _verify_idx_files(self) -> None:
        """Verify every idx file's sha256 (threaded; ~1 GB at 20k ≈ 4 s)."""
        files = self._doc.get("files")
        _require(isinstance(files, Mapping) and files, "IDX2_SCHEMA",
                 "curves.json has no files manifest")
        t0 = time.perf_counter()

        def _hash_one(rel: str) -> tuple[str, str]:
            return rel, _sha256(self._idx_dir / rel)

        if self.verify_manifest:
            rels = sorted(files)
            with ThreadPoolExecutor(max_workers=min(8, len(rels))) as pool:
                for rel, actual in pool.map(_hash_one, rels):
                    meta = files[rel]
                    _require((self._idx_dir / rel).is_file(), "IDX2_MISSING",
                             f"idx file {rel} missing")
                    _require(actual == str(meta["sha256"]), "IDX2_SHA",
                             f"idx file {rel}: sha256 {actual[:12]}… != "
                             f"{str(meta['sha256'])[:12]}…")
        else:
            for rel in files:
                _require((self._idx_dir / rel).is_file(), "IDX2_MISSING",
                         f"idx file {rel} missing")
        self.idx_verify_seconds = time.perf_counter() - t0

    # -- geometry memmap pack (plan §5.2 / lane I2) ---------------------------
    def _load_geometry_pack(self, *, verify: bool) -> None:
        """Open ``geom/`` when present; refuse on binding drift (fail-closed).

        Absence is NOT an error — the v1 shard-LRU path keeps serving (the
        pack is an optional accelerator).  Presence with a drifted binding
        (shard_index sha, data_manifest sha, curve coverage, coords sha)
        IS an error: a stale pack must never silently serve wrong geometry.
        """
        geom_dir = self.tables_dir / GEOM_DIRNAME
        offsets_path = geom_dir / GEOM_OFFSETS_FILENAME
        if not offsets_path.is_file():
            return                          # no pack -> shard LRU (v1 path)
        t0 = time.perf_counter()
        doc = json.loads(offsets_path.read_text(encoding="utf-8"))
        _require(str(doc.get("schema")) == GEOM_SCHEMA, "GEOM_SCHEMA",
                 f"geom offsets schema {doc.get('schema')!r} != "
                 f"{GEOM_SCHEMA!r}")
        binding = doc.get("binding")
        _require(isinstance(binding, Mapping), "GEOM_SCHEMA",
                 "geom offsets have no binding block")
        # binding must match the LIVE shard index + data manifest
        index_path = self.tables_dir / "shard_index_v53mp.json"
        live_index = _sha256(index_path) if index_path.is_file() else ""
        _require(live_index == str(binding.get("shard_index_sha256")),
                 "GEOM_BINDING",
                 "shard_index_v53mp.json sha drifted from geom binding "
                 f"({live_index[:12]}… vs "
                 f"{str(binding.get('shard_index_sha256'))[:12]}…)")
        manifest_path = self.tables_dir / "data_manifest.json"
        live_manifest = _sha256(manifest_path) if manifest_path.is_file() else ""
        _require(live_manifest == str(binding.get("data_manifest_sha256")),
                 "GEOM_BINDING",
                 "data_manifest.json sha drifted from geom binding "
                 f"({live_manifest[:12]}… vs "
                 f"{str(binding.get('data_manifest_sha256'))[:12]}…)")
        curves = doc.get("curves")
        _require(isinstance(curves, Mapping) and curves, "GEOM_SCHEMA",
                 "geom offsets have no curve table")
        # every idx curve must be covered; per-curve n_nodes must agree
        for cid in self._curve_ids:
            _require(cid in curves, "GEOM_COVERAGE",
                     f"curve {cid} missing from geom pack")
            entry = curves[cid]
            n_nodes = self._curves[cid]["n_nodes"]
            _require(int(entry["n_nodes"]) == int(n_nodes), "GEOM_COVERAGE",
                     f"curve {cid}: geom n_nodes {entry['n_nodes']} != idx "
                     f"{n_nodes}")
        coords_path = geom_dir / str(binding.get("coords_filename",
                                                GEOM_COORDS_FILENAME))
        _require(coords_path.is_file(), "GEOM_MISSING",
                 f"{coords_path} not found")
        if verify and str(binding.get("coords_sha256")):
            actual = _sha256(coords_path)
            _require(actual == str(binding["coords_sha256"]), "GEOM_SHA",
                     f"coords file sha256 {actual[:12]}… != binding "
                     f"{str(binding['coords_sha256'])[:12]}…")
        self._geom_coords = np.load(coords_path, mmap_mode="r",
                                    allow_pickle=False)
        _require(self._geom_coords.ndim == 2
                 and self._geom_coords.shape[1] == 3
                 and self._geom_coords.dtype == np.float64,
                 "GEOM_SHAPE", f"coords file has shape "
                 f"{self._geom_coords.shape} dtype {self._geom_coords.dtype}")
        _require(int(self._geom_coords.shape[0])
                 == int(binding.get("total_atom_rows", -1)), "GEOM_SHAPE",
                 "coords length disagrees with binding total_atom_rows")
        # fast per-curve lookup tables (atom-row offsets)
        self._geom_offset: dict[str, int] = {
            cid: int(curves[cid]["offset"]) for cid in self._curve_ids}
        self._geom_natoms: dict[str, int] = {
            cid: int(curves[cid]["n_atoms"]) for cid in self._curve_ids}
        self.geom_pack_binding = dict(binding)
        self.geom_verify_seconds = time.perf_counter() - t0

    def _geom_frame_array(self, curve_id: str,
                          node_index: int) -> Any | None:
        """One ``[n_atoms, 3]`` float64 frame from the geom pack (None = off).

        Returns ``None`` when no pack is loaded so every caller keeps its
        shard-LRU fallback.  Node order follows the pack (ascending node
        index, asserted at idx build time) — same bytes as the shard path.
        """
        if self._geom_coords is None:
            return None
        cid = str(curve_id)
        offset = self._geom_offset.get(cid)
        _require(offset is not None, "CURVE_UNKNOWN",
                 f"curve {cid} not in geom pack")
        n_nodes = int(self._curves[cid]["n_nodes"])
        node = int(node_index)
        _require(0 <= node < n_nodes, "NODE_UNKNOWN",
                 f"curve {cid} has no node {node}")
        n_atoms = self._geom_natoms[cid]
        base = offset + node * n_atoms
        return np.asarray(self._geom_coords[base:base + n_atoms])

    # -- intents --------------------------------------------------------------
    def _load_intents(self):
        """Rebuild IntentRecordV3 objects from the columnar intent table."""
        it = self._idx_dir / "intents"
        ids = [str(s) for s in np.load(it / "intent_ids.npy",
                                       allow_pickle=False)]
        curve_idx = np.load(it / "curve_idx.npy", mmap_mode="r",
                            allow_pickle=False)
        direction = np.load(it / "direction.npy", mmap_mode="r",
                            allow_pickle=False)
        # materialized int8 direction per intent (I4 startup fast path;
        # 52k x int8 = 52 KB — cheap, reused by every startup gather)
        self._intent_dir = np.asarray(direction, dtype=np.int8)
        reverse_of = np.load(it / "reverse_of.npy", mmap_mode="r",
                             allow_pickle=False)
        pair_ids = [str(s) for s in np.load(it / "pair_ids.npy",
                                            allow_pickle=False)]
        g_src_off = np.load(it / "g_src_off.npy", mmap_mode="r",
                            allow_pickle=False)
        g_tgt_off = np.load(it / "g_tgt_off.npy", mmap_mode="r",
                            allow_pickle=False)
        bond_i = np.load(it / "bond_i.npy", mmap_mode="r", allow_pickle=False)
        bond_j = np.load(it / "bond_j.npy", mmap_mode="r", allow_pickle=False)
        bond_order = np.load(it / "bond_order.npy", mmap_mode="r",
                             allow_pickle=False)
        cv_off = np.load(it / "cv_off.npy", mmap_mode="r", allow_pickle=False)
        cv_deltas = np.load(it / "cv_deltas.npy", mmap_mode="r",
                            allow_pickle=False)
        cv_coords = np.load(it / "cv_coords.npy", mmap_mode="r",
                            allow_pickle=False)
        cv_coords_off = np.load(it / "cv_coords_off.npy", mmap_mode="r",
                                allow_pickle=False)
        g_atoms = np.load(it / "g_atoms.npy", mmap_mode="r",
                          allow_pickle=False)
        g_atoms_off = np.load(it / "g_atoms_off.npy", mmap_mode="r",
                              allow_pickle=False)
        version_ids = [str(s) for s in np.load(it / "version_ids.npy",
                                               allow_pickle=False)]
        underdet = np.load(it / "underdetermined.npy", mmap_mode="r",
                           allow_pickle=False)
        cv_kind_ids = [str(s) for s in np.load(it / "cv_kind_ids.npy",
                                               allow_pickle=False)]
        n = len(ids)
        intents: dict[tuple[str, str], IntentRecordV3] = {}
        by_row: list[IntentRecordV3] = []
        row_of: dict[str, int] = {}
        for row in range(n):
            cid = self._curve_ids[int(curve_idx[row])]
            gs, ge = int(g_src_off[row]), int(g_src_off[row + 1])
            ts, te = int(g_tgt_off[row]), int(g_tgt_off[row + 1])
            g_src = tuple(
                (int(bond_i[k]), int(bond_j[k]), float(bond_order[k]))
                for k in range(gs, ge))
            g_tgt = tuple(
                (int(bond_i[k]), int(bond_j[k]), float(bond_order[k]))
                for k in range(ts, te))
            cs, ce = int(cv_off[row]), int(cv_off[row + 1])
            cv_deltas_t = tuple(float(v) for v in cv_deltas[cs:ce])
            cc_s, cc_e = int(cv_coords_off[row]), int(cv_coords_off[row + 1])
            coords = tuple(
                (int(cv_coords[k]), int(cv_coords[k + 1]))
                for k in range(cc_s, cc_e, 2))
            ga_s, ga_e = int(g_atoms_off[row]), int(g_atoms_off[row + 1])
            atoms = tuple(int(v) for v in g_atoms[ga_s:ga_e])
            rev_idx = int(reverse_of[row])
            rev = ids[rev_idx] if rev_idx >= 0 else ""
            intent = IntentRecordV3(
                intent_id=ids[row],
                curve_id=cid,
                direction=("forward" if int(direction[row]) == 0
                           else "reverse"),
                reverse_of_intent_id=rev,
                intent_pair_id=pair_ids[row],
                g_src_bonds=g_src,
                g_tgt_bonds=g_tgt,
                cv_deltas=cv_deltas_t,
                intent_version=version_ids[row],
                intent_underdetermined=bool(underdet[row]),
                g_atoms=atoms,
                cv_coordinates=coords,
                cv_kind=cv_kind_ids[row],
            )
            key = (cid, intent.intent_id)
            _require(key not in intents, "IDX2_SHAPE",
                     f"duplicate intent {key} in idx intent table")
            intents[key] = intent
            by_row.append(intent)
            row_of[intent.intent_id] = row
        _require(len(row_of) == n, "IDX2_SHAPE",
                 "intent_id is not globally unique in the idx intent table")
        return intents, by_row, row_of

    # -- record construction ---------------------------------------------------
    def _record_at(self, row: int) -> SampleRecordV3Lite:
        cols = self._columns
        curve_idx = int(cols["curve_idx"][row])
        curve_id = self._curve_ids[curve_idx]
        intent = self._intent_by_row[int(cols["intent_idx"][row])]
        node_index = int(cols["node_index"][row])
        d0 = float(cols["d0"][row])
        fold8 = int(cols["fold"][row])
        target_node = int(cols["target_node"][row])
        action_len = int(cols["action_len"][row])
        target_u: tuple[tuple[float, float, float], ...] | None = None
        if action_len > 0:
            off = int(cols["action_off"][row])
            # tolist() → native floats: ~3x faster than per-element float()
            # over the memmap slice (hot path: every records()/samples[i])
            target_u = tuple(
                map(tuple, self._actions[off:off + action_len].tolist()))
        return SampleRecordV3Lite(
            sample_id=f"{curve_id}:{node_index}:{intent.intent_id}:{d0}",
            curve_id=curve_id,
            node_index=node_index,
            intent=intent,
            d0=d0,
            label_weight=float(cols["label_weight"][row]),
            curve_weight=float(cols["curve_weight"][row]),
            sample_weight=float(cols["sample_weight"][row]),
            label_status=self._status_of(int(cols["label_status"][row])),
            atomic_numbers=self._curve_atomic[curve_id],
            target_action_u=target_u,
            target_node_index=(None if target_node < 0 else target_node),
            fold=(None if fold8 < 0 else fold8),
        )

    def _status_of(self, code: int) -> str:
        table = self._doc["label_status_table"]
        _require(0 <= code < len(table), "IDX2_SHAPE",
                 f"label_status code {code} out of range")
        return str(table[code])

    # -- public API ------------------------------------------------------------
    def __len__(self) -> int:
        return self._n_rows

    @property
    def samples(self) -> Sequence[SampleRecordV3Lite]:
        """LAZY sequence (constructs records on demand; never a tuple)."""
        return _LazySampleSequence(self)

    def records(
        self, indices: Iterable[int] | None = None,
    ) -> list[SampleRecordV3Lite]:
        if indices is None:
            return [self._record_at(row) for row in range(self._n_rows)]
        return [self._record_at(int(i)) for i in indices]

    def label_tables(self) -> tuple[Any, Any]:
        """Lazy (v52, endpoint) label tables over the SAME idx row order."""
        return (_LazyV52LabelTables(self), _LazyEndpointLabelTables(self))

    def frame_xyz(self, curve_id: str, node_index: int) -> tuple:
        """Full ``[N,3]`` frame geometry.

        Geometry-pack path (plan §5.2): a single memmap slice per call —
        no shard decompression, no LRU.  Falls back to the v1 shard-LRU
        path when no pack is loaded.
        """
        arr = self._geom_frame_array(curve_id, node_index)
        if arr is None:
            return super().frame_xyz(curve_id, node_index)
        return tuple(tuple(float(v) for v in row) for row in arr)

    def to_sample_v3(self, rec: SampleRecordV3Lite) -> Any:
        """Assemble the full v3 sample (geom-pack fast path; v1 fallback).

        Mirrors ``ShardedKeyAtomDatasetV3.to_sample_v3`` exactly (same
        shape checks, same float64→tuple conversion) but sources the frame
        from the geom memmap when the pack is active.
        """
        if self._geom_coords is None:
            return super().to_sample_v3(rec)
        arr = self._geom_frame_array(rec.curve_id, rec.node_index)
        n = len(rec.atomic_numbers)
        if arr.shape == (n, 3):
            xyz = tuple(tuple(float(v) for v in row) for row in arr)
        elif arr.shape == (1, 3) and n == 1:
            xyz = (tuple(float(v) for v in arr[0]),)
        else:
            raise KeyAtomDatasetV3Error(
                "GEOMETRY_INCOMPLETE",
                f"curve {rec.curve_id} node {rec.node_index}: frame "
                f"geometry {tuple(arr.shape)} cannot cover N={n} atoms")
        from pes2ts_core.pesn.contracts.key_atom_v3 import KeyAtomSampleV3
        return KeyAtomSampleV3(
            curve_id=rec.curve_id,
            node_index=rec.node_index,
            intent=rec.intent,
            atomic_numbers=rec.atomic_numbers,
            xyz=xyz,
            d0=rec.d0,
            endpoint_names=("A", "B"),
            y_teacher=rec.target_action_u,
            frame_weight=rec.label_weight,
            label_status=rec.label_status,
            sample_id=rec.sample_id,
            extras={"curve_weight": rec.curve_weight,
                    "sample_weight": rec.sample_weight,
                    "fold": rec.fold,
                    "feature_sha": "",
                    "label_sha": ""},
        )

    def curve_energies(self, curve_id: str) -> dict[int, float] | None:
        """Per-node energies (Hartree→eV as stored) or None if incomplete."""
        row = self._curve_row.get(str(curve_id))
        if row is None or not bool(self._has_energy[row]):
            return None
        n = int(self._curves[str(curve_id)]["n_nodes"])
        start = int(self._node_offsets[row])
        # energies array is opened lazily (most runs never call this)
        if not hasattr(self, "_energies"):
            self._energies = np.load(self._idx_dir / "curve_node_energies.npy",
                                     mmap_mode="r", allow_pickle=False)
        values = self._energies[start:start + n]
        if bool(np.isnan(values).any()):
            return None
        return {node: float(v) for node, v in enumerate(values)}


class _RangeNodeIndex:
    """``curve_id -> set(range(n_nodes))`` on demand (shard node-set check)."""

    __slots__ = ("_n",)

    def __init__(self, n_nodes: Mapping[str, int]) -> None:
        self._n = dict(n_nodes)

    def get(self, curve_id: str, default: Any = None) -> Any:
        n = self._n.get(str(curve_id))
        return set(range(n)) if n is not None else default


__all__ = [
    "IDX2_DIRNAME",
    "IDX2_SCHEMA",
    "SAMPLE_FIELDS",
    "LABELS_V52_FIELDS",
    "LABELS_EP_FIELDS",
    "LABEL_CURVE_FIELDS",
    "INTENT_FIELDS",
    "KeyAtomDatasetV3Error",
    "ColumnarKeyAtomDatasetV3",
]
