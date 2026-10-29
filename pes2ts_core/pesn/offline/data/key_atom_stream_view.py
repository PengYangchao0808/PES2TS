"""Streaming thin view over the frozen six tables (WP5-CERT, V5.4.1).

The 50k-scale loader-v2 certificate cannot afford the v1 reference's full
residency (a 14.7M-entry ``SampleRecordV3Lite`` list ≈ 30 GB + the eagerly
expanded label-cache dicts ≈ 10 GB co-resident OOMs a 12 GiB host).  This
module provides the bounded alternative used by
``scripts/agent_ops/data_gate/r3_v53mp_loader_v2_certificate.py --mode
streaming``:

* :class:`StreamThinViewV3` — light tables (``irc_curve`` /
  ``reaction_intent`` / folds / shard index) + ONE streaming pass over
  ``irc_node.csv`` (node sets + per-curve atomic numbers; energies are NOT
  collected — they are only consumed by endpoint-label recomputation, which
  this structural certificate never performs).  ``sample_view.csv`` is
  consumed in TWO bounded passes: pass 1 builds compact numpy columns
  (``curve_idx`` / ``node_index`` / ``intent_idx`` / ``direction`` /
  ``fold``) in v1 survival order plus the full-group curve-weight
  invariant; pass 2 materialises full ``SampleRecordV3Lite`` objects for
  PROBE positions only (via the very same v1 ``_parse_sample_lite`` — the
  method is inherited-by-assignment, never copied).
* :class:`CompactCurveDirView` — a per-item sampler view (curve_id /
  direction / fold only) so ``v53mp_pool_split`` / ``_CurveDirIndex`` /
  ``_assert_no_confirm`` run their ORIGINAL v1 per-item code paths without
  any record residency.  It deliberately carries no ``_ds`` attribute, so
  ``_startup_of`` returns ``None`` (the v1 reference arm).
* :class:`NpzColumnMemmaps` — zero-copy ``np.memmap`` views onto the
  uncompressed members of ``labels_cache_v53mp.npz`` (``np.savez`` writes
  ``ZIP_STORED``), enabling per-probe-row NumPy column comparison without
  ever expanding ``by_sample`` dicts.  A compressed member is a typed
  refusal.

Every v1 parse/geometry routine reused here is invoked UNBOUND against this
class carrying the same attribute surface (``tables_dir`` /
``verify_manifest`` / ``_node_index`` / ``_frames`` / ``cache_curves`` /
``_shard_*`` / ``d0`` / ``label_statuses`` / ``directions`` / ``fold`` /
``_intents`` / ``_curve_atomic``) — zero logic duplication, so probe-row
records and geometry are bit-identical to the full v1 loader by
construction.

Deviations from the full v1 loader (certificate-scoped, recorded in the
report): energies are not scanned; light-cache read/write is not used
(the certificate hashes the consumed files against ``data_manifest.json``
instead); node FK / intent-involution / curve-weight invariants are re-run
verbatim.
"""

from __future__ import annotations

import ast
import csv
import json
import struct
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: F401
    D0_PRIMARY_ANGSTROM,
    IntentRecordV3,
    KeyAtomV3ContractError,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: F402
    TABLE_FILENAMES,
    WEIGHT_SUM_TOL,
    KeyAtomDatasetV3Error,
    SampleRecordV3Lite,
    ShardedKeyAtomDatasetV3,
    _check_columns,
    _load_folds_at,
    _read_csv,
    _require,
    _require_columns_in,
)

#: Light scan column subset of ``irc_node.csv`` (energies deliberately NOT
#: scanned — see module docstring).
_NODE_SCAN_COLUMNS: Final[tuple[str, ...]] = (
    "curve_id", "node_index", "atomic_numbers")

#: ``sample_view.csv`` fields the streaming pass needs (coords payloads are
#: never parsed).
_SAMPLE_FIELDS: Final[tuple[str, ...]] = (
    "sample_id", "curve_id", "node_index", "intent_id", "d0",
    "label_weight", "curve_weight", "sample_weight", "label_status",
    "target_node_index", "target_action_u", "fold")

_NPY_MAGIC: Final[bytes] = b"\x93NUMPY"


class StreamThinViewError(RuntimeError):
    """Typed refusal of the streaming thin view."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _stream_require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise StreamThinViewError(code, detail)


# ---------------------------------------------------------------------------
# npz -> memmap bridge (labels_cache_v53mp.npz is ZIP_STORED)
# ---------------------------------------------------------------------------
def _read_npy_header(fh, offset: int) -> tuple[np.dtype, tuple[int, ...], int]:
    """Parse the ``.npy`` header of a member stored at ``offset``.

    Returns ``(dtype, shape, data_offset)`` — the absolute byte offset of
    the raw C-order payload inside the container file.
    """
    fh.seek(offset)
    magic = fh.read(6)
    if magic != _NPY_MAGIC:
        raise StreamThinViewError(
            "NPY_HEADER", f"member at offset {offset} is not a .npy stream")
    major = int(fh.read(1)[0])
    fh.read(1)                                   # minor version
    hlen = struct.unpack("<H", fh.read(2))[0] if major == 1 \
        else struct.unpack("<I", fh.read(4))[0]
    header = ast.literal_eval(fh.read(hlen).decode("latin1"))
    if header.get("fortran_order"):
        raise StreamThinViewError(
            "NPY_HEADER", "fortran_order members are not supported")
    descr = header.get("descr")
    if isinstance(descr, list):
        raise StreamThinViewError(
            "NPY_HEADER", "structured dtypes are not supported")
    return np.dtype(descr), tuple(int(x) for x in header["shape"]), fh.tell()


class NpzColumnMemmaps:
    """Lazy ``np.memmap`` views onto every member of an UNCOMPRESSED ``npz``.

    ``np.savez`` stores members with ``ZIP_STORED``; each member is a raw
    ``.npy`` stream at a fixed byte offset, so columns (including the 3.7 GB
    ``sample_ids`` U64 array) can be memory-mapped directly out of the
    container — page-cache backed, never anon-resident.  A compressed
    member is a typed refusal (the certificate's memory model depends on
    offset arithmetic).
    """

    def __init__(self, npz_path: str | Path) -> None:
        self.npz_path = Path(npz_path)
        _stream_require(self.npz_path.is_file(), "NPZ_MISSING",
                        f"{self.npz_path} not found")
        self._offsets: dict[str, tuple[np.dtype, tuple[int, ...], int]] = {}
        self._cache: dict[str, np.ndarray] = {}
        with zipfile.ZipFile(self.npz_path) as zf, \
                self.npz_path.open("rb") as fh:
            for info in zf.infolist():
                if info.compress_type != zipfile.ZIP_STORED:
                    raise StreamThinViewError(
                        "NPZ_COMPRESSED",
                        f"member {info.filename} is compressed "
                        f"(compress_type={info.compress_type}) — memmap "
                        "bridge requires ZIP_STORED")
                fh.seek(info.header_offset)
                head = fh.read(30)
                if len(head) < 30:
                    raise StreamThinViewError(
                        "NPZ_HEADER", f"truncated local header {info.filename}")
                fields = struct.unpack("<IHHHHHIIIHH", head)
                fname_len, extra_len = fields[9], fields[10]
                data_off = info.header_offset + 30 + fname_len + extra_len
                name = info.filename
                if not name.endswith(".npy"):
                    raise StreamThinViewError(
                        "NPZ_MEMBER", f"unexpected member {name!r}")
                dtype, shape, arr_off = _read_npy_header(fh, data_off)
                self._offsets[name[:-4]] = (dtype, shape, arr_off)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._offsets))

    def column(self, name: str) -> np.ndarray:
        """Read-only memmap of member ``name`` (cached handle)."""
        arr = self._cache.get(name)
        if arr is None:
            entry = self._offsets.get(name)
            if entry is None:
                raise StreamThinViewError(
                    "NPZ_MEMBER", f"no member {name!r} in {self.npz_path}")
            dtype, shape, offset = entry
            arr = np.memmap(self.npz_path, dtype=dtype, mode="r",
                            offset=offset, shape=shape)
            self._cache[name] = arr
        return arr

    def release(self, name: str) -> None:
        arr = self._cache.pop(name, None)
        if arr is not None:
            del arr


# ---------------------------------------------------------------------------
# compact sampler view (v1 per-item arm)
# ---------------------------------------------------------------------------
class _DirStub:
    __slots__ = ("direction",)

    def __init__(self, direction: str) -> None:
        self.direction = direction


_FWD_STUB = _DirStub("forward")
_REV_STUB = _DirStub("reverse")


class CompactRecord:
    """Sampler-only record: exactly the attributes the v1 per-item path
    reads (``curve_id`` / ``fold`` / ``intent.direction``)."""

    __slots__ = ("curve_id", "fold", "intent")

    def __init__(self, curve_id: str, fold: int | None,
                 direction: str) -> None:
        self.curve_id = curve_id
        self.fold = fold
        self.intent = _FWD_STUB if direction == "forward" else _REV_STUB


class CompactCurveDirView(Sequence):
    """Per-position (curve_id, direction, fold) sequence over the compact
    columns — the v1 sampler reference arm of the streaming certificate.

    Carries NO ``_ds`` attribute, so ``_startup_of`` returns ``None`` and
    the trainer's ORIGINAL per-item code paths run (byte-identical draw
    semantics by construction).  Records are interned per
    ``(curve_idx, direction, fold)`` triple (≤ #curves × 2 × #folds).
    """

    def __init__(self, curve_ids: Sequence[str], curve_idx: np.ndarray,
                 direction: np.ndarray, fold: np.ndarray) -> None:
        self._curve_ids = list(curve_ids)
        self._curve_idx = curve_idx
        self._direction = direction
        self._fold = fold
        self._cache: dict[tuple[int, int, int], CompactRecord] = {}

    def __len__(self) -> int:
        return int(self._curve_idx.shape[0])

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[int(i)] for i in
                    range(*index.indices(len(self)))]
        i = int(index)
        ci = int(self._curve_idx[i])
        d = int(self._direction[i])
        f = int(self._fold[i])
        key = (ci, d, f)
        rec = self._cache.get(key)
        if rec is None:
            rec = CompactRecord(
                self._curve_ids[ci], (None if f < 0 else f),
                "forward" if d == 0 else "reverse")
            self._cache[key] = rec
        return rec

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class StreamColumns:
    """Compact per-position columns of the v1 survival stream."""

    __slots__ = ("n_rows", "curve_idx", "node_index", "intent_idx",
                 "direction", "fold", "weight_curves_checked", "elapsed_s")

    def __init__(self, n_rows: int, curve_idx: np.ndarray,
                 node_index: np.ndarray, intent_idx: np.ndarray,
                 direction: np.ndarray, fold: np.ndarray,
                 weight_curves_checked: int) -> None:
        self.n_rows = n_rows
        self.curve_idx = curve_idx
        self.node_index = node_index
        self.intent_idx = intent_idx
        self.direction = direction
        self.fold = fold
        self.weight_curves_checked = weight_curves_checked


class StreamThinViewV3:
    """Bounded v1-reference view (never materialises the sample list).

    Carries the attribute surface of ``ShardedKeyAtomDatasetV3`` so the v1
    methods assigned on the class body below run unbound, verbatim.
    """

    # --- verbatim v1 machinery (assigned, never copied) ---------------------
    _check_header_only = ShardedKeyAtomDatasetV3._check_header_only
    _load_shard_index = ShardedKeyAtomDatasetV3._load_shard_index
    _preverify_shards = ShardedKeyAtomDatasetV3._preverify_shards
    _load_curve_frames_shard = ShardedKeyAtomDatasetV3._load_curve_frames_shard
    _load_curve_frames_csv = ShardedKeyAtomDatasetV3._load_curve_frames_csv
    _load_curve_frames = ShardedKeyAtomDatasetV3._load_curve_frames
    _ensure_curve = ShardedKeyAtomDatasetV3._ensure_curve
    to_sample_v3 = ShardedKeyAtomDatasetV3.to_sample_v3
    _parse_sample_lite = ShardedKeyAtomDatasetV3._parse_sample_lite

    def __init__(
        self,
        tables_dir: str | Path,
        *,
        d0: float = D0_PRIMARY_ANGSTROM,
        label_statuses: Iterable[str] = ("ok", "terminal", "masked",
                                         "conflict"),
        directions: Iterable[str] = ("forward", "reverse"),
        cache_curves: int = 64,
        split_manifest: str | Path | None = None,
        fold: int | None = None,
        verify_manifest: bool = False,
    ) -> None:
        t0 = time.perf_counter()
        self.tables_dir = Path(tables_dir)
        self.d0 = float(d0)
        self.fold = fold
        self.label_statuses = frozenset(label_statuses)
        self.directions = frozenset(directions)
        self.cache_curves = int(cache_curves)
        self.verify_manifest = bool(verify_manifest)
        self._split_manifest_input = split_manifest
        _stream_require(self.tables_dir.is_dir(), "TABLES_DIR",
                        f"{self.tables_dir} is not a directory")
        for table, filename in TABLE_FILENAMES.items():
            _stream_require((self.tables_dir / filename).is_file(),
                            "TABLE_MISSING", f"{table} -> {filename} not found")

        # --- light tables (identical to v1 init) ----------------------------
        curve_rows = _read_csv(self.tables_dir / TABLE_FILENAMES["irc_curve"])
        _check_columns("irc_curve", curve_rows)
        self._curves: dict[str, dict[str, str]] = {
            r["curve_id"]: r for r in curve_rows}
        _require(len(self._curves) == len(curve_rows), "CURVE_ID_DUP",
                 "duplicate curve_id in irc_curve")

        intent_rows = _read_csv(
            self.tables_dir / TABLE_FILENAMES["reaction_intent"])
        _check_columns("reaction_intent", intent_rows)
        self._intents: dict[tuple[str, str], IntentRecordV3] = {}
        # intent row order = first appearance in reaction_intent.csv (the
        # idx builder's ordering; asserted by the columns check itself)
        self._intent_row_table: dict[tuple[str, str], int] = {}
        for row in intent_rows:
            try:
                intent = IntentRecordV3.from_table_row(row)
            except (KeyAtomV3ContractError, ValueError,
                    json.JSONDecodeError) as exc:
                raise KeyAtomDatasetV3Error(
                    "INTENT_PARSE", f"{row.get('intent_id')}: {exc}") from exc
            key = (intent.curve_id, intent.intent_id)
            _require(key not in self._intents, "INTENT_DUP",
                     f"duplicate (curve_id, intent_id) {key}")
            _require(intent.curve_id in self._curves, "INTENT_FK",
                     f"intent {intent.intent_id} references unknown curve")
            self._intents[key] = intent
            if key not in self._intent_row_table:
                self._intent_row_table[key] = len(self._intent_row_table)

        self._check_header_only("irc_edge")
        self._check_header_only("curve_quality")

        self._folds = _load_folds_at(self.tables_dir, split_manifest, fold)

        # shard fast path (geometry for the feature probes)
        self._shard_curve_path: dict[str, str] = {}
        self._shard_sha: dict[str, str] = {}
        self._shard_verified: set[str] = set()
        self.shard_verify_seconds = 0.0
        self._load_shard_index()
        self._preverify_shards()

        # --- irc_node light pass: node sets + atomic (NO energies) ----------
        self._node_index: dict[str, set[int]] = {}
        self._curve_atomic: dict[str, tuple[int, ...]] = {}
        self._scan_irc_node_light_stream()

        # --- intent involution (T0.2, verbatim) -----------------------------
        for (curve_id, intent_id), intent in self._intents.items():
            if intent.intent_underdetermined:
                continue
            rev = self._intents.get((curve_id, intent.reverse_of_intent_id))
            _require(rev is not None, "INTENT_REVERSE_MISSING",
                     f"intent {intent_id} has no reverse row "
                     f"{intent.reverse_of_intent_id!r}")
            _require(rev.reverse_of_intent_id == intent_id,
                     "INTENT_INVOLUTION",
                     f"reverse(reverse({intent_id})) != {intent_id}")

        # geometry LRU (empty at init)
        self._frames: "OrderedDict[str, dict[int, np.ndarray]]" = OrderedDict()
        self.init_seconds = round(time.perf_counter() - t0, 1)

    # -- light irc_node pass --------------------------------------------------
    def _scan_irc_node_light_stream(self) -> None:
        """One streaming pass keeping node sets + per-curve atomic numbers.

        Mirrors ``ShardedKeyAtomDatasetV3._scan_irc_node_light`` minus the
        energy collection (not consumed by this certificate).
        """
        path = self.tables_dir / TABLE_FILENAMES["irc_node"]
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            _require_columns_in("irc_node", header)
            col = {name: i for i, name in enumerate(header or ())}
            i_cid, i_node = col["curve_id"], col["node_index"]
            i_atom = col["atomic_numbers"]
            for raw in reader:
                cid = raw[i_cid]
                node = int(raw[i_node])
                nodes = self._node_index.setdefault(cid, set())
                _require(node not in nodes, "NODE_DUP",
                         f"duplicate node row ({cid}, {node})")
                nodes.add(node)
                payload = raw[i_atom] if len(raw) > i_atom else ""
                if payload and cid not in self._curve_atomic:
                    try:
                        self._curve_atomic[cid] = tuple(
                            int(v) for v in json.loads(payload))
                    except (ValueError, json.JSONDecodeError) as exc:
                        raise KeyAtomDatasetV3Error(
                            "ATOMIC_NUMBERS_PARSE",
                            f"{cid}: {exc}") from exc

    # -- sample_view pass 1: compact columns ----------------------------------
    def stream_columns(self) -> StreamColumns:
        """Pass 1 over ``sample_view.csv``: compact numpy columns only.

        Applies the v1 survival filters (primary d0, label_status, fold,
        direction via the intent table), enforces the full-group
        curve-weight invariant (Σ ``curve_weight`` == 1.0 per curve over
        the FULL primary-d0 group, accumulated BEFORE status filtering —
        verbatim v1 semantics) and the per-row node FK / intent / atomic
        checks (typed refusals, never skips).
        """
        t0 = time.perf_counter()
        path = self.tables_dir / TABLE_FILENAMES["sample_view"]
        from array import array
        c_idx = array("i")                        # int32 curve row
        n_idx = array("i")                        # int32 node index
        i_idx = array("i")                        # int32 intent row
        dirn = array("b")                         # int8 direction
        folds = array("b")                        # int8 fold
        weight_totals: dict[str, float] = {}
        curve_row = {cid: i for i, cid in enumerate(self._curves)}
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            _require_columns_in("sample_view", header)
            col = {name: i for i, name in enumerate(header or ())}
            i_d0 = col["d0"]
            i_cw = col["curve_weight"]
            i_status = col["label_status"]
            i_cid = col["curve_id"]
            i_node = col["node_index"]
            i_iid = col["intent_id"]
            i_sid = col["sample_id"]
            i_fold = col.get("fold")
            for raw in reader:
                try:
                    row_d0 = float(raw[i_d0])
                    cw = float(raw[i_cw])
                except (ValueError, IndexError) as exc:
                    raise KeyAtomDatasetV3Error(
                        "SAMPLE_WEIGHT_PARSE",
                        f"{raw[i_sid] if len(raw) > i_sid else '?'}: "
                        f"{exc}") from exc
                if abs(row_d0 - self.d0) <= 1e-12:
                    cid = raw[i_cid]
                    weight_totals[cid] = weight_totals.get(cid, 0.0) + cw
                # --- survival filter (v1 _parse_sample_lite semantics) -------
                if abs(row_d0 - self.d0) > 1e-12:
                    continue
                status = raw[i_status]
                if status not in self.label_statuses:
                    continue
                curve_id = raw[i_cid]
                node_index = int(raw[i_node])
                intent = self._intents.get((curve_id, raw[i_iid]))
                _require(intent is not None, "SAMPLE_INTENT_FK",
                         f"sample {raw[i_sid]} references unknown "
                         f"intent {(curve_id, raw[i_iid])}")
                _require(intent.direction in self.directions,
                         "SAMPLE_DIRECTION",
                         f"sample {raw[i_sid]} direction filtered")
                _require(node_index in self._node_index.get(curve_id, ()),
                         "SAMPLE_NODE_FK",
                         f"sample {raw[i_sid]} references unknown "
                         f"node ({curve_id}, {node_index})")
                atomic = self._curve_atomic.get(curve_id) or intent.g_atoms
                _require(bool(atomic), "ATOMIC_NUMBERS_MISSING",
                         f"curve {curve_id} has no atomic_numbers payload")
                if self.fold is not None:
                    curve_fold = self._folds.get(curve_id)
                    _require(curve_fold is not None, "SPLIT_CURVE_MISSING",
                             f"curve {curve_id} missing from split manifest")
                    if int(curve_fold) != int(self.fold):
                        continue
                fold_value = (raw[i_fold] or "").strip() if i_fold is not None \
                    else ""
                c_idx.append(curve_row[curve_id])
                n_idx.append(node_index)
                i_idx.append(self._intent_row(curve_id, raw[i_iid]))
                dirn.append(0 if intent.direction == "forward" else 1)
                folds.append(int(fold_value) if fold_value
                             else int(self._folds.get(curve_id, -1)))
        _require(len(c_idx) > 0, "NO_SAMPLES",
                 "no sample_view rows survive the dataset filters "
                 f"(d0={self.d0}, statuses={sorted(self.label_statuses)}, "
                 f"fold={self.fold})")
        for curve_id, total in weight_totals.items():
            _require(abs(total - 1.0) <= WEIGHT_SUM_TOL, "CURVE_WEIGHT_SUM",
                     f"curve {curve_id}: sum(curve_weight)={total!r} != 1.0")
        cols = StreamColumns(
            n_rows=len(c_idx),
            curve_idx=np.asarray(c_idx, dtype=np.int32),
            node_index=np.asarray(n_idx, dtype=np.int32),
            intent_idx=np.asarray(i_idx, dtype=np.int32),
            direction=np.asarray(dirn, dtype=np.int8),
            fold=np.asarray(folds, dtype=np.int8),
            weight_curves_checked=len(weight_totals),
        )
        cols.elapsed_s = round(time.perf_counter() - t0, 1)
        return cols

    def _intent_row(self, curve_id: str, intent_id: str) -> int:
        """Intent row = first-appearance order in reaction_intent.csv."""
        row = self._intent_row_table.get((curve_id, intent_id))
        if row is None:
            raise KeyAtomDatasetV3Error(
                "SAMPLE_INTENT_FK", f"unknown intent {(curve_id, intent_id)}")
        return row

    # -- sample_view pass 2: probe records ------------------------------------
    def probe_records(self, positions: Iterable[int],
                      columns: StreamColumns,
                      ) -> dict[int, SampleRecordV3Lite]:
        """Pass 2: full v1 records for PROBE positions only.

        Rows are parsed by the inherited v1 ``_parse_sample_lite`` (same
        typed refusals, same field parsing) — bit-identical records by
        construction.
        """
        wanted = sorted({int(p) for p in positions})
        _stream_require(
            all(0 <= p < columns.n_rows for p in wanted), "PROBE_RANGE",
            f"probe positions outside [0, {columns.n_rows})")
        want = set(wanted)
        out: dict[int, SampleRecordV3Lite] = {}
        path = self.tables_dir / TABLE_FILENAMES["sample_view"]
        pos = -1
        header: list[str] | None = None
        idx: dict[str, int] = {}
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            _require_columns_in("sample_view", header)
            idx = {name: i for i, name in enumerate(header or ())}
            for raw in reader:
                # survival filter first (cheap), full parse only for probes
                try:
                    row_d0 = float(raw[idx["d0"]])
                except ValueError as exc:
                    raise KeyAtomDatasetV3Error(
                        "SAMPLE_D0_PARSE",
                        f"{raw[idx['sample_id']]}: {exc}") from exc
                if abs(row_d0 - self.d0) > 1e-12:
                    continue
                status = str(raw[idx["label_status"]])
                if status not in self.label_statuses:
                    continue
                curve_id = raw[idx["curve_id"]]
                intent = self._intents.get((curve_id, raw[idx["intent_id"]]))
                if intent is None or intent.direction not in self.directions:
                    # v1 would refuse; stream_columns already verified this
                    # for every surviving row — keep the position aligned
                    raise KeyAtomDatasetV3Error(
                        "SAMPLE_INTENT_FK",
                        f"probe pass lost intent for {raw[idx['sample_id']]}")
                if self.fold is not None:
                    curve_fold = self._folds.get(curve_id)
                    if curve_fold is None or int(curve_fold) != int(self.fold):
                        continue
                pos += 1
                if pos in want:
                    row = {name: raw[i] for name, i in idx.items()}
                    rec = self._parse_sample_lite(row, self._folds)
                    _require(rec is not None, "PROBE_ALIGNMENT",
                             f"probe pass dropped row at position {pos} "
                             f"({row.get('sample_id')}) — survival filter "
                             "drift between pass 1 and pass 2")
                    out[pos] = rec
        missing = [p for p in wanted if p not in out]
        _stream_require(not missing, "PROBE_MISSING",
                        f"probe positions not materialised: {missing[:5]}")
        return out

    # -- sampler surface ------------------------------------------------------
    def compact_view(self, columns: StreamColumns) -> CompactCurveDirView:
        """The v1 per-item sampler arm (curve/direction/fold only)."""
        return CompactCurveDirView(
            list(self._curves.keys()), columns.curve_idx,
            columns.direction, columns.fold)

    # -- geometry / features --------------------------------------------------
    def frames_lru_size(self) -> int:
        return len(self._frames)
