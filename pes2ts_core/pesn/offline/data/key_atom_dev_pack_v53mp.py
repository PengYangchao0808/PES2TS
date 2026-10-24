"""V5.3-MP dev feature pack (S4 plan §5.3 / lane I2): frozen dev features,
pickle-free, one lazy mmap view shared by the model pass AND the R3 baseline
pass of the final eval.

The final dev eval currently builds ~136k F2 feature dicts per run (≈5.5 min
serial-equivalent wall + a pinned ≈4.7 GB LRU).  The pack builds them ONCE
from the frozen v2 loader, stores them as per-sample blobs in ONE flat
``uint8`` array + int64 offsets, and serves them lazily via
``np.load(..., mmap_mode="r")`` — at most ONE sample's bytes are
materialised per access, the pages are shared across the 9 runs of a
campaign, and both eval passes hit the same pack.

Files (under ``<tables_dir>/dev_packs/<features_sha256>/<positions_sha>/``)::

    blobs.npy     uint8  [total_blob_bytes]     flat per-sample blobs
    offsets.npy   int64  [n_positions + 1]      blob slice bounds
    positions.npy int32  [n_positions]          dataset positions (ascending)
    <positions_sha>.json                        binding block (see below)

DEVIATION from the task sketch (documented): the pack is a DIRECTORY of
flat ``.npy`` files, not a single ``.npz`` — a zip ``.npz`` member cannot be
memory-mapped, and reading one member materialises ALL of it (4.7 GB),
violating the "≤1 sample per access" requirement.  This mirrors the
accepted ``idx_v2/`` per-field-``.npy`` precedent.

Per-sample blob layout (little-endian)::

    [u32 header_len][header JSON (utf-8)][raw array bytes in header order]

The header lists every key of the BUILT (and ``_FeatureLRU._finalize``-
processed, i.e. ``y_teacher`` never None) sample dict from
``build_sample_features_v3`` with its type tag; arrays follow as raw
C-contiguous bytes.  Decoding copies exactly one sample (≈34 KB) and
returns a dict byte-identical to the live builder output — no pickle, no
eval, no dtype conversion.

Binding (``<positions_sha>.json``): ``features_sha256`` (sha256 of
``key_atom_graph_v3.py`` — the frozen feature builder; READ-ONLY),
``positions_sha256`` (sha256 over the newline-joined decimal position
ids in pack order — the exact rule, so builder/driver/trainer cannot
diverge), ``n_positions``, ``feature_level`` (``F2``), ``intent_off``
(``false``), dataset binding shas (data manifest + idx curves.json),
file shas/sizes, builder git commit, ``created_at``.  Opening a pack
re-verifies features_sha256 against the LIVE feature module and refuses
(typed :class:`DevPackError`) on ANY drift.

Torch-free by design (numpy + stdlib only) so the builder, the trainer,
the certificate and the tests all share ONE codec.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "DEV_PACKS_DIRNAME",
    "DEV_PACK_SCHEMA",
    "DevPackError",
    "FEATURES_MODULE_PATH",
    "blob_decode",
    "blob_encode",
    "dev_positions_from_split",
    "features_sha256_of",
    "open_dev_feature_pack",
    "positions_sha256",
    "write_dev_feature_pack",
]

#: Pack root inside the tables dir.
DEV_PACKS_DIRNAME = "dev_packs"
#: Pack schema id (binding json ``schema``).
DEV_PACK_SCHEMA = "v53mp_dev_pack_v1"
#: Migration-time sha256 cache sidecars (``scripts/infra/sha_sidecar.py``)
#: land NEXT to pack files as ``<target>.sha256sidecar.json``; they are NOT
#: binding jsons and are excluded from binding discovery (still exactly ONE
#: real binding required — fail-closed on anything else).
SIDECAR_JSON_SUFFIX = ".sha256sidecar.json"
#: Frozen feature module (sha-bound; NEVER edited by this lane).
FEATURES_MODULE_PATH = Path("pes2ts_core/pesn/features/key_atom_graph_v3.py")


class DevPackError(RuntimeError):
    """Typed refusal of the dev feature pack (stale / corrupt / mismatched)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise DevPackError(code, detail)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# binding helpers (single source of truth for builder / driver / trainer)
# ---------------------------------------------------------------------------
def features_sha256_of(features_module_path: Optional[Path] = None) -> str:
    """sha256 of the frozen ``key_atom_graph_v3.py`` (the features binding)."""
    path = (Path(features_module_path) if features_module_path is not None
            else FEATURES_MODULE_PATH)
    if not path.is_file():
        # allow an absolute path derived from the imported module
        import pes2ts_core.pesn.features.key_atom_graph_v3 as _mod
        path = Path(_mod.__file__)
        if path.suffix == ".pyc":
            path = path.with_suffix(".py")
    _require(path.is_file(), "FEATURES_MODULE_MISSING", str(path))
    return _sha256(path)


def positions_sha256(positions: Sequence[int]) -> str:
    """Binding sha over the position ids (EXACT rule, single implementation).

    ``sha256("\\n".join(str(int(p)) for p in positions))`` over the pack's
    positions in pack order (ascending dataset positions).  Builder, driver
    and trainer all call THIS function, so a derivation drift is impossible.
    """
    joined = "\n".join(str(int(p)) for p in positions)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def dev_positions_from_split(samples: Any,
                             dev_curve_ids: Iterable[str]) -> list[int]:
    """Frozen dev positions: enumerate order over ``samples`` (driver rule).

    The driver computes ``dev_positions = [i for i, s in enumerate(samples)
    if str(s.curve_id) in dev_curve_ids]``; this helper is that rule in one
    place so the pack builder cannot diverge from the run driver.

    FAST PATH: when the v2 columnar loader's ``samples`` sequence is
    passed, the curve column is read vectorized from the idx memmap (no
    per-row record construction) — measured 45.6 s → 0.3 s at the 20k
    tier (plan §7 B2 startup gate).
    """
    wanted = {str(c) for c in dev_curve_ids}
    ds = getattr(samples, "_ds", None)
    columns = getattr(ds, "_columns", None) if ds is not None else None
    if columns is not None and "curve_idx" in columns:
        # columnar fast path: map wanted curve ids to idx rows, then
        # vectorized membership over the memmapped curve column
        curve_row = getattr(ds, "_curve_row", None)
        wanted_rows = np.asarray(
            sorted(curve_row[c] for c in wanted if c in curve_row),
            dtype=np.int64)
        if wanted_rows.size == 0:
            return []
        col = np.asarray(columns["curve_idx"])
        positions = np.searchsorted(wanted_rows, col)
        positions = np.minimum(positions, wanted_rows.size - 1)
        mask = wanted_rows[positions] == col
        return np.nonzero(mask)[0].tolist()
    return [i for i, rec in enumerate(samples)
            if str(rec.curve_id) in wanted]


# ---------------------------------------------------------------------------
# blob codec (pickle-free; one sample in, one dict out)
# ---------------------------------------------------------------------------
_HEADER_LEN = struct.Struct("<I")


def _encode_value(value: Any, arrays: list) -> Any:
    """Map one built-dict value to a JSON-safe header entry (+array buffer)."""
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        index = len(arrays)
        arrays.append(arr)
        return {"t": "arr", "i": index, "dtype": arr.dtype.str,
                "shape": [int(s) for s in arr.shape]}
    if isinstance(value, (np.float32, np.float64)):
        return {"t": "f", "dtype": value.dtype.str, "v": float(value)}
    if isinstance(value, np.generic):
        return {"t": "f", "dtype": value.dtype.str, "v": float(value)}
    if isinstance(value, str):
        return {"t": "s", "v": value}
    if isinstance(value, bool):
        return {"t": "b", "v": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"t": "i", "v": int(value)}
    if isinstance(value, float):
        # plain Python float (distinct from np.float64 scalars — lossless)
        return {"t": "pfloat", "v": value}
    if value is None:
        return {"t": "null"}
    if isinstance(value, tuple):
        # endpoint_names is tuple-of-tuples; plain tuple-of-str also lands here
        return {"t": "tup", "v": [_encode_value(v, arrays) for v in value]}
    raise DevPackError("BLOB_VALUE_UNSUPPORTED",
                       f"cannot encode value of type {type(value).__name__}")


def blob_encode(built: Mapping[str, Any]) -> bytes:
    """Encode ONE built feature dict into a self-describing blob.

    Lossless for every value kind ``build_sample_features_v3`` produces
    (arrays, np.float32/64 scalars, strings, bools, ints, None, nested
    tuples).  The encoded dict must be POST-finalize (``y_teacher`` never
    None) — the builder applies the same zero-pad the dev LRU applies.
    """
    arrays: list = []
    header: Dict[str, Any] = {}
    for key in sorted(built):
        header[key] = _encode_value(built[key], arrays)
    raw = json.dumps(header, sort_keys=True, separators=(",", ":"))
    head_bytes = raw.encode("utf-8")
    body = b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)
    return _HEADER_LEN.pack(len(head_bytes)) + head_bytes + body


def _decode_value(entry: Mapping[str, Any], body: bytes,
                  cursor: list) -> Any:
    kind = entry["t"]
    if kind == "arr":
        dt = np.dtype(entry["dtype"])
        shape = tuple(int(s) for s in entry["shape"])
        count = int(np.prod(shape)) if shape else 1
        nbytes = count * dt.itemsize
        start = cursor[0]
        cursor[0] = start + nbytes
        # one-sample copy: writable, detached from the mmap after return
        arr = np.frombuffer(body[start:cursor[0]], dtype=dt).reshape(
            shape).copy()
        return arr
    if kind == "f":
        return np.dtype(entry["dtype"]).type(entry["v"])
    if kind == "pfloat":
        return float(entry["v"])
    if kind == "s":
        return str(entry["v"])
    if kind == "b":
        return bool(entry["v"])
    if kind == "i":
        return int(entry["v"])
    if kind == "null":
        return None
    if kind == "tup":
        return tuple(_decode_value(v, body, cursor) for v in entry["v"])
    raise DevPackError("BLOB_VALUE_UNSUPPORTED", f"unknown tag {kind!r}")


def blob_decode(blob: bytes) -> Dict[str, Any]:
    """Decode ONE blob back to the built feature dict (byte-identical)."""
    _require(len(blob) >= _HEADER_LEN.size, "BLOB_TRUNCATED",
             f"blob of {len(blob)} bytes is shorter than the header prefix")
    (header_len,) = _HEADER_LEN.unpack_from(blob, 0)
    start = _HEADER_LEN.size
    end = start + int(header_len)
    _require(end <= len(blob), "BLOB_TRUNCATED",
             f"header claims {header_len} bytes, blob has {len(blob) - start}")
    header = json.loads(blob[start:end].decode("utf-8"))
    _require(isinstance(header, dict), "BLOB_HEADER_INVALID",
             "header is not a JSON object")
    body = blob[end:]
    cursor = [0]
    out: Dict[str, Any] = {}
    for key, entry in header.items():
        out[key] = _decode_value(entry, body, cursor)
    _require(cursor[0] == len(body), "BLOB_BODY_LENGTH",
             f"header describes {cursor[0]} body bytes, blob has {len(body)}")
    return out


# ---------------------------------------------------------------------------
# pack writer (builder entry point)
# ---------------------------------------------------------------------------
def write_dev_feature_pack(
    out_dir: Path,
    positions: Sequence[int],
    blobs: Iterable[bytes],
    *,
    feature_level: str,
    intent_off: bool,
    dataset_binding: Mapping[str, Any],
    git_commit: str = "",
) -> Dict[str, Any]:
    """Persist one pack (blobs streamed to disk) + binding json; returns it.

    ``blobs`` must yield exactly ``len(positions)`` encoded blobs in the
    same order as ``positions`` (ascending).  Memory-bounded: blobs are
    appended to a temp raw file, then copied into the final uint8 ``.npy``
    in 64 MiB chunks (never a full in-RAM concatenation).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = [int(p) for p in positions]
    _require(bool(pos), "PACK_EMPTY", "no positions to pack")
    _require(all(pos[i] < pos[i + 1] for i in range(len(pos) - 1)),
             "PACK_POSITIONS_ORDER", "positions must be strictly ascending")
    tmp = out_dir / ".blobs.raw.tmp"
    offsets = np.zeros(len(pos) + 1, dtype=np.int64)
    total = 0
    n = 0
    with tmp.open("wb") as fh:
        for blob in blobs:
            _require(n < len(pos), "PACK_COUNT",
                     "more blobs than positions")
            fh.write(blob)
            total += len(blob)
            offsets[n + 1] = total
            n += 1
    _require(n == len(pos), "PACK_COUNT",
             f"got {n} blobs for {len(pos)} positions")
    # final .npy: chunked copy from the temp raw file (bounded memory)
    blobs_path = out_dir / "blobs.npy"
    mm = np.lib.format.open_memmap(
        blobs_path, mode="w+", dtype=np.uint8, shape=(total,))
    with tmp.open("rb") as fh:
        written = 0
        chunk = 64 << 20
        while written < total:
            data = fh.read(min(chunk, total - written))
            mm[written:written + len(data)] = np.frombuffer(
                data, dtype=np.uint8)
            written += len(data)
    mm.flush()
    del mm
    tmp.unlink()
    np.save(out_dir / "offsets.npy", offsets, allow_pickle=False)
    np.save(out_dir / "positions.npy",
            np.asarray(pos, dtype=np.int32), allow_pickle=False)
    fsha = features_sha256_of()
    psha = positions_sha256(pos)
    binding = {
        "schema": DEV_PACK_SCHEMA,
        "features_sha256": fsha,
        "positions_sha256": psha,
        "n_positions": len(pos),
        "feature_level": str(feature_level),
        "intent_off": bool(intent_off),
        "dataset_binding": dict(dataset_binding),
        "files": {
            "blobs.npy": {"sha256": _sha256(blobs_path), "bytes": total},
            "offsets.npy": {
                "sha256": _sha256(out_dir / "offsets.npy"),
                "bytes": int((out_dir / "offsets.npy").stat().st_size)},
            "positions.npy": {
                "sha256": _sha256(out_dir / "positions.npy"),
                "bytes": int((out_dir / "positions.npy").stat().st_size)},
        },
        "builder_git_commit": str(git_commit),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    binding_path = out_dir / f"{psha}.json"
    binding_path.write_text(
        json.dumps(binding, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    return binding


# ---------------------------------------------------------------------------
# pack reader (lazy view; the trainer's cache protocol is cache[position])
# ---------------------------------------------------------------------------
class DevFeaturePackView:
    """``cache[position] -> built feature dict`` over a memmapped pack.

    Implements the exact cache protocol ``evaluate_v53mp`` /
    ``_batch_from_cache`` use (``cache[int(p)]``).  Every access copies
    exactly ONE sample's blob (≈34 KB) and decodes it; nothing is cached
    (the eval iterates each position once per pass) so memory stays flat.
    ``positions`` are resolved with ``np.searchsorted`` — zero index memory.
    """

    def __init__(self, pack_dir: Path, binding: Mapping[str, Any],
                 *, verify_files: bool = False) -> None:
        self.pack_dir = Path(pack_dir)
        self.binding = dict(binding)
        self._blobs = np.load(self.pack_dir / "blobs.npy",
                              mmap_mode="r", allow_pickle=False)
        self._offsets = np.load(self.pack_dir / "offsets.npy",
                                mmap_mode="r", allow_pickle=False)
        self._positions = np.load(self.pack_dir / "positions.npy",
                                  mmap_mode="r", allow_pickle=False)
        _require(self._blobs.ndim == 1 and self._blobs.dtype == np.uint8,
                 "PACK_SHAPE", "blobs.npy must be a 1-D uint8 array")
        _require(self._offsets.shape == (len(self._positions) + 1,),
                 "PACK_SHAPE", "offsets.npy length must be n_positions + 1")
        if verify_files:
            for name, meta in self.binding.get("files", {}).items():
                actual = _sha256(self.pack_dir / name)
                _require(actual == str(meta["sha256"]), "PACK_SHA",
                         f"{name}: sha256 {actual[:12]}… != binding "
                         f"{str(meta['sha256'])[:12]}…")

    @classmethod
    def open(cls, pack_dir: Path, *,
             expected_positions: Optional[Sequence[int]] = None,
             feature_level: Optional[str] = None,
             verify_files: bool = False) -> "DevFeaturePackView":
        """Open + fail-closed verify a pack directory.

        Refuses on: missing binding file, schema drift, LIVE features_sha256
        mismatch (the feature builder changed after the pack was built),
        ``expected_positions`` not covered by the pack, feature-level
        mismatch.
        """
        pack_dir = Path(pack_dir)
        fsha = features_sha256_of()
        # migration ``*.sha256sidecar.json`` caches are NOT binding jsons
        candidates = sorted(
            path for path in pack_dir.glob("*.json")
            if not path.name.endswith(SIDECAR_JSON_SUFFIX)
        )
        _require(len(candidates) == 1, "PACK_BINDING_MISSING",
                 f"{pack_dir}: expected exactly one binding json, found "
                 f"{[p.name for p in candidates]}")
        binding = json.loads(candidates[0].read_text(encoding="utf-8"))
        _require(binding.get("schema") == DEV_PACK_SCHEMA, "PACK_SCHEMA",
                 f"binding schema {binding.get('schema')!r} != "
                 f"{DEV_PACK_SCHEMA!r}")
        _require(str(binding.get("features_sha256")) == fsha,
                 "PACK_FEATURES_DRIFT",
                 "pack was built for features_sha256 "
                 f"{str(binding.get('features_sha256'))[:12]}… but the LIVE "
                 f"key_atom_graph_v3.py hashes to {fsha[:12]}… — rebuild the "
                 "pack (feature builder changed)")
        _require(str(binding.get("positions_sha256"))
                 == candidates[0].stem, "PACK_BINDING_NAME",
                 "binding filename does not match its positions_sha256")
        if feature_level is not None:
            _require(str(binding.get("feature_level"))
                     == str(feature_level), "PACK_LEVEL",
                     f"pack level {binding.get('feature_level')!r} != "
                     f"{feature_level!r}")
        view = cls(pack_dir, binding, verify_files=verify_files)
        if expected_positions is not None:
            want = np.asarray([int(p) for p in expected_positions],
                              dtype=np.int64)
            got = view._positions.astype(np.int64)
            missing = np.setdiff1d(want, got, assume_unique=False)
            if missing.size != 0:
                raise DevPackError(
                    "PACK_POSITIONS",
                    "pack does not cover {} expected positions "
                    "(first: {})".format(missing.size, int(missing[0])))
        return view

    @property
    def positions(self) -> np.ndarray:
        """Pack positions (ascending int32 view)."""
        return self._positions

    @property
    def n_positions(self) -> int:
        return int(self._positions.shape[0])

    def _row_of(self, position: int) -> int:
        p = int(position)
        row = int(np.searchsorted(self._positions, p))
        _require(row < self.n_positions
                 and int(self._positions[row]) == p, "PACK_POSITION",
                 f"position {p} is not in the pack")
        return row

    def __getitem__(self, position: int) -> Dict[str, Any]:
        row = self._row_of(position)
        start = int(self._offsets[row])
        end = int(self._offsets[row + 1])
        blob = self._blobs[start:end].tobytes()
        return blob_decode(blob)

    def __len__(self) -> int:
        return self.n_positions


def open_dev_feature_pack(pack_dir: Path, **kwargs: Any) -> DevFeaturePackView:
    """Alias for :meth:`DevFeaturePackView.open` (trainer-facing name)."""
    return DevFeaturePackView.open(Path(pack_dir), **kwargs)
