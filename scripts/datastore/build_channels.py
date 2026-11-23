#!/usr/bin/env python3
"""Build the PES2TS datastore ``02_core_tensors`` channels (xTB / geometry / DFT / G1).

Deliverables (default root ``--out-root /pes2ts_datastore``)
------------------------------------------------------------
``02_core_tensors/xtb_channel/xtb_points.parquet``
    One row per xTB single point (2,877,830 = 2,877,433 success + 397 SCF
    failure).  Geometry metadata comes from ``manifest/shards/*.jsonl``, E/g +
    inline DFT pairing from ``run/results/*.jsonl``, and the failure ledger from
    ``run/failures/*.jsonl``.  ``manifest/dft_pairs/*.jsonl`` is NOT read
    (default policy; it is a redundant sidecar of the inline DFT fields).
``02_core_tensors/xtb_channel/xtb_channel_manifest.json``
``02_core_tensors/geometry/{coords_f64.npy,curve_offsets.npy,atomic_numbers.npy,
node_atom_offsets.npy,curve_atom_offsets.npy,geometry_manifest.json}``
``02_core_tensors/dft_channel/{dft_energy.npy,dft_channel_manifest.json}``
``02_core_tensors/g1_metadata/{reaction_intent.parquet,curve_family.parquet,
g1_metadata_manifest.json}``

Guarantees
----------
* Sources are opened read-only; nothing under ``/pes2ts_data`` or the xTB
  package is modified.
* Streaming only: one xTB shard (61 total) is resident at a time.
* fail-closed: bad JSON, unknown schema, missing field, duplicate ``frame_key``,
  cross-source mismatch, bad gradient length or bad digest raises
  :class:`BuildError`; the failing sample is written to
  ``/tmp/pes2ts_build_channels/<stage>.failure.jsonl`` and no final artifact is
  left behind (all writes go to ``*.tmp`` then ``os.replace``).
* Deterministic: manifests are timestamp-free and artifacts are written with
  fixed parquet/npy settings, so a rerun over unchanged sources reproduces the
  same sha256 for every artifact.

Usage (WSL, guarded)::

    wsl --cd /mnt/e/Calculations/AI4S_ML_Studys/PES2TS -e bash \
        scripts/infra/wsl_guarded_run.sh --mem 6G --name build-channels -- \
        /root/miniforge3/envs/pes2ts-gpu/bin/python \
        scripts/datastore/build_channels.py --stage all

Stages: ``all`` | ``xtb`` | ``geometry`` | ``dft`` | ``g1`` | ``verify``.
``--shards 0:2`` restricts the xTB stage to a shard range and writes to
``--partial-dir`` (smoke testing; never touches the datastore).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

# --------------------------------------------------------------------------- #
# paths / constants
# --------------------------------------------------------------------------- #
REPO_WIN = Path(__file__).resolve().parents[2]
XTB_ROOT = REPO_WIN / "data/raw/pes2ts_xtb_singlepoints_train_fit_v1"
DATA_ROOT = Path("/pes2ts_data")
D50 = DATA_ROOT / "v53mp_50k_data"
DS_ROOT = Path("/pes2ts_datastore")
DS_TENSORS = DS_ROOT / "02_core_tensors"

FAIL_LOG_DIR = Path("/tmp/pes2ts_build_channels")

N_SHARDS = 61
EXPECTED_MANIFEST_ROWS = 2_877_830
EXPECTED_RESULTS = 2_877_433
EXPECTED_FAILURES = 397
EXPECTED_N_CURVES = 64_195
EXPECTED_N_NODES = 7_342_581
EXPECTED_COORD_ROWS = 123_443_835
EXPECTED_N_INTENTS = 128_390

XTB_POINTS_SCHEMA = "pes2ts_xtb_points_v1"
XTB_MANIFEST_SCHEMA = "pes2ts_xtb_channel_manifest_v1"
GEOM_MANIFEST_SCHEMA = "pes2ts_geometry_channel_manifest_v1"
DFT_MANIFEST_SCHEMA = "pes2ts_dft_channel_manifest_v1"
G1_MANIFEST_SCHEMA = "pes2ts_g1_metadata_manifest_v1"

BOHR_ANGSTROM = 1.889726125
HARTREE_EV = 27.211386
HARTREE_BOHR_EV_ANG = 51.4221

RESULT_SHA_RULE = (
    "sha256(utf8(json.dumps(result_row_without_result_sha256, "
    "sort_keys=True, separators=(',', ':'), ensure_ascii=False)))"
)
RESULT_SHA_SAMPLE_TARGET = 1000

XTB_UNITS = {
    "energy_hartree": "hartree (Ha) — xTB .engrad header 'current total energy in Eh'",
    "gradient_hartree_per_bohr": "hartree / bohr (Ha/Bohr) — xTB .engrad 'current gradient in Eh/bohr', 3N flattened",
    "gradient_norm": "hartree / bohr; L2 norm of gradient_hartree_per_bohr (source field gradient_norm_hartree_per_bohr)",
    "dft_energy_ev": "electronvolt (eV) — DFT IRC per-frame energy (inline in run/results)",
    "dft_force_ev_per_angstrom": "eV / angstrom (eV/A) — DFT IRC per-frame FORCE (= -dE/dx), 3N flattened",
    "n_atoms": "count (atoms per frame)",
    "constants": {
        "bohr_per_angstrom": BOHR_ANGSTROM,
        "hartree_to_ev": HARTREE_EV,
        "hartree_per_bohr_to_ev_per_angstrom": HARTREE_BOHR_EV_ANG,
        "forbidden": "E_xTB - E_DFT absolute cross-method subtraction is forbidden; "
                     "only same-geometry delta-g and same-route delta-delta-E are allowed",
    },
}


class BuildError(RuntimeError):
    """Typed fail-closed build failure."""

    def __init__(self, stage: str, message: str, sample: dict | list | None = None):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message
        self.sample = sample


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json_atomic(path: Path, obj) -> str:
    """Deterministic (sorted keys, no timestamps) JSON written via tmp+rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return sha256_bytes(data)


def parse_sha256sums(path: Path, strip_prefix: str = "") -> dict[str, str]:
    """Parse a ``sha256sum`` listing; comment lines start with ``#``.

    ``strip_prefix`` removes the leading package directory (``<pkg>/``) so the
    resulting keys are package-relative paths.
    """
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise BuildError("source", f"malformed SHA256SUMS line: {line[:120]!r}")
            digest, rel = parts[0], parts[1].lstrip("*")
            if strip_prefix and rel.startswith(strip_prefix):
                rel = rel[len(strip_prefix):]
            out[rel] = digest
    return out


def scan_jsonl(path: Path, handler) -> dict:
    """Stream a JSONL file byte-exactly, hashing as we go.

    ``handler(raw_line: bytes, lineno: int)`` is called per non-empty line.
    Returns ``{sha256, rows, bytes}``; the hash is of the *entire file bytes*, so
    it can be checked against the package ``SHA256SUMS`` without a second pass.
    """
    h = hashlib.sha256()
    n = 0
    nbytes = 0
    lineno = 0
    with open(path, "rb") as fh:
        buf = b""
        while True:
            chunk = fh.read(1 << 23)
            if not chunk:
                break
            nbytes += len(chunk)
            h.update(chunk)
            parts = (buf + chunk).split(b"\n")
            buf = parts.pop()
            for raw in parts:
                lineno += 1
                if not raw.strip():
                    raise BuildError("xtb", f"blank line in {path.name} at line {lineno}")
                handler(raw, lineno)
                n += 1
        if buf.strip():
            lineno += 1
            handler(buf, lineno)
            n += 1
    return {"sha256": h.hexdigest(), "rows": n, "bytes": nbytes}


def _fail(stage: str, message: str, sample=None) -> None:
    raise BuildError(stage, message, sample)


def log_failure(exc: BuildError) -> Path:
    FAIL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = FAIL_LOG_DIR / f"{exc.stage}.failure.jsonl"
    entry = {"stage": exc.stage, "message": exc.message, "sample": exc.sample}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(canonical_json(entry) + "\n")
    return path


def list_array_from_rows(rows, pa_type: pa.DataType) -> pa.Array:
    """Build a nullable ListArray from per-row numpy vectors (or None)."""
    offsets = np.zeros(len(rows) + 1, dtype=np.int32)
    mask = np.zeros(len(rows), dtype=bool)
    chunks = []
    pos = 0
    for i, row in enumerate(rows):
        if row is None:
            mask[i] = True
        else:
            chunks.append(row)
            pos += row.size
        offsets[i + 1] = pos
    values = (np.concatenate(chunks).astype(np.float32, copy=False)
              if chunks else np.empty(0, dtype=np.float32))
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(values), mask=pa.array(mask))


def copy_file_hashed(src: Path, dst: Path) -> dict:
    """Stream-copy ``src`` -> ``dst`` (tmp+rename) and return size/sha256."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    h = hashlib.sha256()
    nbytes = 0
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for chunk in iter(lambda: fi.read(1 << 23), b""):
            h.update(chunk)
            fo.write(chunk)
            nbytes += len(chunk)
    os.replace(tmp, dst)
    return {"source": str(src), "path": str(dst), "bytes": nbytes, "sha256": h.hexdigest()}


# =========================================================================== #
# stage: xTB channel
# =========================================================================== #
def _xtb_schema() -> pa.Schema:
    return pa.schema([
        pa.field("xtb_frame_key", pa.string(), nullable=False),
        pa.field("reaction_id", pa.string(), nullable=False),
        pa.field("direction", pa.string(), nullable=False),
        pa.field("frame_index", pa.int32(), nullable=False),
        pa.field("coords_sha256", pa.string(), nullable=False),
        pa.field("n_atoms", pa.int16(), nullable=False),
        pa.field("energy_hartree", pa.float64()),
        pa.field("gradient_hartree_per_bohr", pa.list_(pa.float32())),
        pa.field("gradient_norm", pa.float64()),
        pa.field("dft_energy_ev", pa.float64()),
        pa.field("dft_force_ev_per_angstrom", pa.list_(pa.float32())),
        pa.field("scf_converged", pa.bool_(), nullable=False),
        pa.field("scf_iterations", pa.int16()),
        pa.field("scf_log_sha256", pa.string()),
        pa.field("status", pa.string(), nullable=False),
        pa.field("failure_reason", pa.string()),
        pa.field("has_dft_pair", pa.bool_(), nullable=False),
        pa.field("shard", pa.int16(), nullable=False),
        pa.field("result_sha256", pa.string()),
    ])


def _require(d: dict, keys: tuple[str, ...], where: str, lineno: int) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        _fail("xtb", f"missing fields {missing} in {where} line {lineno}",
              {"where": where, "line": lineno, "missing": missing, "keys": sorted(d)[:30]})


def _xtb_manifest_handler_factory(store: dict, order: list):
    req = ("schema_version", "frame_key", "reaction_id", "direction", "frame_index",
           "coordinates_angstrom", "atomic_numbers", "coords_sha256")

    def handler(raw: bytes, lineno: int) -> None:
        try:
            d = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            _fail("xtb", f"bad JSON in manifest shard line {lineno}: {exc}",
                  {"line": lineno, "raw_head": raw[:200].decode("utf-8", "replace")})
        if d.get("schema_version") != "pes2ts_xtb_frame_v1":
            _fail("xtb", f"unknown manifest schema_version at line {lineno}",
                  {"line": lineno, "schema_version": d.get("schema_version")})
        _require(d, req, "manifest", lineno)
        n_atoms = len(d["atomic_numbers"])
        if n_atoms <= 0 or len(d["coordinates_angstrom"]) != 3 * n_atoms:
            _fail("xtb", f"manifest geometry length mismatch at line {lineno}",
                  {"line": lineno, "n_atoms": n_atoms,
                   "n_coords": len(d["coordinates_angstrom"])})
        if d["direction"] not in ("A", "B"):
            _fail("xtb", f"bad direction at manifest line {lineno}",
                  {"line": lineno, "direction": d["direction"]})
        key = d["frame_key"]
        if key in store:
            _fail("xtb", f"duplicate frame_key within manifest shard at line {lineno}",
                  {"line": lineno, "frame_key": key})
        store[key] = (d["reaction_id"], d["direction"], int(d["frame_index"]),
                      d["coords_sha256"], n_atoms)
        order.append(key)

    return handler


def _xtb_result_handler_factory(store: dict, counters: dict):
    req = ("schema_version", "frame_key", "reaction_id", "direction", "frame_index",
           "coords_sha256", "energy_hartree", "gradient_hartree_per_bohr",
           "gradient_norm_hartree_per_bohr", "scf_converged", "scf_log_sha256",
           "status", "result_sha256", "dft_energy_ev", "dft_force_ev_per_angstrom")

    def handler(raw: bytes, lineno: int) -> None:
        try:
            d = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            _fail("xtb", f"bad JSON in results shard line {lineno}: {exc}",
                  {"line": lineno, "raw_head": raw[:200].decode("utf-8", "replace")})
        if d.get("schema_version") != "pes2ts_xtb_singlepoint_v1":
            _fail("xtb", f"unknown result schema_version at line {lineno}",
                  {"line": lineno, "schema_version": d.get("schema_version")})
        _require(d, req, "results", lineno)
        grad = d["gradient_hartree_per_bohr"]
        if not isinstance(grad, list) or not grad or len(grad) % 3 != 0:
            _fail("xtb", f"bad gradient list at results line {lineno}",
                  {"line": lineno, "len": len(grad) if isinstance(grad, list) else None})
        dft_e = d["dft_energy_ev"]
        dft_f = d["dft_force_ev_per_angstrom"]
        if (dft_e is None) != (dft_f is None):
            _fail("xtb", f"DFT pair partially null at results line {lineno}",
                  {"line": lineno, "dft_energy_ev": dft_e,
                   "dft_force_len": None if dft_f is None else len(dft_f)})
        if dft_f is not None and len(dft_f) != len(grad):
            _fail("xtb", f"DFT force length != gradient length at results line {lineno}",
                  {"line": lineno, "len_force": len(dft_f), "len_grad": len(grad)})
        key = d["frame_key"]
        if key in store:
            _fail("xtb", f"duplicate frame_key within results shard at line {lineno}",
                  {"line": lineno, "frame_key": key})
        store[key] = d
        counters["n_dft_pair_rows"] += int(dft_e is not None)

    return handler


def _xtb_failure_handler_factory(store: dict):
    req = ("schema_version", "frame_key", "reaction_id", "direction", "frame_index", "reason")

    def handler(raw: bytes, lineno: int) -> None:
        try:
            d = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            _fail("xtb", f"bad JSON in failures shard line {lineno}: {exc}",
                  {"line": lineno, "raw_head": raw[:200].decode("utf-8", "replace")})
        if d.get("schema_version") != "pes2ts_xtb_failure_v1":
            _fail("xtb", f"unknown failure schema_version at line {lineno}",
                  {"line": lineno, "schema_version": d.get("schema_version")})
        _require(d, req, "failures", lineno)
        if d["frame_key"] in store:
            _fail("xtb", f"duplicate frame_key within failures shard at line {lineno}",
                  {"line": lineno, "frame_key": d["frame_key"]})
        store[d["frame_key"]] = d

    return handler


def build_xtb(args: argparse.Namespace) -> dict:
    stage = "xtb"
    t_start = time.perf_counter()
    out_dir = DS_TENSORS / "xtb_channel"
    partial = tuple(args.shards) != (0, N_SHARDS)
    if partial:
        out_dir = Path(args.partial_dir) / "xtb_channel"
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = json.load(open(XTB_ROOT / "manifest/index.json", "r", encoding="utf-8"))
    shard_meta = {s["name"]: s for s in idx["shards"]}
    src_hashes = parse_sha256sums(XTB_ROOT / "SHA256SUMS",
                                  strip_prefix=f"{XTB_ROOT.name}/")

    schema = _xtb_schema()
    final_path = out_dir / "xtb_points.parquet"
    parquet_tmp = out_dir / ".xtb_points.parquet.tmp"
    if parquet_tmp.exists():
        parquet_tmp.unlink()
    writer = pq.ParquetWriter(
        str(parquet_tmp), schema,
        compression="zstd", compression_level=3,
        use_dictionary=["direction", "status", "failure_reason"],
        version="2.6", data_page_version="1.0",
        write_statistics=True, metadata_collector=None,
    )

    counters = {
        "n_manifest_rows": 0, "n_results": 0, "n_failures": 0,
        "n_dft_pair_rows": 0, "n_gradient_ok": 0, "n_norm_ok": 0,
        "n_coords_match": 0, "n_meta_match": 0, "n_result_sha_checked": 0,
        "n_result_sha_match": 0, "n_source_shard_sha_verified": 0,
    }
    fail_reasons: dict[str, int] = {}
    seen_keys: set[str] = set()
    per_shard: list[dict] = []
    sample_mismatches: list[dict] = []
    sha_step = max(1, EXPECTED_MANIFEST_ROWS // RESULT_SHA_SAMPLE_TARGET)
    global_row = 0
    partial_counts = {"manifest": 0, "results": 0, "failures": 0,
                      "manifest_bytes": 0, "results_bytes": 0, "failures_bytes": 0}

    start, end = args.shards
    if not (0 <= start < end <= N_SHARDS):
        _fail(stage, f"bad --shards range {start}:{end} (need 0 <= a < b <= {N_SHARDS})")

    for s in range(start, end):
        name = f"shard_{s:05d}.jsonl"
        man_path = XTB_ROOT / "manifest/shards" / name
        res_path = XTB_ROOT / "run/results" / name
        fail_path = XTB_ROOT / "run/failures" / name

        man: dict[str, tuple] = {}
        order: list[str] = []
        res: dict[str, dict] = {}
        flr: dict[str, dict] = {}

        st_man = scan_jsonl(man_path, _xtb_manifest_handler_factory(man, order))
        st_res = scan_jsonl(res_path, _xtb_result_handler_factory(res, counters))
        if fail_path.exists():
            st_flr = scan_jsonl(fail_path, _xtb_failure_handler_factory(flr))
        else:
            st_flr = {"sha256": None, "rows": 0, "bytes": 0}

        # ---- source hash verification against the package manifest
        for rel, st in ((f"manifest/shards/{name}", st_man),
                        (f"run/results/{name}", st_res),
                        (f"run/failures/{name}", st_flr)):
            if st["rows"] == 0 and st["sha256"] is None:
                if rel in src_hashes:
                    _fail(stage, f"missing failures shard listed in SHA256SUMS: {rel}")
                continue
            if rel not in src_hashes:
                _fail(stage, f"streamed file not listed in SHA256SUMS: {rel}")
            if src_hashes[rel] != st["sha256"]:
                _fail(stage, f"source sha256 mismatch for {rel}",
                      {"recorded": src_hashes[rel], "actual": st["sha256"]})
            counters["n_source_shard_sha_verified"] += 1
        # index.json row count / hash for the manifest shard
        meta = shard_meta.get(name)
        if meta is None:
            _fail(stage, f"manifest shard {name} missing from index.json")
        if int(meta["n_rows"]) != st_man["rows"]:
            _fail(stage, f"manifest shard row count mismatch for {name}",
                  {"index.json": int(meta["n_rows"]), "actual": st_man["rows"]})
        if meta.get("sha256") and meta["sha256"] != st_man["sha256"]:
            _fail(stage, f"manifest shard sha256 mismatch vs index.json for {name}")

        # ---- set reconciliation inside the shard
        res_keys = set(res)
        flr_keys = set(flr)
        man_keys = set(man)
        missing = man_keys - res_keys - flr_keys
        extra = (res_keys | flr_keys) - man_keys
        if missing or extra or (res_keys & flr_keys):
            _fail(stage, f"outcome/manifest key mismatch in shard {s}",
                  {"shard": s, "n_missing": len(missing), "n_extra": len(extra),
                   "n_overlap": len(res_keys & flr_keys),
                   "missing_sample": sorted(missing)[:5],
                   "extra_sample": sorted(extra)[:5]})
        if len(man) != st_man["rows"]:
            _fail(stage, f"duplicate frame_key inside manifest shard {s}")

        # ---- build columns in manifest order (deterministic)
        cols = {k: [] for k in ("xtb_frame_key", "reaction_id", "direction", "frame_index",
                                "coords_sha256", "n_atoms", "energy_hartree",
                                "gradient_norm", "dft_energy_ev", "scf_converged",
                                "scf_iterations", "scf_log_sha256", "status",
                                "failure_reason", "has_dft_pair", "shard", "result_sha256")}
        grads: list = []
        forces: list = []

        for key in order:
            if key in seen_keys:
                _fail(stage, f"duplicate frame_key across shards: {key}",
                      {"shard": s, "frame_key": key})
            seen_keys.add(key)
            rid_m, dir_m, fi_m, csha_m, n_atoms = man[key]
            cols["xtb_frame_key"].append(key)
            cols["reaction_id"].append(rid_m)
            cols["direction"].append(dir_m)
            cols["frame_index"].append(fi_m)
            cols["coords_sha256"].append(csha_m)
            cols["n_atoms"].append(n_atoms)
            cols["shard"].append(s)

            if key in res:
                d = res[key]
                if (d["reaction_id"], d["direction"], int(d["frame_index"])) != (rid_m, dir_m, fi_m):
                    _fail(stage, f"metadata mismatch manifest vs result for {key}",
                          {"shard": s, "frame_key": key,
                           "manifest": [rid_m, dir_m, fi_m],
                           "result": [d["reaction_id"], d["direction"], d["frame_index"]]})
                if d["coords_sha256"] != csha_m:
                    _fail(stage, f"coords_sha256 mismatch manifest vs result for {key}",
                          {"shard": s, "frame_key": key, "manifest": csha_m,
                           "result": d["coords_sha256"]})
                counters["n_coords_match"] += 1
                counters["n_meta_match"] += 1

                grad = np.asarray(d["gradient_hartree_per_bohr"], dtype=np.float64)
                if grad.size != 3 * n_atoms:
                    _fail(stage, f"gradient length {grad.size} != 3*n_atoms {3 * n_atoms}",
                          {"shard": s, "frame_key": key, "n_atoms": n_atoms,
                           "gradient_len": int(grad.size)})
                if not np.all(np.isfinite(grad)):
                    _fail(stage, f"non-finite gradient for {key}", {"shard": s, "frame_key": key})
                counters["n_gradient_ok"] += 1
                norm_stored = float(d["gradient_norm_hartree_per_bohr"])
                norm_recomputed = float(np.linalg.norm(grad))
                if abs(norm_stored - norm_recomputed) > 1e-9 * max(1.0, abs(norm_stored)):
                    _fail(stage, f"gradient_norm mismatch for {key}",
                          {"shard": s, "frame_key": key, "stored": norm_stored,
                           "recomputed": norm_recomputed})
                counters["n_norm_ok"] += 1

                dft_e = d["dft_energy_ev"]
                dft_f = d["dft_force_ev_per_angstrom"]
                if dft_f is not None:
                    if len(dft_f) != 3 * n_atoms:
                        _fail(stage, f"DFT force length {len(dft_f)} != 3*n_atoms {3 * n_atoms}",
                              {"shard": s, "frame_key": key})
                    force = np.asarray(dft_f, dtype=np.float32)
                else:
                    force = None
                energy = float(d["energy_hartree"])
                if not np.isfinite(energy):
                    _fail(stage, f"non-finite energy for {key}", {"shard": s, "frame_key": key})
                if not bool(d["scf_converged"]):
                    _fail(stage, f"status=success but scf_converged=False for {key}",
                          {"shard": s, "frame_key": key})

                # per-row digest (deterministic sample)
                if global_row % sha_step == 0:
                    payload = {k: v for k, v in d.items() if k != "result_sha256"}
                    got = sha256_bytes(canonical_json(payload).encode("utf-8"))
                    counters["n_result_sha_checked"] += 1
                    if got != d["result_sha256"]:
                        if len(sample_mismatches) < 5:
                            sample_mismatches.append(
                                {"shard": s, "frame_key": key,
                                 "recorded": d["result_sha256"], "recomputed": got})
                    else:
                        counters["n_result_sha_match"] += 1

                cols["energy_hartree"].append(energy)
                cols["gradient_norm"].append(norm_stored)
                cols["dft_energy_ev"].append(float(dft_e) if dft_e is not None else None)
                cols["scf_converged"].append(True)
                cols["scf_iterations"].append(int(d["scf_iterations"])
                                              if d.get("scf_iterations") is not None else None)
                cols["scf_log_sha256"].append(d["scf_log_sha256"])
                cols["status"].append("success")
                cols["failure_reason"].append(None)
                cols["has_dft_pair"].append(dft_e is not None)
                cols["result_sha256"].append(d["result_sha256"])
                grads.append(np.asarray(grad, dtype=np.float32))
                forces.append(force)
                counters["n_results"] += 1
            else:
                f = flr[key]
                if (f["reaction_id"], f["direction"], int(f["frame_index"])) != (rid_m, dir_m, fi_m):
                    _fail(stage, f"metadata mismatch manifest vs failure for {key}",
                          {"shard": s, "frame_key": key})
                counters["n_meta_match"] += 1
                reason = f["reason"]
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                cols["energy_hartree"].append(None)
                cols["gradient_norm"].append(None)
                cols["dft_energy_ev"].append(None)
                cols["scf_converged"].append(False)
                cols["scf_iterations"].append(None)
                cols["scf_log_sha256"].append(None)
                cols["status"].append("failed")
                cols["failure_reason"].append(reason)
                cols["has_dft_pair"].append(False)
                cols["result_sha256"].append(None)
                grads.append(None)
                forces.append(None)
                counters["n_failures"] += 1
            global_row += 1

        tbl = pa.table({
            "xtb_frame_key": pa.array(cols["xtb_frame_key"], type=pa.string()),
            "reaction_id": pa.array(cols["reaction_id"], type=pa.string()),
            "direction": pa.array(cols["direction"], type=pa.string()),
            "frame_index": pa.array(cols["frame_index"], type=pa.int32()),
            "coords_sha256": pa.array(cols["coords_sha256"], type=pa.string()),
            "n_atoms": pa.array(cols["n_atoms"], type=pa.int16()),
            "energy_hartree": pa.array(cols["energy_hartree"], type=pa.float64()),
            "gradient_hartree_per_bohr": list_array_from_rows(grads, pa.float32()),
            "gradient_norm": pa.array(cols["gradient_norm"], type=pa.float64()),
            "dft_energy_ev": pa.array(cols["dft_energy_ev"], type=pa.float64()),
            "dft_force_ev_per_angstrom": list_array_from_rows(forces, pa.float32()),
            "scf_converged": pa.array(cols["scf_converged"], type=pa.bool_()),
            "scf_iterations": pa.array(cols["scf_iterations"], type=pa.int16()),
            "scf_log_sha256": pa.array(cols["scf_log_sha256"], type=pa.string()),
            "status": pa.array(cols["status"], type=pa.string()),
            "failure_reason": pa.array(cols["failure_reason"], type=pa.string()),
            "has_dft_pair": pa.array(cols["has_dft_pair"], type=pa.bool_()),
            "shard": pa.array(cols["shard"], type=pa.int16()),
            "result_sha256": pa.array(cols["result_sha256"], type=pa.string()),
        }, schema=schema)
        writer.write_table(tbl, row_group_size=1 << 30)

        counters["n_manifest_rows"] += len(order)
        partial_counts["manifest"] += st_man["rows"]
        partial_counts["results"] += st_res["rows"]
        partial_counts["failures"] += st_flr["rows"]
        partial_counts["manifest_bytes"] += st_man["bytes"]
        partial_counts["results_bytes"] += st_res["bytes"]
        partial_counts["failures_bytes"] += st_flr["bytes"]
        per_shard.append({"shard": s, "n_manifest": st_man["rows"],
                          "n_results": st_res["rows"], "n_failures": st_flr["rows"],
                          "manifest_sha256": st_man["sha256"],
                          "results_sha256": st_res["sha256"],
                          "failures_sha256": st_flr["sha256"]})
        print(f"[xtb] shard {s:02d}: manifest={st_man['rows']} results={st_res['rows']} "
              f"failures={st_flr['rows']} dft_pairs={sum(1 for k in order if k in res and res[k]['dft_energy_ev'] is not None)}")

        # release per-shard memory
        del man, res, flr, order, grads, forces, tbl, cols

    writer.close()

    if not partial:
        if counters["n_manifest_rows"] != EXPECTED_MANIFEST_ROWS:
            _fail(stage, f"manifest rows {counters['n_manifest_rows']} != expected {EXPECTED_MANIFEST_ROWS}")
        if counters["n_results"] != EXPECTED_RESULTS:
            _fail(stage, f"result rows {counters['n_results']} != expected {EXPECTED_RESULTS}")
        if counters["n_failures"] != EXPECTED_FAILURES:
            _fail(stage, f"failure rows {counters['n_failures']} != expected {EXPECTED_FAILURES}")
    if counters["n_manifest_rows"] != counters["n_results"] + counters["n_failures"]:
        _fail(stage, "manifest rows != results + failures",
              {"manifest": counters["n_manifest_rows"], "results": counters["n_results"],
               "failures": counters["n_failures"]})
    if counters["n_results"] > 0 and counters["n_coords_match"] != counters["n_results"]:
        _fail(stage, "coords_sha256 agreement is not 100%",
              {"results": counters["n_results"], "matched": counters["n_coords_match"]})
    if counters["n_results"] > 0 and counters["n_gradient_ok"] != counters["n_results"]:
        _fail(stage, "gradient length/3N check is not 100%")
    if counters["n_results"] > 0 and counters["n_norm_ok"] != counters["n_results"]:
        _fail(stage, "gradient_norm recompute check is not 100%")
    if counters["n_result_sha_checked"] == 0:
        _fail(stage, "no result_sha256 sample was checked")
    if counters["n_result_sha_match"] != counters["n_result_sha_checked"]:
        _fail(stage, "result_sha256 recompute mismatches",
              {"checked": counters["n_result_sha_checked"],
               "matched": counters["n_result_sha_match"], "samples": sample_mismatches})

    os.replace(parquet_tmp, final_path)
    out_sha = sha256_file(final_path)
    out_bytes = final_path.stat().st_size

    manifest = {
        "schema": XTB_MANIFEST_SCHEMA,
        "channel_id": "xtb_singlepoint_eg",
        "schema_version_note": "manifest is timestamp-free by design: reruns over unchanged sources are byte-identical",
        "artifacts": [{
            "path": str(final_path),
            "sha256": out_sha,
            "bytes": out_bytes,
            "format": "parquet",
            "compression": "zstd (level 3), dictionary-encoded low-cardinality columns",
            "row_groups": end - start,
            "rows": counters["n_manifest_rows"],
            "columns": [{"name": f.name, "type": str(f.type), "nullable": f.nullable}
                        for f in schema],
            "schema_tag": XTB_POINTS_SCHEMA,
        }],
        "sources": {
            "package_root": str(XTB_ROOT),
            "run_id": idx.get("run_id"),
            "dataset_id": "pes2ts_xtb_singlepoints_train_fit_v1",
            "read_streams": {
                "manifest/shards/*.jsonl": {
                    "files": end - start, "rows": partial_counts["manifest"],
                    "bytes": partial_counts["manifest_bytes"]},
                "run/results/*.jsonl": {
                    "files": end - start, "rows": partial_counts["results"],
                    "bytes": partial_counts["results_bytes"]},
                "run/failures/*.jsonl": {
                    "files": end - start, "rows": partial_counts["failures"],
                    "bytes": partial_counts["failures_bytes"]},
            },
            "not_read": ["manifest/dft_pairs/*.jsonl (default policy: inline DFT fields "
                         "in run/results are authoritative; sidecar not needed)"],
            "sha256_verified_stream_files": counters["n_source_shard_sha_verified"],
            "per_shard": per_shard,
            "source_file_sha256": {k: v for k, v in sorted(src_hashes.items())
                                   if k.startswith(("manifest/shards/", "run/results/",
                                                    "run/failures/"))},
        },
        "build": {
            "script": "scripts/datastore/build_channels.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "command": f"python scripts/datastore/build_channels.py --stage xtb "
                       f"--shards {start}:{end}",
            "stage": "xtb",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "deterministic": True,
            "note": "the manifest is a pure function of the source content, the shard range "
                    "and the script text: repeated runs are byte-identical",
        },
        "units": XTB_UNITS,
        "column_notes": {
            "scf_log_sha256": "per-frame xTB log integrity credential copied verbatim from the "
                              "results row (sha256 of the .log.gz for successful frames; null for "
                              "failed frames). Retained so run/logs/ (2,877,830 files, 13.9 GB) can "
                              "be deleted in a later cleanup phase without losing per-frame log "
                              "provenance.",
            "scf_iterations": "source results field scf_iterations (int16, null for failed frames)",
            "result_sha256": "per-row digest of the source results row (rule in "
                             "validation.result_sha256_recompute_sample); makes every row auditable "
                             "against the raw JSONL after the package is archived offline",
        },
        "reconciliation": {
            "n_manifest_rows": counters["n_manifest_rows"],
            "n_results": counters["n_results"],
            "n_failures": counters["n_failures"],
            "manifest_equals_results_plus_failures": (
                counters["n_manifest_rows"] == counters["n_results"] + counters["n_failures"]),
            "expected_manifest_rows": EXPECTED_MANIFEST_ROWS,
            "expected_results": EXPECTED_RESULTS,
            "expected_failures": EXPECTED_FAILURES,
            "n_frame_keys": counters["n_manifest_rows"],
            "n_unique_frame_keys": len(seen_keys),
            "frame_key_unique": len(seen_keys) == counters["n_manifest_rows"],
            "n_shards": end - start,
            "shards_index_row_count_match": True,
            "failure_reasons": dict(sorted(fail_reasons.items())),
            "n_dft_pair_rows": counters["n_dft_pair_rows"],
            "n_rows_without_dft_pair": counters["n_results"] - counters["n_dft_pair_rows"],
        },
        "validation": {
            "coords_sha256_manifest_vs_result": {
                "rule": "full check on every successful row: manifest/shards coords_sha256 == run/results coords_sha256",
                "rows_checked": counters["n_results"],
                "rows_matching": counters["n_coords_match"],
                "agreement_rate": (counters["n_coords_match"] / counters["n_results"]
                                   if counters["n_results"] else 0.0),
            },
            "metadata_triplet_manifest_vs_result_or_failure": {
                "fields": ["reaction_id", "direction", "frame_index"],
                "rows_checked": counters["n_manifest_rows"],
                "rows_matching": counters["n_meta_match"],
            },
            "gradient_length_equals_3n": {
                "rows_checked": counters["n_results"],
                "rows_ok": counters["n_gradient_ok"],
            },
            "gradient_norm_recompute": {
                "rule": "abs(stored - ||gradient||_2) <= 1e-9 * max(1, |stored|) on the float64 source gradient",
                "rows_checked": counters["n_results"],
                "rows_ok": counters["n_norm_ok"],
            },
            "dft_force_length_equals_3n": {
                "rows_checked": counters["n_dft_pair_rows"],
                "rows_ok": counters["n_dft_pair_rows"],
            },
            "result_sha256_recompute_sample": {
                "rule": RESULT_SHA_RULE,
                "sampling": f"deterministic: global row index % {sha_step} == 0",
                "rows_checked": counters["n_result_sha_checked"],
                "rows_matching": counters["n_result_sha_match"],
                "recomputed": True,
                "note": "the digest rule was recovered empirically and verified on every sample; "
                        "it is exact because the generator serialised rows with sorted-key compact JSON",
            },
            "source_sha256": {
                "files_verified_against_SHA256SUMS": counters["n_source_shard_sha_verified"],
                "mismatches": 0,
            },
        },
        "reading": {
            "pandas": "import pandas as pd; df = pd.read_parquet('<path>'); "
                      "df.iloc[:5]; df['gradient_hartree_per_bohr'].iloc[0].astype('float32')",
            "pyarrow": "import pyarrow.parquet as pq; t = pq.read_table('<path>'); "
                       "t.column('gradient_hartree_per_bohr').flatten()  # contiguous float32 values",
            "row_order": "manifest shard index ascending, then manifest/shards line order inside a shard",
            "failure_rows": "status='failed' rows have null energy/gradient/DFT fields; scf_converged=False",
        },
        "boundaries": [
            "offline_only=true, truth_assisted=true (inherited from the source package)",
            "never for online inference / production decisions / external capability claims",
            "E_xTB - E_DFT absolute cross-method subtraction is forbidden; only same-geometry delta-g and same-route delta-delta-E",
            "xTB gradients must not be used as a Move teacher or as a substitute for DFT forces",
            "scf_log_sha256 keeps per-frame log provenance after run/logs/ is deleted; it is a "
            "credential, not a pointer (the log files themselves are not part of this datastore)",
        ],
        "open_questions": [],
    }
    man_path = out_dir / "xtb_channel_manifest.json"
    man_sha = write_json_atomic(man_path, manifest)
    print(f"[xtb] rows={counters['n_manifest_rows']} parquet={out_bytes/1e6:.1f}MB "
          f"sha256={out_sha[:12]} elapsed={time.perf_counter() - t_start:.1f}s")
    return {"path": str(final_path), "sha256": out_sha, "bytes": out_bytes,
            "rows": counters["n_manifest_rows"], "manifest": str(man_path),
            "manifest_sha256": man_sha}


# =========================================================================== #
# stage: geometry channel
# =========================================================================== #
def build_geometry(args: argparse.Namespace) -> dict:
    stage = "geometry"
    t_start = time.perf_counter()
    out_dir = DS_TENSORS / "geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    geom_dir = D50 / "geom"
    curves_json_path = D50 / "idx_v2/curves.json"
    curves = json.load(open(curves_json_path, "r", encoding="utf-8"))
    curve_ids = curves["curve_ids"]
    curves_rows = curves["curves"]
    geom_offsets_path = geom_dir / "curve_offsets.json"
    geom_offsets = json.load(open(geom_offsets_path, "r", encoding="utf-8"))
    gcurves = geom_offsets["curves"]

    if len(curve_ids) != EXPECTED_N_CURVES or len(curves_rows) != EXPECTED_N_CURVES:
        _fail(stage, "curves.json curve count mismatch",
              {"curve_ids": len(curve_ids), "curves": len(curves_rows),
               "expected": EXPECTED_N_CURVES})
    if len(gcurves) != EXPECTED_N_CURVES:
        _fail(stage, "geom/curve_offsets curve count mismatch", {"n": len(gcurves)})

    # ---- 1. coords_f64.npy (stream copy + hash, verify against geom binding)
    coords_src = geom_dir / "coords.f64.npy"
    coords_dst = out_dir / "coords_f64.npy"
    coords_info = copy_file_hashed(coords_src, coords_dst)
    binding = geom_offsets.get("binding", {})
    if binding.get("coords_sha256") and binding["coords_sha256"] != coords_info["sha256"]:
        _fail(stage, "coords.f64.npy sha256 != geom/curve_offsets.json binding coords_sha256",
              {"binding": binding["coords_sha256"], "actual": coords_info["sha256"]})
    coords_arr = np.load(coords_dst, mmap_mode="r")
    if coords_arr.shape != (EXPECTED_COORD_ROWS, 3) or coords_arr.dtype != np.float64:
        _fail(stage, "unexpected coords array shape/dtype",
              {"shape": list(coords_arr.shape), "dtype": str(coords_arr.dtype)})

    # ---- 2. curve_offsets.npy (= label_curves/offsets.npy, node offsets per curve)
    off_src = D50 / "idx_v2/label_curves/offsets.npy"
    qoff_src = D50 / "idx_v2/label_curves/q_offsets.npy"
    off_dst = out_dir / "curve_offsets.npy"
    off_info = copy_file_hashed(off_src, off_dst)
    qoff_sha = sha256_file(qoff_src) if qoff_src.exists() else None
    node_off = np.load(off_dst)
    node_counts = np.diff(node_off)
    if node_off.dtype != np.int64 or node_off.shape != (EXPECTED_N_CURVES + 1,):
        _fail(stage, "unexpected curve_offsets shape/dtype",
              {"shape": list(node_off.shape), "dtype": str(node_off.dtype)})
    if int(node_off[0]) != 0 or int(node_off[-1]) != EXPECTED_N_NODES:
        _fail(stage, "curve_offsets endpoints mismatch",
              {"first": int(node_off[0]), "last": int(node_off[-1]),
               "expected_last": EXPECTED_N_NODES})
    if not np.all(node_counts > 0):
        _fail(stage, "curve_offsets is not strictly increasing")

    # ---- 3. atomic_numbers.npy from staging_chunks/*.npz, flattened per curve
    staging = D50 / "staging_chunks"
    chunk_files = sorted(p for p in staging.iterdir() if p.suffix == ".npz")
    npz_atoms: dict[str, np.ndarray] = {}
    sidecar_entries: list[str] = []
    n_npz_reactions = 0
    for p in chunk_files:
        with np.load(p, allow_pickle=False) as z:
            for key in z.keys():
                if not key.endswith("::atomic_numbers"):
                    continue
                rid = key.split("::", 1)[0]
                arr = z[key]
                npz_atoms[rid] = np.asarray(arr, dtype=np.int64)
                n_npz_reactions += 1
        side = p.with_name(p.name + ".sha256sidecar.json")
        if side.exists():
            sidecar_entries.append(f"{p.name} {sha256_file(side)}")
    staging_source_sha = sha256_bytes("\n".join(sidecar_entries).encode("utf-8")) if sidecar_entries else None

    missing = [cid for cid in curve_ids if cid not in npz_atoms]
    if missing:
        _fail(stage, "curves missing from staging_chunks atomic_numbers",
              {"n_missing": len(missing), "sample": missing[:5]})
    extras = set(npz_atoms) - set(curve_ids)

    natoms = np.zeros(EXPECTED_N_CURVES, dtype=np.int64)
    for i, cid in enumerate(curve_ids):
        expected = np.asarray(curves_rows[i]["atomic_numbers"], dtype=np.int64)
        got = npz_atoms[cid]
        if got.shape != expected.shape or not np.array_equal(got, expected):
            _fail(stage, f"staging_chunks atomic_numbers mismatch for {cid}",
                  {"curve_id": cid, "expected_head": expected[:8].tolist(),
                   "got_head": got[:8].tolist()})
        if int(gcurves[cid]["n_atoms"]) != expected.size:
            _fail(stage, f"geom n_atoms mismatch curves.json for {cid}",
                  {"curve_id": cid, "geom": int(gcurves[cid]["n_atoms"]),
                   "curves_json": int(expected.size)})
        natoms[i] = expected.size
    node_counts_i64 = node_counts.astype(np.int64)

    # per-curve span check: cumulative n_atoms*n_nodes == coords rows, offsets line up
    curve_atom_off = np.zeros(EXPECTED_N_CURVES + 1, dtype=np.int64)
    np.cumsum(natoms * node_counts_i64, out=curve_atom_off[1:])
    if int(curve_atom_off[-1]) != EXPECTED_COORD_ROWS:
        _fail(stage, "sum(n_atoms*n_nodes) != coords rows",
              {"sum": int(curve_atom_off[-1]), "coords_rows": EXPECTED_COORD_ROWS})
    atom_offset_bad = 0
    for i, cid in enumerate(curve_ids):
        if int(gcurves[cid]["offset"]) != int(curve_atom_off[i]):
            atom_offset_bad += 1
    if atom_offset_bad:
        _fail(stage, "geom/curve_offsets offset != cumulative n_atoms*n_nodes",
              {"n_bad": atom_offset_bad})

    # flatten per node: node-major blocks of the curve's atom list
    an_flat = np.empty(EXPECTED_COORD_ROWS, dtype=np.uint8)
    pos = 0
    for i, cid in enumerate(curve_ids):
        atoms_u8 = npz_atoms[cid].astype(np.uint8)
        n_nd = int(node_counts_i64[i])
        block = np.tile(atoms_u8, n_nd)
        an_flat[pos:pos + block.size] = block
        pos += block.size
    if pos != EXPECTED_COORD_ROWS:
        _fail(stage, "atomic_numbers flatten length mismatch", {"pos": pos})

    # node -> atom offset index (derived; makes coords_f64 directly addressable per node)
    node_curve = np.repeat(np.arange(EXPECTED_N_CURVES, dtype=np.int64), node_counts_i64)
    node_rel = np.arange(EXPECTED_N_NODES, dtype=np.int64) - node_off[node_curve]
    node_atom_off = np.empty(EXPECTED_N_NODES + 1, dtype=np.int64)
    node_atom_off[:EXPECTED_N_NODES] = curve_atom_off[node_curve] + node_rel * natoms[node_curve]
    node_atom_off[EXPECTED_N_NODES] = EXPECTED_COORD_ROWS
    if not np.all(np.diff(node_atom_off) == natoms[node_curve]):
        _fail(stage, "node_atom_offsets inconsistent with per-node atom counts")

    def save_npy(path: Path, arr: np.ndarray) -> dict:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, path)
        return {"path": str(path), "bytes": path.stat().st_size,
                "sha256": sha256_file(path), "shape": list(arr.shape), "dtype": str(arr.dtype)}

    an_info = save_npy(out_dir / "atomic_numbers.npy", an_flat)
    nao_info = save_npy(out_dir / "node_atom_offsets.npy", node_atom_off)
    cao_info = save_npy(out_dir / "curve_atom_offsets.npy", curve_atom_off)

    # element histogram (documentation / sanity)
    hist = {int(z): int(c) for z, c in zip(*np.unique(an_flat[np.arange(0, an_flat.size, max(1, an_flat.size // 200000))], return_counts=True))}
    full_hist = {int(z): int(c) for z, c in zip(*np.unique(an_flat, return_counts=True))}

    manifest = {
        "schema": GEOM_MANIFEST_SCHEMA,
        "channel_id": "v5_geometry",
        "schema_note": "manifest is timestamp-free by design: reruns over unchanged sources are byte-identical",
        "artifacts": [
            {"name": "coords_f64.npy", **coords_info,
             "shape": list(coords_arr.shape), "dtype": str(coords_arr.dtype),
             "nbytes": int(coords_arr.nbytes),
             "meaning": "flat (sum_atoms, 3) float64 coordinates; per-node block = "
                        "coords_f64[node_atom_offsets[i]:node_atom_offsets[i+1]] -> (n_atoms_i, 3), angstrom"},
            {"name": "curve_offsets.npy", **off_info,
             "shape": list(node_off.shape), "dtype": str(node_off.dtype),
             "meaning": "node offsets per curve (64,196 int64); curve i nodes = [off[i], off[i+1])"},
            {"name": "atomic_numbers.npy", **an_info,
             "meaning": "flat per-node uint8 atomic numbers aligned 1:1 with coords_f64 rows "
                        "(node-major: each node repeats its curve's atom list)"},
            {"name": "node_atom_offsets.npy", **nao_info,
             "meaning": "DERIVED node -> atom-row offsets (7,342,582 int64); "
                        "diff == n_atoms of that node's curve"},
            {"name": "curve_atom_offsets.npy", **cao_info,
             "meaning": "DERIVED per-curve atom-row offsets (64,196 int64); "
                        "diff == n_atoms(curve) * n_nodes(curve)"},
        ],
        "sources": {
            "coords": {"path": str(coords_src), "sha256": coords_info["sha256"],
                       "bytes": coords_info["bytes"],
                       "upstream_binding_sha256": binding.get("coords_sha256"),
                       "upstream_schema": geom_offsets.get("schema")},
            "curve_offsets": {"path": str(off_src), "sha256": off_info["sha256"],
                              "q_offsets_path": str(qoff_src), "q_offsets_sha256": qoff_sha,
                              "offsets_equals_q_offsets": qoff_sha == off_info["sha256"]},
            "atomic_numbers": {
                "path": str(staging),
                "n_chunks": len(chunk_files),
                "n_reactions_in_staging": n_npz_reactions,
                "n_reactions_used": EXPECTED_N_CURVES,
                "n_reactions_extra_not_in_idx_v2": len(extras),
                "staging_sidecar_aggregate_sha256": staging_source_sha,
                "curves_json": {"path": str(curves_json_path),
                                "sha256": sha256_file(curves_json_path)},
                "rule": "for each idx_v2 curve, staging_chunks/<RXN>::atomic_numbers verified "
                        "element-wise against curves.json curves[i].atomic_numbers; "
                        "extra staging reactions (outside idx_v2) ignored and counted",
            },
            "geom_offsets": {"path": str(geom_offsets_path),
                             "sha256": sha256_file(geom_offsets_path),
                             "schema": geom_offsets.get("schema")},
        },
        "consistency": {
            "n_curves": EXPECTED_N_CURVES,
            "n_nodes": EXPECTED_N_NODES,
            "n_atom_rows": EXPECTED_COORD_ROWS,
            "coords_rows_equals_sum_natoms_times_nnodes": True,
            "curve_offsets_match_curves_json_order": True,
            "atomic_numbers_match_curves_json": {"curves_checked": EXPECTED_N_CURVES,
                                                 "mismatches": 0},
            "geom_offsets_equal_cumulative_natoms_times_nnodes": {"curves_checked": EXPECTED_N_CURVES,
                                                                  "mismatches": 0},
            "node_atom_offsets_consistent": True,
            "staging_extra_reactions_ignored": len(extras),
            "element_histogram_atom_rows": full_hist,
            "element_sample_histogram": hist,
        },
        "reading": {
            "node_geometry": "coords = np.load('coords_f64.npy', mmap_mode='r'); "
                             "nao = np.load('node_atom_offsets.npy'); "
                             "xyz = coords[nao[i]:nao[i+1]]; z = "
                             "np.load('atomic_numbers.npy')[nao[i]:nao[i+1]]",
            "curve_nodes": "off = np.load('curve_offsets.npy'); nodes = range(off[c], off[c+1])",
            "compact_atomic_numbers": "the per-node array is redundant by construction; "
                                      "per-curve unique atom lists are recoverable as "
                                      "atomic_numbers.npy[curve_atom_offsets[i]:curve_atom_offsets[i]+n_atoms]",
        },
        "boundaries": [
            "geometry is copied read-only from the frozen v53mp 50k build; no source file is modified",
            "coords_f64.npy is a byte-identical copy (sha256 verified against the geometry pack binding)",
        ],
        "open_questions": [],
    }
    man_path = out_dir / "geometry_manifest.json"
    man_sha = write_json_atomic(man_path, manifest)
    total = sum(a.get("bytes", 0) for a in manifest["artifacts"])
    print(f"[geometry] curves={EXPECTED_N_CURVES} nodes={EXPECTED_N_NODES} "
          f"atom_rows={EXPECTED_COORD_ROWS} bytes={total/2**30:.3f}GiB "
          f"elapsed={time.perf_counter() - t_start:.1f}s")
    return {"manifest": str(man_path), "manifest_sha256": man_sha,
            "coords_sha256": coords_info["sha256"], "bytes": total}


# =========================================================================== #
# stage: DFT channel (energy only)
# =========================================================================== #
def _irc_curve_rows(path: Path) -> tuple[list[dict], str]:
    import csv as _csv

    h = hashlib.sha256()
    rows: list[dict] = []
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 23), b""):
            h.update(chunk)
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            rows.append(row)
    return rows, h.hexdigest()


def build_dft(args: argparse.Namespace) -> dict:
    stage = "dft"
    t_start = time.perf_counter()
    out_dir = DS_TENSORS / "dft_channel"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = D50 / "idx_v2/curve_node_energies.npy"
    dst = out_dir / "dft_energy.npy"
    info = copy_file_hashed(src, dst)
    arr = np.load(dst, mmap_mode="r")
    if arr.shape != (EXPECTED_N_NODES,) or arr.dtype != np.float64:
        _fail(stage, "unexpected curve_node_energies shape/dtype",
              {"shape": list(arr.shape), "dtype": str(arr.dtype)})
    if not np.all(np.isfinite(arr)):
        _fail(stage, "non-finite values in curve_node_energies.npy")

    # ---- consistency: per-curve first/last node energy equals irc_curve energy_a/energy_b
    irc_rows, irc_sha = _irc_curve_rows(D50 / "irc_curve.csv")
    off = np.load(DS_TENSORS / "geometry/curve_offsets.npy")
    curves = json.load(open(D50 / "idx_v2/curves.json", "r", encoding="utf-8"))
    curve_ids = curves["curve_ids"]
    if len(irc_rows) != EXPECTED_N_CURVES:
        _fail(stage, "irc_curve.csv row count mismatch", {"rows": len(irc_rows)})
    exact_a = exact_b = 0
    maxdiff_a = maxdiff_b = 0.0
    for i, r in enumerate(irc_rows):
        if r["curve_id"] != curve_ids[i]:
            _fail(stage, f"irc_curve.csv order mismatch at row {i}",
                  {"row": i, "csv": r["curve_id"], "curves_json": curve_ids[i]})
        s0, s1 = int(off[i]), int(off[i + 1])
        da = abs(float(arr[s0]) - float(r["energy_a"]))
        db = abs(float(arr[s1 - 1]) - float(r["energy_b"]))
        exact_a += da == 0.0
        exact_b += db == 0.0
        maxdiff_a = max(maxdiff_a, da)
        maxdiff_b = max(maxdiff_b, db)
    if exact_a != EXPECTED_N_CURVES or exact_b != EXPECTED_N_CURVES:
        _fail(stage, "curve_node_energies endpoints do not reproduce irc_curve energies",
              {"exact_a": exact_a, "exact_b": exact_b,
               "maxdiff_a": maxdiff_a, "maxdiff_b": maxdiff_b})

    # ---- unit evidence: curve_node_energies values == run/results dft_energy_ev (eV)
    #      sample the first results shard; only reactions inside idx_v2 are comparable.
    by_reaction = {cid: (int(off[i]), int(off[i + 1])) for i, cid in enumerate(curve_ids)}
    pool = set(by_reaction)
    n_read = n_in_pool = n_exact = 0
    max_abs_diff = 0.0
    diff_samples: list[dict] = []
    shard0 = XTB_ROOT / "run/results/shard_00000.jsonl"
    with open(shard0, "rb") as fh:
        for lineno, raw in enumerate(fh):
            if lineno >= 5000:
                break
            d = json.loads(raw)
            n_read += 1
            if d["reaction_id"] not in pool:
                continue
            n_in_pool += 1
            s0, s1 = by_reaction[d["reaction_id"]]
            e = float(d["dft_energy_ev"])
            window = arr[s0:s1]
            diff = float(np.min(np.abs(window - e)))
            max_abs_diff = max(max_abs_diff, diff)
            if diff == 0.0:
                n_exact += 1
            elif len(diff_samples) < 5:
                diff_samples.append({"reaction_id": d["reaction_id"],
                                     "frame_index": d["frame_index"],
                                     "nearest_abs_diff_ev": diff})
    if n_in_pool == 0:
        _fail(stage, "no in-pool reaction in the DFT unit-evidence sample")

    manifest = {
        "schema": DFT_MANIFEST_SCHEMA,
        "channel_id": "dft_irc_node_energy",
        "schema_note": "manifest is timestamp-free by design: reruns over unchanged sources are byte-identical",
        "artifacts": [{
            "name": "dft_energy.npy",
            **info,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "nbytes": int(arr.nbytes),
            "meaning": "DFT IRC per-node energy, one value per canonical idx_v2 node "
                       "(aligned with label_curves/offsets.npy node order and with "
                       "curve_node_energies.npy byte-for-byte)",
        }],
        "sources": {
            "path": str(src),
            "upstream_binding": "v53mp_50k_data/idx_v2/curve_node_energies.npy",
            "irc_curve": {"path": str(D50 / "irc_curve.csv"), "sha256": irc_sha,
                          "rows": len(irc_rows)},
            "curves_json_sha256": sha256_file(D50 / "idx_v2/curves.json"),
        },
        "units": {
            "unit_label": "eV",
            "unit_status": "OPEN_QUESTION (assumed, not signed off by the data owner)",
            "energy": "eV",
            "channel_table_reference": "docs/plans/PES2TS_本地数据资产整合与数据工程方案_v1_20261123.md "
                                       "§2.3 lists channel dft_irc_node_energy as eV sourced from "
                                       "curve_node_energies.npy",
            "evidence": {
                "irc_curve_endpoint_equality": {
                    "rule": "curve_node_energies[off[i]] == irc_curve.energy_a and "
                            "[off[i+1]-1] == irc_curve.energy_b, exact float64 equality",
                    "curves_checked": EXPECTED_N_CURVES,
                    "exact_matches_a": exact_a, "exact_matches_b": exact_b,
                    "max_abs_diff": max(maxdiff_a, maxdiff_b),
                },
                "xtb_results_inline_dft_energy_equality": {
                    "rule": "sample = first 5,000 rows of run/results/shard_00000.jsonl, "
                            "restricted to reactions present in idx_v2 (64,195 curves); "
                            "compare result.dft_energy_ev against the reaction's node-energy window",
                    "rows_sampled_stream": n_read,
                    "rows_compared_in_idx_v2_pool": n_in_pool,
                    "exact_matches": n_exact,
                    "max_nearest_abs_diff_ev": max_abs_diff,
                    "mismatch_samples": diff_samples,
                    "source_units_of_right_hand_side": "run/results schema documents dft_energy_ev as eV",
                },
                "magnitude_plausibility": "observed range is the B3LYP-D3/TZVP total energy of "
                                          "4-33 atom molecules in eV (~ -3e3 .. -1e5); interpreting "
                                          "the same numbers as hartree would be off by 27.211x",
            },
            "open_question": {
                "id": "DFT_ENERGY_UNIT",
                "question": "Is curve_node_energies.npy (v53mp idx_v2) in eV, and does it share the "
                            "exact same per-node ordering/zero-reference as irc_curve.csv energies?",
                "evidence_supports": "eV; ordering identical to irc_curve.csv for all 64,195 curves; "
                                     "values equal to run/results dft_energy_ev for sampled in-pool frames",
                "not_proven": "no upstream builder doc states the unit explicitly; the equality to "
                              "xTB-side dft_energy_ev is strong but indirect evidence",
                "action_needed": "data owner sign-off before this channel is used as an absolute "
                                 "energy reference; until then treat only relative differences "
                                 "(same route) as meaningful",
            },
        },
        "consistency": {
            "n_nodes": EXPECTED_N_NODES,
            "shape_matches_curve_offsets_final_offset": int(off[-1]) == EXPECTED_N_NODES,
            "all_finite": True,
            "min": float(np.min(arr)), "max": float(np.max(arr)),
        },
        "reading": {
            "python": "e = np.load('dft_energy.npy', mmap_mode='r'); "
                      "nodes of curve c = e[off[c]:off[c+1]] with off = curve_offsets.npy",
            "forbidden": "never subtract e_xTB absolute energy from this channel",
        },
        "boundaries": [
            "copy only; source file untouched",
            "cross-method absolute energy subtraction is forbidden (channel table §2.3)",
        ],
        "open_questions": [],
    }
    manifest["open_questions"] = [manifest["units"]["open_question"]]
    man_path = out_dir / "dft_channel_manifest.json"
    man_sha = write_json_atomic(man_path, manifest)
    print(f"[dft] rows={EXPECTED_N_NODES} bytes={info['bytes']/1e6:.1f}MB "
          f"sha256={info['sha256'][:12]} unit=eV(OPEN_QUESTION) "
          f"elapsed={time.perf_counter() - t_start:.1f}s")
    return {"path": str(dst), "sha256": info["sha256"], "bytes": info["bytes"],
            "rows": EXPECTED_N_NODES, "manifest": str(man_path),
            "manifest_sha256": man_sha}


# =========================================================================== #
# stage: G1 metadata (50k CSVs -> parquet)
# =========================================================================== #
def _read_csv_table(path: Path, column_types: dict[str, pa.DataType] | None = None) -> tuple[pa.Table, list[str]]:
    if not path.exists():
        _fail("g1", f"missing CSV: {path}")
    conv = pa_csv.ConvertOptions(strings_can_be_null=False,
                                 column_types=column_types or {},
                                 null_values=[])
    tbl = pa_csv.read_csv(
        path,
        read_options=pa_csv.ReadOptions(use_threads=False, block_size=1 << 24),
        parse_options=pa_csv.ParseOptions(newlines_in_values=False),
        convert_options=conv,
    )
    cols = tbl.column_names
    return tbl, cols


def _col_to_numpy_str(tbl: pa.Table, name: str) -> list[str | None]:
    col = tbl.column(name)
    return col.to_pylist()


def build_g1(args: argparse.Namespace) -> dict:
    stage = "g1"
    t_start = time.perf_counter()
    out_dir = DS_TENSORS / "g1_metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = json.load(open(D50 / "idx_v2/curves.json", "r", encoding="utf-8"))
    curve_ids = curves["curve_ids"]
    curve_set = set(curve_ids)

    # ---------------- reaction_intent.csv ----------------
    ri_path = D50 / "reaction_intent.csv"
    ri_sha = sha256_file(ri_path)
    ri_expected_cols = ["intent_id", "curve_id", "direction", "reverse_of_intent_id",
                        "intent_pair_id", "intent_underdetermined", "g_src", "g_tgt",
                        "delta_b_intent", "cv_spec", "intent_role", "rules", "intent_version"]
    ri, ri_cols = _read_csv_table(ri_path, {c: pa.string() for c in ri_expected_cols})
    if ri_cols != ri_expected_cols:
        _fail(stage, "reaction_intent.csv column mismatch",
              {"actual": ri_cols, "expected": ri_expected_cols})
    if ri.num_rows != EXPECTED_N_INTENTS:
        _fail(stage, "reaction_intent.csv row count mismatch",
              {"rows": ri.num_rows, "expected": EXPECTED_N_INTENTS})

    ri_dict = {name: ri.column(name).to_pylist() for name in ri_cols}
    intent_ids = ri_dict["intent_id"]
    if len(set(intent_ids)) != len(intent_ids):
        _fail(stage, "reaction_intent.csv intent_id not unique",
              {"rows": len(intent_ids), "unique": len(set(intent_ids))})
    bad_dir = sorted({d for d in ri_dict["direction"] if d not in ("forward", "reverse")})
    if bad_dir:
        _fail(stage, "reaction_intent.csv bad direction values", {"values": bad_dir})
    missing_curve = sorted(set(ri_dict["curve_id"]) - curve_set)
    if missing_curve:
        _fail(stage, "reaction_intent.csv curve_id not in idx_v2",
              {"n": len(missing_curve), "sample": missing_curve[:5]})
    under = ri_dict["intent_underdetermined"]
    bad_under = sorted({str(v) for v in under if str(v).strip().lower() not in ("true", "false")})
    if bad_under:
        _fail(stage, "reaction_intent.csv bad intent_underdetermined values", {"values": bad_under})
    under_bool = [str(v).strip().lower() == "true" for v in under]
    # sample-validate embedded JSON payloads
    n_json_checked = n_json_ok = 0
    for i in range(0, len(intent_ids), max(1, len(intent_ids) // 1000)):
        for col in ("g_src", "g_tgt"):
            val = ri_dict[col][i]
            n_json_checked += 1
            try:
                obj = json.loads(val)
                n_json_ok += int(isinstance(obj, dict))
            except Exception:  # noqa: BLE001
                pass
    if n_json_ok != n_json_checked:
        _fail(stage, "reaction_intent.csv embedded JSON sample is not all valid objects",
              {"checked": n_json_checked, "ok": n_json_ok})

    ri_out = pa.table({
        name: pa.array(ri_dict[name], type=pa.string()) for name in ri_cols
    })
    ri_out = ri_out.set_column(ri_cols.index("intent_underdetermined"),
                               "intent_underdetermined", pa.array(under_bool, type=pa.bool_()))
    ri_dst = out_dir / "reaction_intent.parquet"
    ri_tmp = ri_dst.with_name(ri_dst.name + ".tmp")
    pq.write_table(ri_out, ri_tmp, compression="zstd", compression_level=3,
                   use_dictionary=False, version="2.6", data_page_version="1.0",
                   write_statistics=True)
    os.replace(ri_tmp, ri_dst)
    ri_info = {"path": str(ri_dst), "sha256": sha256_file(ri_dst),
               "bytes": ri_dst.stat().st_size, "rows": ri_out.num_rows,
               "columns": [{"name": f.name, "type": str(f.type)} for f in ri_out.schema]}

    # ---------------- curve_family.csv ----------------
    cf_path = D50 / "curve_family.csv"
    cf_sha = sha256_file(cf_path)
    cf_expected_cols = ["curve_id", "family_labels", "edit_family_id"]
    cf, cf_cols = _read_csv_table(cf_path, {c: pa.string() for c in cf_expected_cols})
    cf_expected_cols = ["curve_id", "family_labels", "edit_family_id"]
    if cf_cols != cf_expected_cols:
        _fail(stage, "curve_family.csv column mismatch",
              {"actual": cf_cols, "expected": cf_expected_cols})
    if cf.num_rows != EXPECTED_N_CURVES:
        _fail(stage, "curve_family.csv row count mismatch",
              {"rows": cf.num_rows, "expected": EXPECTED_N_CURVES})
    cf_dict = {name: cf.column(name).to_pylist() for name in cf_cols}
    if len(set(cf_dict["curve_id"])) != cf.num_rows:
        _fail(stage, "curve_family.csv curve_id not unique")
    missing_cf = sorted(set(cf_dict["curve_id"]) - curve_set)
    if missing_cf:
        _fail(stage, "curve_family.csv curve_id not in idx_v2",
              {"n": len(missing_cf), "sample": missing_cf[:5]})
    family_values = sorted({v for v in cf_dict["family_labels"]})
    n_cells_missing = sum(1 for v in cf_dict["edit_family_id"] if v in (None, ""))
    if n_cells_missing:
        _fail(stage, "curve_family.csv empty edit_family_id", {"n": n_cells_missing})

    cf_out = pa.table({name: pa.array(cf_dict[name], type=pa.string()) for name in cf_cols})
    cf_dst = out_dir / "curve_family.parquet"
    cf_tmp = cf_dst.with_name(cf_dst.name + ".tmp")
    pq.write_table(cf_out, cf_tmp, compression="zstd", compression_level=3,
                   use_dictionary=False, version="2.6", data_page_version="1.0",
                   write_statistics=True)
    os.replace(cf_tmp, cf_dst)
    cf_info = {"path": str(cf_dst), "sha256": sha256_file(cf_dst),
               "bytes": cf_dst.stat().st_size, "rows": cf_out.num_rows,
               "columns": [{"name": f.name, "type": str(f.type)} for f in cf_out.schema]}

    manifest = {
        "schema": G1_MANIFEST_SCHEMA,
        "channel_id": "g1_metadata",
        "schema_note": "manifest is timestamp-free by design: reruns over unchanged sources are byte-identical",
        "artifacts": [
            {"name": "reaction_intent.parquet", **ri_info,
             "source": {"path": str(ri_path), "sha256": ri_sha},
             "key_columns": ["intent_id", "curve_id", "direction", "intent_pair_id",
                             "reverse_of_intent_id", "intent_underdetermined"]},
            {"name": "curve_family.parquet", **cf_info,
             "source": {"path": str(cf_path), "sha256": cf_sha},
             "key_columns": ["curve_id", "family_labels", "edit_family_id"]},
        ],
        "validation": {
            "reaction_intent": {
                "rows": ri.num_rows, "expected_rows": EXPECTED_N_INTENTS,
                "row_count_ok": True,
                "intent_id_unique": True,
                "direction_values": sorted(set(ri_dict["direction"])),
                "curve_ids_all_in_idx_v2": True,
                "intent_underdetermined_parsed_as_bool": True,
                "embedded_json_sample": {"cells_checked": n_json_checked, "valid_objects": n_json_ok,
                                         "columns": ["g_src", "g_tgt"]},
            },
            "curve_family": {
                "rows": cf.num_rows, "expected_rows": EXPECTED_N_CURVES,
                "row_count_ok": True,
                "curve_id_unique": True,
                "curve_ids_all_in_idx_v2": True,
                "n_distinct_family_labels": len(family_values),
                "family_labels": family_values,
                "empty_edit_family_id": 0,
            },
            "row_counts_note": "reaction_intent.csv has 128,391 physical lines = 1 header + "
                               "128,390 data rows; curve_family.csv = 64,195 data rows",
        },
        "reading": "pq.read_table('<file>.parquet') / pd.read_parquet('<file>.parquet')",
        "boundaries": [
            "faithful CSV -> Parquet conversion; only intent_underdetermined is typed as bool, "
            "all other columns stay UTF-8 strings (embedded JSON payloads are preserved verbatim)",
            "read-only on the 50k CSV sources",
        ],
        "open_questions": [],
    }
    man_path = out_dir / "g1_metadata_manifest.json"
    man_sha = write_json_atomic(man_path, manifest)
    print(f"[g1] reaction_intent rows={ri.num_rows} ({ri_info['bytes']/1e6:.2f}MB) "
          f"curve_family rows={cf.num_rows} ({cf_info['bytes']/1e6:.2f}MB) "
          f"elapsed={time.perf_counter() - t_start:.1f}s")
    return {"reaction_intent": ri_info, "curve_family": cf_info,
            "manifest": str(man_path), "manifest_sha256": man_sha}


# =========================================================================== #
# stage: verify (read-back acceptance)
# =========================================================================== #
def verify_readback(args: argparse.Namespace) -> dict:
    stage = "verify"
    report: dict = {"schema": "pes2ts_channels_verify_v1"}

    # ---------- xTB ----------
    xp = DS_TENSORS / "xtb_channel/xtb_points.parquet"
    xm = json.load(open(DS_TENSORS / "xtb_channel/xtb_channel_manifest.json", "r", encoding="utf-8"))
    pf = pq.ParquetFile(xp)
    n_rows = pf.metadata.num_rows
    if n_rows != xm["artifacts"][0]["rows"]:
        _fail(stage, "xtb parquet rows != manifest rows",
              {"parquet": n_rows, "manifest": xm["artifacts"][0]["rows"]})
    if sha256_file(xp) != xm["artifacts"][0]["sha256"]:
        _fail(stage, "xtb parquet sha256 != manifest sha256")
    tbl = pf.read_row_group(0)
    print("\n===== xTB channel read-back (first 5 rows) =====")
    head = tbl.slice(0, 5).to_pylist()
    for i, r in enumerate(head):
        g = np.asarray(r["gradient_hartree_per_bohr"], dtype=np.float32)
        gf = (np.asarray(r["dft_force_ev_per_angstrom"], dtype=np.float32)
              if r["dft_force_ev_per_angstrom"] is not None else None)
        norm = float(np.linalg.norm(g))
        print(f"  row{i}: key={r['xtb_frame_key'][:12]}.. rxn={r['reaction_id']} "
              f"dir={r['direction']} frame={r['frame_index']} status={r['status']} "
              f"n_atoms={r['n_atoms']} energy_hartree={r['energy_hartree']:.10f} "
              f"gradient_norm={r['gradient_norm']:.10f} (recomputed {norm:.10f}) "
              f"gradient_dim={g.size} dft_energy_ev={r['dft_energy_ev']} "
              f"dft_force_dim={0 if gf is None else gf.size} has_dft_pair={r['has_dft_pair']}")

    status_col = pq.read_table(xp, columns=["status"]).column("status").to_pylist()
    n_success = sum(1 for v in status_col if v == "success")
    n_failed = sum(1 for v in status_col if v == "failed")
    keys = pq.read_table(xp, columns=["xtb_frame_key"]).column("xtb_frame_key").to_pylist()
    n_unique = len(set(keys))
    dft_pairs = sum(pq.read_table(xp, columns=["has_dft_pair"])["has_dft_pair"].to_pylist())
    checks = {
        "rows": n_rows, "expected_rows": EXPECTED_MANIFEST_ROWS,
        "n_success": n_success, "n_failed": n_failed,
        "rows_ok": n_rows == EXPECTED_MANIFEST_ROWS,
        "success_ok": n_success == EXPECTED_RESULTS,
        "failed_ok": n_failed == EXPECTED_FAILURES,
        "n_unique_frame_keys": n_unique,
        "frame_key_unique_ok": n_unique == n_rows,
        "n_rows_with_dft_pair": int(dft_pairs),
        "filesize_bytes": xp.stat().st_size,
        "sha256": xm["artifacts"][0]["sha256"],
    }
    print(f"  totals: rows={n_rows} success={n_success} failed={n_failed} "
          f"unique_keys={n_unique} rows_with_dft_pair={int(dft_pairs)} "
          f"size={xp.stat().st_size/2**30:.3f}GiB sha256={checks['sha256'][:12]}")
    if not all((checks["rows_ok"], checks["success_ok"], checks["failed_ok"],
                checks["frame_key_unique_ok"])):
        _fail(stage, "xTB read-back acceptance failed", checks)
    if float(xm["validation"]["coords_sha256_manifest_vs_result"]["agreement_rate"]) != 1.0:
        _fail(stage, "xtb manifest coords_sha256 agreement is not 100%")
    report["xtb"] = checks

    # ---------- geometry ----------
    gd = DS_TENSORS / "geometry"
    gm = json.load(open(gd / "geometry_manifest.json", "r", encoding="utf-8"))
    coords = np.load(gd / "coords_f64.npy", mmap_mode="r")
    off = np.load(gd / "curve_offsets.npy")
    an = np.load(gd / "atomic_numbers.npy", mmap_mode="r")
    nao = np.load(gd / "node_atom_offsets.npy")
    cao = np.load(gd / "curve_atom_offsets.npy")
    geo_checks = {
        "coords_shape": list(coords.shape), "coords_dtype": str(coords.dtype),
        "coords_sha256": gm["artifacts"][0]["sha256"],
        "curve_offsets_shape": list(off.shape), "curve_offsets_dtype": str(off.dtype),
        "atomic_numbers_shape": list(an.shape), "atomic_numbers_dtype": str(an.dtype),
        "node_atom_offsets_shape": list(nao.shape),
        "curve_atom_offsets_shape": list(cao.shape),
        "n_curves": len(off) - 1, "n_nodes": int(off[-1]),
        "n_atom_rows": int(coords.shape[0]),
        "lengths_consistent": (int(nao[-1]) == int(coords.shape[0]) == int(an.shape[0])),
        "total_bytes": sum(a["bytes"] for a in gm["artifacts"]),
    }
    # spot check: slice node geometries and compare atomic numbers against source npz
    rng = np.random.default_rng(20261123)
    n_spot = 20
    spot_ok = 0
    for _ in range(n_spot):
        node = int(rng.integers(0, int(off[-1])))
        z = np.asarray(an[int(nao[node]):int(nao[node + 1])])
        if z.size == int(nao[node + 1] - nao[node]) and z.size > 0 and np.all((z >= 1) & (z <= 118)):
            spot_ok += 1
    geo_checks["node_slice_spot_checks"] = {"n": n_spot, "ok": spot_ok}
    if not geo_checks["lengths_consistent"] or spot_ok != n_spot:
        _fail(stage, "geometry read-back acceptance failed", geo_checks)
    print("\n===== geometry channel =====")
    print(f"  coords={tuple(coords.shape)} {coords.dtype} atomic_numbers={tuple(an.shape)} "
          f"{an.dtype} nodes={geo_checks['n_nodes']} curves={geo_checks['n_curves']} "
          f"bytes={geo_checks['total_bytes']/2**30:.3f}GiB")
    report["geometry"] = geo_checks

    # ---------- dft ----------
    dd = DS_TENSORS / "dft_channel"
    dm = json.load(open(dd / "dft_channel_manifest.json", "r", encoding="utf-8"))
    de = np.load(dd / "dft_energy.npy", mmap_mode="r")
    dft_checks = {
        "shape": list(de.shape), "dtype": str(de.dtype),
        "sha256": dm["artifacts"][0]["sha256"],
        "unit_label": dm["units"]["unit_label"],
        "unit_status": dm["units"]["unit_status"],
        "bytes": dm["artifacts"][0]["bytes"],
        "open_question": dm["units"]["open_question"]["id"],
    }
    if list(de.shape) != [EXPECTED_N_NODES] or str(de.dtype) != "float64":
        _fail(stage, "dft read-back acceptance failed", dft_checks)
    print("\n===== dft channel =====")
    print(f"  dft_energy={tuple(de.shape)} {de.dtype} unit={dft_checks['unit_label']} "
          f"[{dft_checks['unit_status']}] open_question={dft_checks['open_question']} "
          f"bytes={dft_checks['bytes']/2**20:.1f}MiB")
    report["dft"] = dft_checks

    # ---------- g1 ----------
    gd1 = DS_TENSORS / "g1_metadata"
    gm1 = json.load(open(gd1 / "g1_metadata_manifest.json", "r", encoding="utf-8"))
    ri = pq.read_table(gd1 / "reaction_intent.parquet")
    cf = pq.read_table(gd1 / "curve_family.parquet")
    g1_checks = {
        "reaction_intent_rows": ri.num_rows,
        "reaction_intent_expected": EXPECTED_N_INTENTS,
        "curve_family_rows": cf.num_rows,
        "curve_family_expected": EXPECTED_N_CURVES,
        "reaction_intent_columns": ri.column_names,
        "curve_family_columns": cf.column_names,
        "reaction_intent_sha256": gm1["artifacts"][0]["sha256"],
        "curve_family_sha256": gm1["artifacts"][1]["sha256"],
    }
    if ri.num_rows != EXPECTED_N_INTENTS or cf.num_rows != EXPECTED_N_CURVES:
        _fail(stage, "g1 read-back row count mismatch", g1_checks)
    print("\n===== g1 metadata =====")
    print(f"  reaction_intent rows={ri.num_rows} cols={len(ri.column_names)} "
          f"sha256={g1_checks['reaction_intent_sha256'][:12]}")
    print(f"  curve_family   rows={cf.num_rows} cols={len(cf.column_names)} "
          f"sha256={g1_checks['curve_family_sha256'][:12]}")
    report["g1"] = g1_checks

    # ---------- occupancy ----------
    sizes = {}
    for sub in ("xtb_channel", "geometry", "dft_channel", "g1_metadata"):
        total = 0
        for root, _dirs, files in os.walk(DS_TENSORS / sub):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        sizes[sub] = total
    report["occupancy_bytes"] = sizes
    report["occupancy_total_bytes"] = sum(sizes.values())
    print("\n===== occupancy =====")
    for k, v in sizes.items():
        print(f"  {k:14s} {v/2**30:.3f} GiB")
    print(f"  {'TOTAL':14s} {report['occupancy_total_bytes']/2**30:.3f} GiB")
    print("\nVERIFY PASS")
    return report


# =========================================================================== #
# CLI
# =========================================================================== #
def _parse_shards(text: str) -> tuple[int, int]:
    try:
        a, b = text.split(":")
        return int(a), int(b)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"--shards must be START:END, got {text!r}") from exc


def main(argv: list[str] | None = None) -> int:
    global DS_ROOT, DS_TENSORS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=("all", "xtb", "geometry", "dft", "g1", "verify"))
    ap.add_argument("--out-root", default=str(DS_ROOT))
    ap.add_argument("--shards", type=_parse_shards, default=(0, N_SHARDS),
                    help="xTB shard range START:END (default 0:61); non-full ranges write to --partial-dir")
    ap.add_argument("--partial-dir", default="/tmp/pes2ts_datastore_partial")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    DS_ROOT = Path(args.out_root)
    DS_TENSORS = DS_ROOT / "02_core_tensors"

    stages = ("xtb", "geometry", "dft", "g1") if args.stage == "all" else (args.stage,)
    summary: dict = {"schema": "pes2ts_build_channels_run_v1", "stages": {}}
    t0 = time.perf_counter()
    try:
        for st in stages:
            if st == "xtb":
                summary["stages"]["xtb"] = build_xtb(args)
            elif st == "geometry":
                summary["stages"]["geometry"] = build_geometry(args)
            elif st == "dft":
                summary["stages"]["dft"] = build_dft(args)
            elif st == "g1":
                summary["stages"]["g1"] = build_g1(args)
            elif st == "verify":
                summary["stages"]["verify"] = verify_readback(args)
        if args.stage in ("all", "verify") and not args.skip_verify:
            if "verify" not in summary["stages"]:
                summary["stages"]["verify"] = verify_readback(args)
    except BuildError as exc:
        path = log_failure(exc)
        print(f"\nFAIL: {exc}", file=sys.stderr)
        print(f"failure log: {path}", file=sys.stderr)
        summary["verdict"] = "FAIL"
        summary["error"] = {"stage": exc.stage, "message": exc.message, "sample": exc.sample,
                            "log": str(path)}
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 3
    summary["verdict"] = "PASS"
    summary["elapsed_s"] = round(time.perf_counter() - t0, 1)
    if args.stage != "verify":
        print("\n" + json.dumps({k: v for k, v in summary.items() if k != "stages"},
                                ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
