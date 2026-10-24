#!/usr/bin/env python3
"""Build the V5.3-MP geometry memmap pack (S4 plan §5.2 / lane I2).

Concatenates the per-curve ``[n_nodes, n_atoms, 3]`` float64 coords of every
``shard_*.npz`` into ONE flat memmap file ``<tables_dir>/geom/coords.f64.npy``
plus an offset table ``<tables_dir>/geom/curve_offsets.json``::

    {
      "schema": "v53mp_geom_pack_v1",
      "binding": {
        "shard_index_sha256": <sha of shard_index_v53mp.json>,
        "data_manifest_sha256": <sha of data_manifest.json>,
        "n_curves": 26038, "total_frames": ..., "created_at": ...,
        "builder_git_commit": ...,
      },
      "curves": {curve_id: {"offset": <atom-row offset>, "n_nodes": n,
                             "n_atoms": N, "shard_sha256": <source sha>}}
    }

* ``offset`` is in ATOM-ROWS: the flat file is laid out as
  ``[total_atom_rows, 3]`` float64; curve ``c``'s node ``k`` lives at
  ``coords[offset + k*n_atoms : offset + (k+1)*n_atoms]``.
* Values are copied BYTE-IDENTICALLY from the npz arrays (both float64;
  no dtype conversion, no arithmetic) — the I2 certificate re-verifies
  equality against the shards on random curves.
* The reader (:class:`ColumnarKeyAtomDatasetV3`) fail-closed verifies the
  binding (shard_index sha + data_manifest sha) at init; when ``geom/`` is
  absent the reader keeps the v1 shard-LRU path unchanged.

Reads only existing artifacts (shards + shard_index + data_manifest);
writes only ``<tables_dir>/geom/``.  Progress on stderr, summary JSON on
stdout, typed refusals (exit 1) on any invariant violation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3Error,
    SHARD_INDEX_FILENAME,
    SHARD_INDEX_VERSION,
    _require,
    _sha256,
)

GEOM_DIRNAME = "geom"
COORDS_FILENAME = "coords.f64.npy"
OFFSETS_FILENAME = "curve_offsets.json"
GEOM_SCHEMA = "v53mp_geom_pack_v1"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, check=True)
        commit = proc.stdout.strip()
        if len(commit) == 40:
            return commit
    except (OSError, subprocess.CalledProcessError):
        pass
    return ""


def build_geometry_pack(tables_dir: Path, out_dir: Path, *,
                        overwrite: bool = False) -> dict:
    """Build ``geom/`` from the frozen shards; returns the summary dict."""
    t_start = time.perf_counter()
    tables = Path(tables_dir)
    out = Path(out_dir)
    _require(tables.is_dir(), "TABLES_DIR", f"{tables} is not a directory")
    index_path = tables / SHARD_INDEX_FILENAME
    manifest_path = tables / "data_manifest.json"
    _require(index_path.is_file(), "SOURCE_MISSING", f"{index_path} not found")
    _require(manifest_path.is_file(), "SOURCE_MISSING",
             f"{manifest_path} not found")
    if out.exists() and any(out.iterdir()) and not overwrite:
        _require(False, "GEOM_EXISTS",
                 f"{out} is not empty; pass --overwrite to rebuild")

    _log("[1/4] reading shard index …")
    doc = json.loads(index_path.read_text(encoding="utf-8"))
    _require(str(doc.get("version")) == SHARD_INDEX_VERSION, "SHARD_INDEX_VERSION",
             f"unexpected shard index version {doc.get('version')!r}")
    shards = doc.get("shards")
    _require(isinstance(shards, list) and shards, "SHARD_INDEX_SHAPE",
             "shard index has no shards")
    index_sha = _sha256(index_path)
    manifest_sha = _sha256(manifest_path)

    # pass 1: verify shard shas + total size (fail-closed BEFORE writing)
    _log("[2/4] verifying shard sha256s + sizing …")
    entries: list[tuple[str, str, list[str]]] = []
    total_atom_rows = 0
    for entry in shards:
        rel = str(entry.get("path") or "")
        expected = str(entry.get("sha256") or "")
        curve_ids = [str(c) for c in entry.get("curve_ids", ())]
        _require(bool(rel) and bool(expected) and curve_ids,
                 "SHARD_INDEX_SHAPE", f"bad shard entry {entry!r}")
        shard_path = tables / rel
        _require(shard_path.is_file(), "SHARD_FILE", f"{rel} not found")
        actual = _sha256(shard_path)
        _require(actual == expected, "SHARD_SHA",
                 f"{rel}: sha256 {actual[:12]}… != index {expected[:12]}…")
        entries.append((rel, expected, curve_ids))

    # pass 2: size + write.  The flat file is [total_atom_rows, 3] float64;
    # offsets are atom-row indices (curve block = n_nodes * n_atoms rows).
    curves: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    # first pass over members to get the total atom-row count
    for rel, _sha, curve_ids in entries:
        with np.load(tables / rel, allow_pickle=False) as data:
            for cid in curve_ids:
                _require(cid not in seen, "CURVE_DUP",
                         f"curve {cid} appears in more than one shard")
                seen.add(cid)
                coords_key = f"{cid}::coords"
                nodes_key = f"{cid}::node_indices"
                _require(coords_key in data.files
                         and nodes_key in data.files, "SHARD_CURVE_MISSING",
                         f"{rel} has no geometry for curve {cid}")
                coords = data[coords_key]
                _require(coords.dtype == np.float64, "COORDS_DTYPE",
                         f"{cid}: coords dtype {coords.dtype} != float64")
                _require(coords.ndim == 3 and coords.shape[-1] == 3,
                         "COORDS_SHAPE", f"{cid}: bad coords shape "
                         f"{coords.shape}")
                total_atom_rows += int(coords.shape[0]) * int(coords.shape[1])
    _log(f"    {len(seen)} curves, {total_atom_rows} atom rows "
         f"({total_atom_rows * 24 / 1e9:.2f} GB)")

    out.mkdir(parents=True, exist_ok=True)
    coords_path = out / COORDS_FILENAME
    mm = np.lib.format.open_memmap(
        coords_path, mode="w+", dtype=np.float64,
        shape=(total_atom_rows, 3))
    cursor = 0
    n_written = 0
    for rel, shard_sha, curve_ids in entries:
        with np.load(tables / rel, allow_pickle=False) as data:
            for cid in curve_ids:
                coords = np.asarray(data[f"{cid}::coords"], dtype=np.float64)
                node_indices = [int(v) for v in data[f"{cid}::node_indices"]]
                # fail-closed: the flat pack indexes frames by node position
                # (offset + node * n_atoms), so the shard MUST store nodes
                # in ascending order (write_geometry_shards guarantees it)
                _require(node_indices == list(range(len(node_indices))),
                         "NODE_ORDER",
                         f"{cid}: shard node_indices {node_indices[:5]}… "
                         "are not ascending 0..n-1")
                n_nodes, n_atoms = int(coords.shape[0]), int(coords.shape[1])
                # BYTE-IDENTICAL copy: same dtype, no arithmetic
                mm[cursor:cursor + n_nodes * n_atoms] = coords.reshape(
                    n_nodes * n_atoms, 3)
                curves[cid] = {
                    "offset": int(cursor),
                    "n_nodes": n_nodes,
                    "n_atoms": n_atoms,
                    "shard_sha256": shard_sha,
                }
                cursor += n_nodes * n_atoms
                n_written += 1
                if n_written % 5000 == 0:
                    _log(f"    {n_written}/{len(seen)} curves …")
    _require(cursor == total_atom_rows, "PACK_SIZE",
             f"wrote {cursor} atom rows, expected {total_atom_rows}")
    mm.flush()
    del mm
    _log(f"    wrote {coords_path} ({time.perf_counter() - t_start:.1f}s)")

    _log("[3/4] writing curve_offsets.json …")
    offsets_doc = {
        "schema": GEOM_SCHEMA,
        "binding": {
            "shard_index_sha256": index_sha,
            "data_manifest_sha256": manifest_sha,
            "n_curves": len(curves),
            "total_atom_rows": total_atom_rows,
            "coords_filename": COORDS_FILENAME,
            "coords_sha256": _sha256(coords_path),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "builder_git_commit": _git_commit(),
        },
        "curves": curves,
    }
    (out / OFFSETS_FILENAME).write_text(
        json.dumps(offsets_doc, sort_keys=True) + "\n", encoding="utf-8")

    total_bytes = int(coords_path.stat().st_size)
    summary = {
        "schema": GEOM_SCHEMA,
        "geom_dir": str(out),
        "n_curves": len(curves),
        "total_atom_rows": total_atom_rows,
        "total_bytes": total_bytes,
        "coords_sha256": offsets_doc["binding"]["coords_sha256"],
        "shard_index_sha256": index_sha,
        "data_manifest_sha256": manifest_sha,
        "elapsed_s": round(time.perf_counter() - t_start, 1),
        "created_at": offsets_doc["binding"]["created_at"],
        "builder_git_commit": offsets_doc["binding"]["builder_git_commit"],
    }
    _log("[4/4] done")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_v53mp_build_geometry_pack",
        description="Concatenate shard coords into one flat memmap (plan §5.2)")
    parser.add_argument(
        "--tables-dir",
        default=str(PROJECT_ROOT / "outputs/pesn/v53mp/data"))
    parser.add_argument("--out", default=None,
                        help="default: <tables-dir>/geom")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    tables = Path(args.tables_dir)
    out = Path(args.out) if args.out else tables / GEOM_DIRNAME
    try:
        summary = build_geometry_pack(tables, out, overwrite=args.overwrite)
    except KeyAtomDatasetV3Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
