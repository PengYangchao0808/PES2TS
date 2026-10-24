#!/usr/bin/env python3
"""Geometry memmap-pack certificate (S4 plan §5.2 / lane I2 gate).

Verifies on the REAL tables that the geom pack serves EXACTLY the bytes
the shard path serves, plus the binding and the latency win:

1. **frame_xyz equality** — 100 random curves × ALL their nodes: the
   geom-pack path (``use_geometry_pack=True``) vs the shard-LRU path
   (``use_geometry_pack=False``) on two ``ColumnarKeyAtomDatasetV3``
   instances, compared as float64 BYTES (``tobytes()``).
2. **to_sample_v3 equality** — one sample per probed curve through
   ``collate_v3(build_sample_features_v3(...))``: every batch array
   ``np.array_equal`` + dtype-equal.
3. **binding** — the pack's binding block vs the LIVE shard_index /
   data_manifest shas (the reader already fail-closed refuses on drift;
   this re-reports the matched shas).
4. **latency** — frame_xyz first-access latency over the 100 random
   curves on BOTH paths (cold LRU vs memmap slice): median/p95 ms.

Writes ``outputs/pesn/v53mp/reports/geometry_pack_certificate_<sha12>.json``
(+ ``.md``); exit code 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    build_sample_features_v3,
    collate_v3,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    LABEL_STATUSES,
    _sha256,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (  # noqa: E402
    GEOM_DIRNAME,
    GEOM_OFFSETS_FILENAME,
    ColumnarKeyAtomDatasetV3,
)

DEFAULT_TABLES_DIR = PROJECT_ROOT / "outputs/pesn/v53mp/data"
REPORT_DIR = PROJECT_ROOT / "outputs/pesn/v53mp/reports"
SEED = 17
N_CURVES = 100


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_certificate(tables_dir: Path, report_dir: Path, *,
                    n_curves: int = N_CURVES,
                    verify_manifest: bool = False) -> dict[str, Any]:
    tables_dir = Path(tables_dir)
    timings: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    rss_start = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) // 1024

    offsets_path = tables_dir / GEOM_DIRNAME / GEOM_OFFSETS_FILENAME
    if not offsets_path.is_file():
        raise SystemExit(
            "error: GEOM_MISSING: {} not found — run "
            "r3_v53mp_build_geometry_pack.py first".format(offsets_path))

    _log("[1/5] opening datasets (pack vs shard paths) …")
    t0 = time.perf_counter()
    ds_pack = ColumnarKeyAtomDatasetV3(
        tables_dir, label_statuses=LABEL_STATUSES,
        verify_manifest=verify_manifest, use_geometry_pack=True)
    timings["open_pack_s"] = round(time.perf_counter() - t0, 1)
    _require(ds_pack.geom_pack_binding is not None, "PACK_NOT_ACTIVE",
             "geom pack did not activate")
    t0 = time.perf_counter()
    ds_shard = ColumnarKeyAtomDatasetV3(
        tables_dir, label_statuses=LABEL_STATUSES,
        verify_manifest=verify_manifest, use_geometry_pack=False)
    timings["open_shard_s"] = round(time.perf_counter() - t0, 1)
    _require(ds_shard.geom_pack_binding is None, "PACK_UNEXPECTED",
             "shard-path dataset unexpectedly activated the pack")
    _log(f"    pack open {timings['open_pack_s']}s "
         f"(geom verify {round(ds_pack.geom_verify_seconds, 2)}s), "
         f"shard open {timings['open_shard_s']}s")

    # --- binding -------------------------------------------------------------
    _log("[2/5] verifying binding …")
    binding = ds_pack.geom_pack_binding
    index_path = tables_dir / "shard_index_v53mp.json"
    manifest_path = tables_dir / "data_manifest.json"
    binding_ok = (
        str(binding.get("shard_index_sha256")) == _sha256(index_path)
        and str(binding.get("data_manifest_sha256")) == _sha256(manifest_path)
        and str(binding.get("coords_sha256"))
        == _sha256(tables_dir / GEOM_DIRNAME
                   / str(binding.get("coords_filename"))))
    checks["binding"] = {
        "passed": bool(binding_ok),
        "shard_index_sha256": str(binding.get("shard_index_sha256")),
        "data_manifest_sha256": str(binding.get("data_manifest_sha256")),
        "coords_sha256": str(binding.get("coords_sha256")),
        "n_curves": binding.get("n_curves"),
        "total_atom_rows": binding.get("total_atom_rows"),
    }

    # --- sample curves -------------------------------------------------------
    _log(f"[3/5] frame_xyz equality on {n_curves} random curves …")
    t0 = time.perf_counter()
    rng = np.random.RandomState(SEED)
    curve_ids = list(ds_pack._curve_ids)
    probe = [int(x) for x in rng.choice(
        len(curve_ids), size=min(n_curves, len(curve_ids)), replace=False)]
    n_nodes_total = 0
    byte_mismatch = 0
    first_bad = None
    for c in probe:
        cid = curve_ids[c]
        for node in range(int(ds_pack._curves[cid]["n_nodes"])):
            got = np.asarray(ds_pack.frame_xyz(cid, node), dtype=np.float64)
            want = np.asarray(ds_shard.frame_xyz(cid, node), dtype=np.float64)
            n_nodes_total += 1
            if got.shape != want.shape or got.tobytes() != want.tobytes():
                byte_mismatch += 1
                if first_bad is None:
                    first_bad = {"curve": cid, "node": node}
    checks["frame_xyz_bytes_equal"] = {
        "passed": byte_mismatch == 0,
        "curves_checked": len(probe),
        "frames_checked": n_nodes_total,
        "byte_mismatches": byte_mismatch,
        "first_mismatch": first_bad,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    _log(f"    {n_nodes_total} frames, {byte_mismatch} mismatches")

    _log("[4/5] to_sample_v3 feature equality …")
    t0 = time.perf_counter()
    feat_mismatch = 0
    feat_checked = 0
    for c in probe:
        cid = curve_ids[c]
        row = ds_pack._curve_row[cid]
        pos = int(np.searchsorted(
            ds_pack._columns["curve_idx"], row, side="left"))
        # advance to the first sample row of this curve (columns are grouped)
        while (pos < ds_pack._n_rows
               and int(ds_pack._columns["curve_idx"][pos]) != row):
            pos += 1
        _require(pos < ds_pack._n_rows, "CURVE_NO_SAMPLE",
                 f"curve {cid} has no samples")
        rec_p = ds_pack.records([pos])[0]
        rec_s = ds_shard.records([pos])[0]
        b_p = collate_v3([build_sample_features_v3(
            ds_pack.to_sample_v3(rec_p), level="F2", intent_off=False)])
        b_s = collate_v3([build_sample_features_v3(
            ds_shard.to_sample_v3(rec_s), level="F2", intent_off=False)])
        feat_checked += 1
        for key in sorted(set(b_p) | set(b_s)):
            va, vb = b_p.get(key), b_s.get(key)
            if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
                if (va.dtype != vb.dtype
                        or not np.array_equal(va, vb)):
                    feat_mismatch += 1
                    _log("    MISMATCH {} key {}".format(cid, key))
            elif va != vb:
                feat_mismatch += 1
                _log("    MISMATCH {} key {}".format(cid, key))
    checks["features_equal"] = {
        "passed": feat_mismatch == 0,
        "samples_checked": feat_checked,
        "array_mismatches": feat_mismatch,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }

    # --- latency -------------------------------------------------------------
    _log("[5/5] frame_xyz latency (first access per curve) …")

    def _latency(dataset: Any) -> dict:
        lat = []
        for c in probe:
            cid = curve_ids[c]
            node = int(rng.randint(0, int(dataset._curves[cid]["n_nodes"])))
            t = time.perf_counter()
            dataset.frame_xyz(cid, node)
            lat.append((time.perf_counter() - t) * 1000.0)
        lat.sort()
        return {
            "median_ms": round(lat[len(lat) // 2], 4),
            "p95_ms": round(lat[int(round(0.95 * (len(lat) - 1)))], 4),
            "mean_ms": round(float(np.mean(lat)), 4),
            "max_ms": round(lat[-1], 4),
        }

    # cold shard path (fresh instance, empty LRU) vs warm pack path
    ds_cold = ColumnarKeyAtomDatasetV3(
        tables_dir, label_statuses=LABEL_STATUSES,
        verify_manifest=False, use_geometry_pack=False, cache_curves=1)
    checks["latency"] = {
        "pack_first_access": _latency(ds_pack),
        "shard_cold_first_access": _latency(ds_cold),
        "note": ("first access per curve: shard path pays npz "
                 "decompression, pack path pays one memmap slice"),
    }
    _log("    pack: {} | shard cold: {}".format(
        checks["latency"]["pack_first_access"],
        checks["latency"]["shard_cold_first_access"]))

    passed = all(c.get("passed", True) for c in checks.values())
    rss_end = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) // 1024
    verdict = {
        "schema": "v53mp_geometry_pack_certificate_v1",
        "passed": bool(passed),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tables_dir": str(tables_dir),
        "protocol": {"seed": SEED, "n_curves": n_curves},
        "checks": checks,
        "timings": timings,
        "rss_start_mib": rss_start,
        "rss_end_mib": rss_end,
    }
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    short = str(binding.get("coords_sha256"))[:12]
    report_path = report_dir / "geometry_pack_certificate_{}.json".format(short)
    report_path.write_text(
        json.dumps(verdict, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    md = ["# Geometry pack certificate {}".format(verdict["created_at"]), "",
          "- passed: **{}**".format(verdict["passed"]),
          "- tables: `{}`".format(tables_dir),
          "- coords_sha256: `{}`".format(binding.get("coords_sha256")),
          "- curves checked: {} ({} frames)".format(
              checks["frame_xyz_bytes_equal"]["curves_checked"],
              checks["frame_xyz_bytes_equal"]["frames_checked"]), "",
          "| check | passed |", "|---|---|",
          "| binding | {} |".format(checks["binding"]["passed"]),
          "| frame_xyz bytes equal | {} |".format(
              checks["frame_xyz_bytes_equal"]["passed"]),
          "| features equal | {} |".format(
              checks["features_equal"]["passed"]), "",
          "## Latency (first access per curve)", "",
          "```json", json.dumps(checks["latency"], indent=1,
                                sort_keys=True), "```", ""]
    report_path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    verdict["report_path"] = str(report_path)
    return verdict


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise SystemExit("error: {}: {}".format(code, detail))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_v53mp_geometry_pack_certificate",
        description="Bitwise certificate: geom pack vs shard path (plan §5.2)")
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--n-curves", type=int, default=N_CURVES)
    parser.add_argument("--verify-manifest", action="store_true",
                        help="also verify every idx/shard sha at open "
                             "(slower; the binding check is always on)")
    args = parser.parse_args(argv)
    verdict = run_certificate(
        Path(args.tables_dir), Path(args.report_dir),
        n_curves=args.n_curves, verify_manifest=args.verify_manifest)
    print(json.dumps(verdict, indent=1, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
