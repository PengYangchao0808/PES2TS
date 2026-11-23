#!/usr/bin/env python3
"""Thread-env A/B checker (plan §5.5 / I3 / B3 gate evidence).

A noise-lane finding (S4 pre-work) showed BLAS thread count can perturb
LOW-ORDER float bits in feature builds.  Plan §5.5 mandates
``OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`` per run with an expected 5–15%
gain — but ONLY if it does not change numerics.  This script is the gate
evidence: it builds the SAME K feature dicts (seed-17 positions on the v2
loader) in THREE subprocess environments —

  * ``default`` — the parent's inherited thread env,
  * ``t1``      — OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1,
  * ``t8``      — ...=8

— each in a fresh interpreter (thread pools bind at import, so an in-
process env change would not take effect), and compares the encoded
feature blobs BITWISE across envs, reporting per-env wall time.

The worker (``--_worker``) encodes each built dict with the pickle-free
dev-pack codec (:func:`blob_encode`) and reports one sha256 per position;
the parent compares the three hash lists.  CPU only (no GPU, no training,
no ledger).

Verdict: ``{bitwise_identical: bool, timings: {...}}`` written as
``reports/thread_env_ab_<ts>.json`` (+ ``.md``).  Exit 0 iff the three
envs are bitwise identical (the §5.5 thread caps are then safe to
mandate); a difference is a REAL finding, reported, exit 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TABLES_DIR = PROJECT_ROOT / "outputs/pesn/v53mp/data"
REPORT_DIR = PROJECT_ROOT / "outputs/pesn/v53mp/reports"
SEED = 17
DEFAULT_K = 512
THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
#: env name -> per-key values (None = inherit the parent env untouched)
ENVS = {
    "default": None,
    "t1": {k: "1" for k in THREAD_ENV_KEYS},
    "t8": {k: "8" for k in THREAD_ENV_KEYS},
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# worker (subprocess): build K features, emit per-position blob hashes
# ---------------------------------------------------------------------------
def run_worker(tables_dir: Path, positions_path: Path, out_path: Path) -> int:
    """Build features for the positions in ``positions_path`` (json list).

    Emits ``{"positions": [...], "hashes": [...], "wall_s": float,
    "env": {...}}`` — one sha256 per encoded feature dict.
    """
    from pes2ts_core.pesn.features.key_atom_graph_v3 import (
        build_sample_features_v3,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        LABEL_STATUSES,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        ColumnarKeyAtomDatasetV3,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
        blob_encode,
    )

    positions = [int(p) for p in json.loads(
        positions_path.read_text(encoding="utf-8"))]
    ds = ColumnarKeyAtomDatasetV3(
        tables_dir, label_statuses=LABEL_STATUSES, verify_manifest=False)
    _require_len = len(ds)
    for p in positions:
        if not 0 <= p < _require_len:
            raise SystemExit("error: POSITION_RANGE: {}".format(p))
    t0 = time.perf_counter()
    hashes: List[str] = []
    for p in positions:
        rec = ds.records([p])[0]
        built = build_sample_features_v3(
            ds.to_sample_v3(rec), level="F2", intent_off=False)
        blob = blob_encode(built)
        hashes.append(hashlib.sha256(blob).hexdigest())
    wall = time.perf_counter() - t0
    env = {k: os.environ.get(k) for k in THREAD_ENV_KEYS}
    out_path.write_text(json.dumps({
        "positions": positions,
        "hashes": hashes,
        "wall_s": round(wall, 3),
        "env": env,
        "per_feature_ms": round(wall * 1000.0 / max(len(positions), 1), 3),
    }, sort_keys=True) + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# parent: sample positions, run the3 envs, compare bitwise
# ---------------------------------------------------------------------------
def _sample_positions(tables_dir: Path, k: int) -> List[int]:
    import numpy as np

    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        LABEL_STATUSES,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        ColumnarKeyAtomDatasetV3,
    )
    ds = ColumnarKeyAtomDatasetV3(
        tables_dir, label_statuses=LABEL_STATUSES, verify_manifest=False)
    rng = np.random.RandomState(SEED)
    n = min(int(k), len(ds))
    positions = sorted(int(x) for x in rng.choice(
        len(ds), size=n, replace=False))
    return positions


def run_ab(tables_dir: Path, report_dir: Path, *, k: int = DEFAULT_K,
           keep_worker_outputs: bool = False) -> Dict[str, Any]:
    tables_dir = Path(tables_dir)
    report_dir = Path(report_dir)
    _log("[1/3] sampling {} positions (seed {}) …".format(k, SEED))
    positions = _sample_positions(tables_dir, k)
    work = report_dir / ".thread_env_ab_work"
    work.mkdir(parents=True, exist_ok=True)
    pos_path = work / "positions.json"
    pos_path.write_text(json.dumps(positions), encoding="utf-8")

    results: Dict[str, Dict[str, Any]] = {}
    for name, overrides in ENVS.items():
        _log("[2/3] env {} …".format(name))
        out_path = work / "{}.json".format(name)
        env = dict(os.environ)
        if overrides:
            for key in THREAD_ENV_KEYS:
                env.pop(key, None)
            env.update(overrides)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_worker",
             "--tables-dir", str(tables_dir),
             "--positions-file", str(pos_path),
             "--out", str(out_path)],
            cwd=str(PROJECT_ROOT), env=env,
            capture_output=True, text=True)
        if proc.returncode != 0 or not out_path.is_file():
            raise SystemExit(
                "error: WORKER_FAILED env={} rc={} stderr_tail={}".format(
                    name, proc.returncode, proc.stderr[-500:]))
        results[name] = json.loads(out_path.read_text(encoding="utf-8"))
        _log("    {}: {} hashes, {} s ({}/feature)".format(
            name, len(results[name]["hashes"]), results[name]["wall_s"],
            results[name]["per_feature_ms"]))

    _log("[3/3] comparing bitwise …")
    names = list(ENVS)
    ref = results[names[0]]
    bitwise = True
    first_diff: Optional[Dict[str, Any]] = None
    for name in names[1:]:
        other = results[name]
        if other["positions"] != ref["positions"]:
            bitwise = False
            first_diff = {"envs": [names[0], name], "kind": "positions"}
            break
        for i, (a, b) in enumerate(zip(ref["hashes"], other["hashes"])):
            if a != b:
                bitwise = False
                first_diff = {"envs": [names[0], name],
                              "kind": "feature_blob",
                              "position": ref["positions"][i],
                              "hash_a": a, "hash_b": b}
                break
        if not bitwise:
            break
    verdict = {
        "schema": "v53mp_thread_env_ab_v1",
        "bitwise_identical": bool(bitwise),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tables_dir": str(tables_dir),
        "n_positions": len(positions),
        "seed": SEED,
        "thread_env_keys": list(THREAD_ENV_KEYS),
        "timings": {name: {"wall_s": r["wall_s"],
                           "per_feature_ms": r["per_feature_ms"],
                           "env": r["env"]}
                    for name, r in results.items()},
        "first_diff": first_diff,
        "note": ("thread caps (plan §5.5) are numerics-safe iff "
                 "bitwise_identical is true"),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = report_dir / "thread_env_ab_{}.json".format(stamp)
    report_path.write_text(
        json.dumps(verdict, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    md = ["# Thread-env A/B {}".format(verdict["created_at"]), "",
          "- bitwise_identical: **{}**".format(verdict["bitwise_identical"]),
          "- positions: {} (seed {})".format(len(positions), SEED),
          "- tables: `{}`".format(tables_dir), "",
          "| env | wall (s) | ms/feature |", "|---|---|---|"]
    for name in ENVS:
        md.append("| {} | {} | {} |".format(
            name, verdict["timings"][name]["wall_s"],
            verdict["timings"][name]["per_feature_ms"]))
    if first_diff:
        md += ["", "First difference: {}".format(
            json.dumps(first_diff, sort_keys=True))]
    md += ["", "```json", json.dumps(verdict["timings"], indent=1,
                                      sort_keys=True), "```", ""]
    report_path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    verdict["report_path"] = str(report_path)
    if not keep_worker_outputs:
        for name in ENVS:
            (work / "{}.json".format(name)).unlink(missing_ok=True)
    return verdict


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="thread_env_ab",
        description="BLAS/OMP thread-env A/B: bitwise feature equality "
                    "across {default, threads=1, threads=8} (plan §5.5)")
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--positions", type=int, default=DEFAULT_K,
                        help="K random positions (seed 17; default 512)")
    parser.add_argument("--_worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--positions-file", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        if not args.positions_file or not args.out:
            raise SystemExit(
                "error: WORKER_ARGS: --positions-file and --out required")
        return run_worker(Path(args.tables_dir),
                          Path(args.positions_file), Path(args.out))
    verdict = run_ab(Path(args.tables_dir), Path(args.report_dir),
                     k=args.positions)
    print(json.dumps(verdict, indent=1, sort_keys=True))
    return 0 if verdict["bitwise_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
