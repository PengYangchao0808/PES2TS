#!/usr/bin/env python3
"""Startup fast-path certificate (S4 plan §7-B2 / lane I4).

On the REAL 20k tables + split manifest, for seeds 17/29/43, compares the
I4 columnar startup fast paths against the ORIGINAL per-item path (the
fast path force-disabled via the ``tr._COLUMNAR_FAST`` module flag — the
same switch ``V53MP_DISABLE_COLUMNAR_FAST=1`` sets at import):

* ``v53mp_pool_split`` — train / inner_val position lists EXACTLY equal;
* ``_CurveDirIndex.build`` — STRING-sorted curve list + every pool
  (positions per (curve, direction)) exactly equal;
* ``_assert_no_confirm`` — both accept the same clean sets (a confirm
  refusal is pinned by the unit tests on the synthetic fixture);
* dev positions — the vectorized rule vs the per-item enumeration.

Plus a startup BREAKDOWN with the fast path ON: dataset init, first
column materialization, pool split, curve-dir index, model + R3 ckpt
load, and a first-update proxy (one sampler draw + records +
``to_sample_v3`` + collate).  CPU only (no GPU required); the real
spawn→first-train_log startup is proven by the 600-update bench run.

Report: ``outputs/pesn/v53mp/reports/startup_certificate_<sha12>.json``
(+ ``.md``).  Exit code 0 iff every equality holds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    LABEL_STATUSES,
    _sha256,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (  # noqa: E402
    ColumnarKeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.training import key_atom_v53mp_train_v3 as tr  # noqa: E402
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (  # noqa: E402
    V53MPTrainerConfig,
    _CurveDirIndex,
    _assert_no_confirm,
    _curve_batch_positions,
    _update_rng,
    v53mp_pool_split,
)

DEFAULT_TABLES_DIR = PROJECT_ROOT / "outputs/pesn/v53mp/data"
REPORT_DIR = PROJECT_ROOT / "outputs/pesn/v53mp/reports"
SPLIT_DEFAULT = "split_manifest_v53mp.json"
R3_WINNER_CKPT = (
    "outputs/pesn/30h_campaign/r3_runs/5372d9b12807c3f0/checkpoint_best.pt")
SEEDS = (17, 29, 43)
TIER = "20k"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _select_pool_ids(split_doc: Any) -> list:
    """The frozen tier's pool curve ids (driver's subsets.pool rule)."""
    pool = split_doc.get("subsets", {}).get("pool")
    if isinstance(pool, dict):
        ids = pool.get(TIER)
        if ids is None and isinstance(pool.get("tiers"), dict):
            ids = pool["tiers"].get(TIER)
        _require(bool(isinstance(ids, (list, tuple))),
                 "SPLIT_SCHEMA", "subsets.pool has no {} tier".format(TIER))
        return [str(x) for x in ids]
    limit = {"2k": 2000, "5k": 5000, "20k": 20000}[TIER]
    return [str(e["curve_id"]) if isinstance(e, dict) else str(e)
            for e in pool
            if (not isinstance(e, dict)) or int(e.get("rank", 0)) <= limit]


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise SystemExit("error: {}: {}".format(code, detail))


def run_certificate(tables_dir: Path, report_dir: Path, *,
                    r3_ckpt: Path | None = None) -> dict:
    tables_dir = Path(tables_dir)
    timings: dict[str, Any] = {}
    checks: dict[str, Any] = {}

    _log("[1/6] opening v2 dataset (fast path ON) …")
    t0 = time.perf_counter()
    ds = ColumnarKeyAtomDatasetV3(
        tables_dir, label_statuses=LABEL_STATUSES, verify_manifest=False)
    timings["dataset_init_s"] = round(time.perf_counter() - t0, 2)
    samples = ds.samples
    st = ds.startup

    _log("[2/6] materializing startup columns …")
    t0 = time.perf_counter()
    _ = st._ci_arr()                 # the one-time curve_idx gather
    timings["columns_materialize_s"] = round(time.perf_counter() - t0, 3)

    split = tables_dir / SPLIT_DEFAULT
    split_doc = json.loads(split.read_text(encoding="utf-8"))
    pool_ids = _select_pool_ids(split_doc)
    dev_ids = [str(c) for c in
               split_doc.get("subsets", {}).get("dev", [])]
    _require(bool(pool_ids) and bool(dev_ids), "SPLIT_SCHEMA",
             "split manifest missing pool/dev subsets")

    # --- dev positions: vectorized vs per-item reference --------------------
    _log("[3/6] dev positions: fast vs per-item …")
    from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
        dev_positions_from_split,
    )
    t0 = time.perf_counter()
    dev_fast = dev_positions_from_split(samples, dev_ids)
    timings["dev_positions_fast_s"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    dev_ref = [i for i, s in enumerate(samples)
               if str(s.curve_id) in set(dev_ids)]
    timings["dev_positions_per_item_s"] = round(time.perf_counter() - t0, 1)
    checks["dev_positions_equal"] = {
        "passed": dev_fast == dev_ref,
        "n_fast": len(dev_fast), "n_ref": len(dev_ref),
    }
    _log("    {} positions, equal={}".format(
        len(dev_fast), checks["dev_positions_equal"]["passed"]))

    # --- per-seed: pool split + curve-dir index, fast vs per-item ----------
    seed_checks: dict[str, Any] = {}
    for seed in SEEDS:
        _log("[4/6] seed {}: pool split + curve-dir index …".format(seed))
        tr._COLUMNAR_FAST = False
        t0 = time.perf_counter()
        ref_train, ref_val = v53mp_pool_split(
            samples, pool_curve_ids=pool_ids, seed=seed)
        ref_index = _CurveDirIndex.build(samples, ref_train)
        _assert_no_confirm(samples, ref_train, "cert-train")
        _assert_no_confirm(samples, ref_val, "cert-inner-val")
        t_ref = time.perf_counter() - t0
        tr._COLUMNAR_FAST = True
        t0 = time.perf_counter()
        got_train, got_val = v53mp_pool_split(
            samples, pool_curve_ids=pool_ids, seed=seed)
        got_index = _CurveDirIndex.build(samples, got_train)
        _assert_no_confirm(samples, got_train, "cert-train")
        _assert_no_confirm(samples, got_val, "cert-inner-val")
        t_fast = time.perf_counter() - t0
        pools_equal = (
            got_index.curves == ref_index.curves
            and set(got_index.pools.keys()) == set(ref_index.pools.keys())
            and all(list(got_index.pools[k]) == list(ref_index.pools[k])
                    for k in ref_index.pools))
        seed_checks[str(seed)] = {
            "train_equal": got_train == ref_train,
            "inner_val_equal": got_val == ref_val,
            "curve_dir_index_equal": pools_equal,
            "n_train": len(got_train), "n_inner_val": len(got_val),
            "per_item_s": round(t_ref, 1),
            "fast_s": round(t_fast, 3),
            "speedup_x": round(t_ref / max(t_fast, 1e-9), 1),
        }
        _log("    seed {}: train={} val={} index={} "
             "(per-item {:.1f}s → fast {:.3f}s)".format(
                 seed, got_train == ref_train, got_val == ref_val,
                 pools_equal, t_ref, t_fast))
    checks["per_seed"] = seed_checks
    checks["all_seeds_passed"] = {
        "passed": all(c["train_equal"] and c["inner_val_equal"]
                      and c["curve_dir_index_equal"]
                      for c in seed_checks.values())}

    # --- startup breakdown with the fast path ON ---------------------------
    _log("[5/6] startup breakdown (fast path ON) …")
    config = V53MPTrainerConfig(arm="A", seed=17, backbone_init="r3")
    if r3_ckpt is not None and Path(r3_ckpt).is_file():
        t0 = time.perf_counter()
        try:
            tr.load_model_v53mp(
                config, r3_ckpt=str(r3_ckpt), device="cpu")
            timings["model_r3_load_s"] = round(time.perf_counter() - t0, 1)
        except Exception as exc:                     # noqa: BLE001
            timings["model_r3_load_s"] = None
            timings["model_r3_load_error"] = "{}: {}".format(
                type(exc).__name__, exc)
    else:
        timings["model_r3_load_s"] = None
    # first-update proxy: sampler draw + records + to_sample_v3 + collate
    from pes2ts_core.pesn.features.key_atom_graph_v3 import (
        build_sample_features_v3,
        collate_v3,
    )
    t0 = time.perf_counter()
    positions = _curve_batch_positions(
        samples, got_train, config=config,
        rng=_update_rng(17, 1, salt=0))
    recs = ds.records(positions)
    built = [build_sample_features_v3(ds.to_sample_v3(r), level="F2",
                                      intent_off=False) for r in recs]
    collate_v3(built)
    timings["first_update_proxy_s"] = round(time.perf_counter() - t0, 3)
    timings["pool_split_fast_s"] = seed_checks["17"]["fast_s"]
    timings["curve_dir_index_fast_s"] = seed_checks["17"]["fast_s"]

    tr._COLUMNAR_FAST = True
    passed = (checks["dev_positions_equal"]["passed"]
              and checks["all_seeds_passed"]["passed"])
    verdict = {
        "schema": "v53mp_startup_certificate_v1",
        "passed": bool(passed),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tables_dir": str(tables_dir),
        "tier": TIER,
        "seeds": list(SEEDS),
        "checks": checks,
        "timings": timings,
        "fast_path_env": "V53MP_DISABLE_COLUMNAR_FAST=1 forces per-item",
    }
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    short = _sha256(tables_dir / "data_manifest.json")[:12]
    report_path = report_dir / "startup_certificate_{}.json".format(short)
    report_path.write_text(
        json.dumps(verdict, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    md = ["# Startup fast-path certificate {}".format(verdict["created_at"]),
          "",
          "- passed: **{}**".format(verdict["passed"]),
          "- tables: `{}` (tier {})".format(tables_dir, TIER),
          "- seeds: {}".format(list(SEEDS)), "",
          "| seed | train==ref | val==ref | index==ref | per-item (s) | "
          "fast (s) | speedup |", "|---|---|---|---|---|---|---|"]
    for seed in SEEDS:
        c = seed_checks[str(seed)]
        md.append("| {} | {} | {} | {} | {} | {} | {}× |".format(
            seed, c["train_equal"], c["inner_val_equal"],
            c["curve_dir_index_equal"], c["per_item_s"], c["fast_s"],
            c["speedup_x"]))
    md += ["", "## Startup breakdown (fast path ON)", "",
           "```json", json.dumps(timings, indent=1, sort_keys=True), "```",
           ""]
    report_path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    _log("[6/6] report → {}".format(report_path))
    verdict["report_path"] = str(report_path)
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_v53mp_startup_certificate",
        description="Fast vs per-item startup equality + breakdown (B2)")
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--r3-ckpt", default=str(R3_WINNER_CKPT))
    args = parser.parse_args(argv)
    r3 = Path(args.r3_ckpt) if args.r3_ckpt else None
    verdict = run_certificate(Path(args.tables_dir), Path(args.report_dir),
                              r3_ckpt=r3)
    print(json.dumps(verdict, indent=1, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
