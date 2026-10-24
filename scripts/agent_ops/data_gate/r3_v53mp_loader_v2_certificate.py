#!/usr/bin/env python3
"""Loader-v2 bitwise certificate (plan §4.2; S4 lane I1 gate).

Compares the columnar loader (``ColumnarKeyAtomDatasetV3`` + its lazy label
tables) against the v1 reference (``ShardedKeyAtomDatasetV3`` + eager
``load_label_cache``) on the REAL tables, in ONE process:

1. **column equality** — v1 ``samples`` vs idx_v2 columns: EXACT count
   equality + per-field comparison over first/last 1000 rows and a ≥1%
   random sample (seed 17).  Exact fields (int/str/None/d0/fold/status/
   intent dataclass/atomic_numbers) compare bit-exactly; the two plan-§4.1
   float32-stored field groups (weights, ``target_action_u``) compare as
   ``float32(v1) == v2`` — the trainer consumes both only through an
   ``np.float32`` cast, so this is the numerically meaningful equality.
   Label tables (v52 + endpoint) compare field-exactly under the same
   sampling protocol, plus per-curve backbone (node_order/arc_s/q_canonical)
   over first/last 100 curves + 1% random.
2. **sampler equality** — fixed seed 17, 200 updates of
   ``_curve_batch_positions`` (trainer machinery imported, never copied):
   identical ``v53mp_pool_split``, identical ``_CurveDirIndex``, identical
   per-update position lists.
3. **feature equality** — 1000 random positions (seed 17):
   ``collate_v3(build_sample_features_v3(to_sample_v3(rec)))`` compared
   array-by-array (``np.array_equal`` + dtype equality) and sample_ids.
4. **verdict** — ``{passed, checks, timings, anon_rss_before/after}``;
   exit code is non-zero on ANY failure.

Report: ``outputs/pesn/v53mp/reports/loader_v2_certificate_<sha12>.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.key_atom_v3 import IntentRecordV3  # noqa: E402
from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    build_sample_features_v3,
    collate_v3,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    LABEL_STATUSES,
    ShardedKeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (  # noqa: E402
    ColumnarKeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (  # noqa: E402
    load_label_cache,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (  # noqa: E402
    FEATURE_LEVEL_V53MP,
    V53MPTrainerConfig,
    _CurveDirIndex,
    _curve_batch_positions,
    _update_rng,
    v53mp_pool_split,
)

DEFAULT_TABLES_DIR = PROJECT_ROOT / "outputs/pesn/v53mp/data"
REPORT_DIR = PROJECT_ROOT / "outputs/pesn/v53mp/reports"
SEED = 17
N_EDGE = 1000
N_UPDATES = 200
N_FEATURE_PROBES = 1000
TIER_20K_LIMIT = 20000

#: certificate sampling protocol (recorded verbatim in the report)
PROTOCOL = {
    "seed": SEED,
    "edge_rows": N_EDGE,
    "random_fraction": 0.01,
    "sampler_updates": N_UPDATES,
    "feature_probes": N_FEATURE_PROBES,
    "float32_stored_field_groups": ["weights", "target_action_u"],
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _probe_rows(n_total: int, rng: np.random.RandomState) -> np.ndarray:
    """First/last N_EDGE rows + ≥1% random rows (seed 17), sorted unique."""
    n_rand = max(int(n_total * 0.01), N_EDGE)
    n_rand = min(n_rand, n_total)
    edge = list(range(min(N_EDGE, n_total)))
    tail = list(range(max(0, n_total - N_EDGE), n_total))
    rand = [int(x) for x in rng.choice(n_total, size=n_rand, replace=False)]
    return np.asarray(sorted(set(edge + tail + rand)), dtype=np.int64)


def _intent_mismatches(a: IntentRecordV3, b: IntentRecordV3) -> list[str]:
    out: list[str] = []
    for field in ("intent_id", "curve_id", "direction",
                  "reverse_of_intent_id", "intent_pair_id",
                  "intent_version", "intent_underdetermined", "cv_kind"):
        if getattr(a, field) != getattr(b, field):
            out.append(field)
    for field in ("g_src_bonds", "g_tgt_bonds", "cv_deltas", "g_atoms",
                  "cv_coordinates"):
        va, vb = getattr(a, field), getattr(b, field)
        if len(va) != len(vb):
            out.append(field)
            continue
        for x, y in zip(va, vb):
            if isinstance(x, tuple):
                if any(float(p) != float(q) for p, q in zip(x, y)):
                    out.append(field)
                    break
            elif float(x) != float(y):
                out.append(field)
                break
    return out


def check1_columns(ds_v1: Any, ds_v2: Any, labels_v1: Any, ep_v1: Any,
                   labels_v2: Any, ep_v2: Any,
                   rng: np.random.RandomState) -> dict[str, Any]:
    """Field equality: v1 samples/labels vs idx_v2 columns + exact counts."""
    t0 = time.perf_counter()
    samples_v1 = ds_v1.samples
    samples_v2 = ds_v2.samples
    n1, n2 = len(samples_v1), len(samples_v2)
    _require_eq = n1 == n2
    rows = _probe_rows(n1, rng)
    field_fail: dict[str, int] = {}
    n_checked = 0
    for p in rows:
        a = samples_v1[int(p)]
        b = samples_v2[int(p)]
        n_checked += 1
        exact = {
            "sample_id": (a.sample_id, b.sample_id),
            "curve_id": (a.curve_id, b.curve_id),
            "node_index": (int(a.node_index), int(b.node_index)),
            "d0": (float(a.d0), float(b.d0)),
            "label_status": (a.label_status, b.label_status),
            "fold": (a.fold, b.fold),
            "target_node_index": (a.target_node_index, b.target_node_index),
            "atomic_numbers": (tuple(a.atomic_numbers),
                               tuple(b.atomic_numbers)),
        }
        for field, (va, vb) in exact.items():
            if va != vb:
                field_fail[field] = field_fail.get(field, 0) + 1
        # float32-stored groups: compare at the consumed precision
        for field in ("label_weight", "curve_weight", "sample_weight"):
            if np.float32(getattr(a, field)) != np.float32(getattr(b, field)):
                field_fail[field] = field_fail.get(field, 0) + 1
        ta, tb = a.target_action_u, b.target_action_u
        if (ta is None) != (tb is None):
            field_fail["target_action_u_None"] = \
                field_fail.get("target_action_u_None", 0) + 1
        elif ta is not None:
            aa = np.asarray(ta, dtype=np.float32)
            bb = np.asarray(tb, dtype=np.float32)
            if aa.shape != bb.shape or not np.array_equal(aa, bb):
                field_fail["target_action_u"] = \
                    field_fail.get("target_action_u", 0) + 1
        intent_diff = _intent_mismatches(a.intent, b.intent)
        for field in intent_diff:
            key = f"intent.{field}"
            field_fail[key] = field_fail.get(key, 0) + 1

    # --- label tables under the same protocol --------------------------------
    label_fail: dict[str, int] = {}

    def _eq(va: Any, vb: Any) -> bool:
        # NaN-safe: both-NaN is equality (cache stores NaN for missing q/d)
        if isinstance(va, float) and isinstance(vb, float):
            if np.isnan(va) and np.isnan(vb):
                return True
        return va == vb

    sids = [samples_v1[int(p)].sample_id for p in rows]
    for sid in sids:
        la = labels_v1.by_sample[sid]
        lb = labels_v2.by_sample[sid]
        for field in ("sample_id", "curve_id", "node_index", "intent_id",
                      "direction", "fold", "label_status", "q", "q_mask",
                      "stop", "d_key_ts"):
            va, vb = getattr(la, field), getattr(lb, field)
            if not _eq(va, vb):
                label_fail[f"v52.{field}"] = label_fail.get(f"v52.{field}", 0) + 1
        ea = ep_v1.by_sample[sid]
        eb = ep_v2.by_sample[sid]
        for field in ("sample_id", "curve_id", "node_index", "intent_id",
                      "direction", "fold", "label_status", "endpoint_label",
                      "d_end_angstrom", "d_key_ts_angstrom",
                      "bond_edit_complete", "geometry_legal", "ts_row",
                      "start_row", "terminal_row", "degenerate_curve",
                      "energy_ev_within_window"):
            if not _eq(getattr(ea, field), getattr(eb, field)):
                label_fail[f"ep.{field}"] = label_fail.get(f"ep.{field}", 0) + 1
    # membership / len surface
    surface_ok = (
        len(labels_v2.by_sample) == len(labels_v1.by_sample)
        and len(ep_v2.by_sample) == len(ep_v1.by_sample)
        and sids[0] in labels_v2.by_sample
        and sids[-1] in labels_v2.by_sample
        and ep_v2.positive_window == ep_v1.positive_window
        and ep_v2.negative_window == ep_v1.negative_window
        and ep_v2.n_rows == ep_v1.n_rows
    )

    # --- per-curve label backbone (first/last 100 + 1% curves) ---------------
    curve_ids = list(ds_v1._curves.keys())
    n_curves = len(curve_ids)
    c_rng = np.random.RandomState(SEED)
    c_probe = sorted(set(
        list(range(min(100, n_curves)))
        + list(range(max(0, n_curves - 100), n_curves))
        + [int(x) for x in c_rng.choice(
            n_curves, size=max(min(n_curves, 261), 1), replace=False)]))
    curve_fail: dict[str, int] = {}
    for c in c_probe:
        cid = curve_ids[c]
        ca = labels_v1.curves[cid]
        cb = labels_v2.curves[cid]
        if tuple(ca.node_order) != tuple(cb.node_order):
            curve_fail["node_order"] = curve_fail.get("node_order", 0) + 1
        if tuple(map(float, ca.arc_s)) != tuple(map(float, cb.arc_s)):
            curve_fail["arc_s"] = curve_fail.get("arc_s", 0) + 1
        if (ca.q_canonical is None) != (cb.q_canonical is None):
            curve_fail["q_canonical_None"] = \
                curve_fail.get("q_canonical_None", 0) + 1
        elif ca.q_canonical is not None and \
                tuple(map(float, ca.q_canonical)) != \
                tuple(map(float, cb.q_canonical)):
            curve_fail["q_canonical"] = curve_fail.get("q_canonical", 0) + 1
        if (ca.ts_index != cb.ts_index
                or ca.q_zero_arc_reason != cb.q_zero_arc_reason):
            curve_fail["meta"] = curve_fail.get("meta", 0) + 1

    passed = (_require_eq and not field_fail and not label_fail
              and not curve_fail and surface_ok)
    return {
        "passed": bool(passed),
        "n_samples_v1": n1,
        "n_samples_v2": n2,
        "count_equal": bool(_require_eq),
        "n_rows_checked": int(n_checked),
        "sample_field_failures": field_fail,
        "label_field_failures": label_fail,
        "curve_failures": curve_fail,
        "label_surface_ok": bool(surface_ok),
        "curves_checked": len(c_probe),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def check2_sampler(ds_v1: Any, ds_v2: Any, pool_curve_ids: list[str]) -> dict[str, Any]:
    """200 updates of _curve_batch_positions: positional equality."""
    t0 = time.perf_counter()
    samples_v1, samples_v2 = ds_v1.samples, ds_v2.samples
    train1, val1 = v53mp_pool_split(
        samples_v1, pool_curve_ids=pool_curve_ids, seed=SEED)
    train2, val2 = v53mp_pool_split(
        samples_v2, pool_curve_ids=pool_curve_ids, seed=SEED)
    split_equal = train1 == train2 and val1 == val2
    index1 = _CurveDirIndex.build(samples_v1, train1)
    index2 = _CurveDirIndex.build(samples_v2, train2)
    index_equal = (index1.curves == index2.curves
                   and index1.pools.keys() == index2.pools.keys()
                   and all(index1.pools[k] == index2.pools[k]
                           for k in index1.pools))
    config = V53MPTrainerConfig(arm="A", seed=SEED, backbone_init="r3")
    mismatch = 0
    first_bad = None
    for update in range(1, N_UPDATES + 1):
        b1 = _curve_batch_positions(
            samples_v1, train1, config=config,
            rng=_update_rng(SEED, update, salt=0), index=index1)
        b2 = _curve_batch_positions(
            samples_v2, train2, config=config,
            rng=_update_rng(SEED, update, salt=0), index=index2)
        if b1 != b2:
            mismatch += 1
            if first_bad is None:
                first_bad = {"update": update, "v1_head": b1[:8],
                             "v2_head": b2[:8]}
    return {
        "passed": bool(split_equal and index_equal and mismatch == 0),
        "pool_curves": len(pool_curve_ids),
        "train_positions": len(train1),
        "inner_val_positions": len(val1),
        "split_equal": bool(split_equal),
        "curve_dir_index_equal": bool(index_equal),
        "updates": N_UPDATES,
        "position_mismatches": int(mismatch),
        "first_mismatch": first_bad,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def check3_features(ds_v1: Any, ds_v2: Any,
                    rng: np.random.RandomState) -> dict[str, Any]:
    """1000 random positions: collated feature arrays bit-identical."""
    t0 = time.perf_counter()
    n = len(ds_v1.samples)
    positions = sorted(int(x) for x in rng.choice(
        n, size=min(N_FEATURE_PROBES, n), replace=False))
    array_fail: dict[str, int] = {}
    dtype_fail: dict[str, int] = {}
    n_compared = 0
    for p in positions:
        recs = ds_v1.records([p])
        recs2 = ds_v2.records([p])
        s1 = ds_v1.to_sample_v3(recs[0])
        s2 = ds_v2.to_sample_v3(recs2[0])
        b1 = collate_v3([build_sample_features_v3(
            s1, level=FEATURE_LEVEL_V53MP, intent_off=False)])
        b2 = collate_v3([build_sample_features_v3(
            s2, level=FEATURE_LEVEL_V53MP, intent_off=False)])
        if tuple(b1["sample_ids"]) != tuple(b2["sample_ids"]):
            array_fail["sample_ids"] = array_fail.get("sample_ids", 0) + 1
        for key in sorted(set(b1) | set(b2)):
            va, vb = b1.get(key), b2.get(key)
            if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
                if va.dtype != vb.dtype:
                    dtype_fail[key] = dtype_fail.get(key, 0) + 1
                if not np.array_equal(va, vb):
                    array_fail[key] = array_fail.get(key, 0) + 1
            elif isinstance(va, tuple) and isinstance(vb, tuple):
                if va != vb:
                    array_fail[key] = array_fail.get(key, 0) + 1
            elif key != "sample_ids" and va != vb:
                array_fail[key] = array_fail.get(key, 0) + 1
        n_compared += 1
    return {
        "passed": bool(not array_fail and not dtype_fail),
        "positions": n_compared,
        "array_failures": array_fail,
        "dtype_failures": dtype_fail,
        "feature_level": FEATURE_LEVEL_V53MP,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def run_certificate(tables_dir: Path, report_dir: Path) -> dict[str, Any]:
    tables_dir = Path(tables_dir)
    report_dir = Path(report_dir)
    rng = np.random.RandomState(SEED)
    timings: dict[str, Any] = {}
    checks: dict[str, Any] = {}

    rss_start = _rss_kb()
    _log("[cert] constructing v1 dataset (light cache) …")
    t0 = time.perf_counter()
    # verify_manifest=True is REQUIRED for the light-cache fast path: the
    # cache binding includes the manifest sha, which is only populated when
    # verification runs (otherwise v1 falls back to the ~10 min CSV scan).
    ds_v1 = ShardedKeyAtomDatasetV3(
        tables_dir, label_statuses=LABEL_STATUSES, light_cache=True,
        verify_manifest=True)
    timings["v1_dataset_init_s"] = round(time.perf_counter() - t0, 1)
    _log(f"    done in {timings['v1_dataset_init_s']}s "
         f"(rss_max={_rss_kb() // 1024} MB)")
    _log("[cert] loading v1 label cache (eager) …")
    t0 = time.perf_counter()
    # verify_manifest=False leaves dataset.manifest_sha256 empty; the label
    # cache binds to the manifest file's OWN sha, so compute it directly
    # (the file is small — not the multi-GB table re-hash).
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import _sha256
    manifest_sha = _sha256(tables_dir / "data_manifest.json")
    labels_v1, ep_v1 = load_label_cache(
        tables_dir, dataset_manifest_sha256=manifest_sha)
    timings["v1_label_load_s"] = round(time.perf_counter() - t0, 1)
    _log(f"    done in {timings['v1_label_load_s']}s "
         f"(rss_max={_rss_kb() // 1024} MB)")

    _log("[cert] constructing v2 columnar dataset …")
    t0 = time.perf_counter()
    ds_v2 = ColumnarKeyAtomDatasetV3(
        tables_dir, label_statuses=LABEL_STATUSES, verify_manifest=False)
    timings["v2_dataset_init_s"] = round(ds_v2.init_seconds, 1)
    timings["v2_init_total_s"] = round(time.perf_counter() - t0, 1)
    timings["v2_idx_verify_s"] = round(ds_v2.idx_verify_seconds, 1)
    _log(f"    done in {timings['v2_init_total_s']}s "
         f"(idx verify {timings['v2_idx_verify_s']}s, "
         f"rss_max={_rss_kb() // 1024} MB)")
    _log("[cert] opening v2 lazy label tables …")
    t0 = time.perf_counter()
    labels_v2, ep_v2 = ds_v2.label_tables()
    timings["v2_label_open_s"] = round(time.perf_counter() - t0, 1)

    _log("[cert] check 1: column + label field equality …")
    checks["column_equality"] = check1_columns(
        ds_v1, ds_v2, labels_v1, ep_v1, labels_v2, ep_v2, rng)
    _log(f"    passed={checks['column_equality']['passed']} "
         f"rows={checks['column_equality']['n_rows_checked']}")

    # pool curve ids: the frozen 20k tier of the split manifest — mirror the
    # driver's _tier_pool_curve_ids logic (list-ranked OR tier-mapping shapes)
    split_path = tables_dir / "split_manifest_v53mp.json"
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    pool = split_doc["subsets"]["pool"]
    if isinstance(pool, dict):
        ids = pool.get("20k")
        if ids is None and isinstance(pool.get("tiers"), dict):
            ids = pool["tiers"].get("20k")
        if not isinstance(ids, (list, tuple)):
            raise RuntimeError("split manifest subsets.pool has no 20k tier")
        pool_curve_ids = [str(x) for x in ids]
    else:
        pool_curve_ids = [
            str(e["curve_id"]) if isinstance(e, dict) else str(e)
            for e in pool
            if (not isinstance(e, dict))
            or int(e.get("rank", 0)) <= TIER_20K_LIMIT
        ]
    _log("[cert] check 2: 200-update sampler positional equality …")
    checks["sampler_equality"] = check2_sampler(ds_v1, ds_v2, pool_curve_ids)
    _log(f"    passed={checks['sampler_equality']['passed']} "
         f"train={checks['sampler_equality']['train_positions']}")

    _log("[cert] check 3: 1000-position collated feature equality …")
    checks["feature_equality"] = check3_features(ds_v1, ds_v2, rng)
    _log(f"    passed={checks['feature_equality']['passed']} "
         f"positions={checks['feature_equality']['positions']}")

    # free v1 before the verdict block (v2 must survive alone)
    del ds_v1, labels_v1, ep_v1
    gc.collect()
    rss_end = _rss_kb()

    idx_sha = ""
    curves_json = tables_dir / "idx_v2" / "curves.json"
    if curves_json.is_file():
        from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import _sha256
        idx_sha = _sha256(curves_json)

    passed = all(c["passed"] for c in checks.values())
    verdict = {
        "schema": "v53mp_loader_v2_certificate_v1",
        "passed": bool(passed),
        "created_at": _utc_now(),
        "tables_dir": str(tables_dir),
        "idx_curves_json_sha256": idx_sha,
        "protocol": PROTOCOL,
        "checks": checks,
        "timings": timings,
        "anon_rss_before_kb": rss_start,
        "anon_rss_after_kb": rss_end,
        "anon_rss_before_mib": rss_start // 1024,
        "anon_rss_after_mib": rss_end // 1024,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    short = idx_sha[:12] if idx_sha else "nosha"
    report_path = report_dir / f"loader_v2_certificate_{short}.json"
    report_path.write_text(
        json.dumps(verdict, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    verdict["report_path"] = str(report_path)
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_v53mp_loader_v2_certificate",
        description="Bitwise A/B certificate: loader v2 vs loader v1 (§4.2)")
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args(argv)
    verdict = run_certificate(Path(args.tables_dir), Path(args.report_dir))
    print(json.dumps(verdict, indent=1, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
