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
import hashlib
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
#: default tier = the frozen 20k build (byte-identical default behaviour);
#: ``--tier 50k`` certifies a 50k build (list-shaped pool fallback limit).
DEFAULT_TIER = "20k"
TIER_LIMITS = {"2k": 2000, "5k": 5000, "20k": 20000, "50k": 50000}

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


def _select_pool_curve_ids(split_doc: Any,
                           tier: str = DEFAULT_TIER) -> list[str]:
    """The frozen tier's pool curve ids (driver's _tier_pool_curve_ids rule:
    list-ranked OR tier-mapping shapes), parameterized by ``tier``.
    """
    if tier not in TIER_LIMITS:
        raise RuntimeError(
            "unknown tier {!r} (expected one of {})".format(
                tier, sorted(TIER_LIMITS)))
    pool = split_doc["subsets"]["pool"]
    if isinstance(pool, dict):
        ids = pool.get(tier)
        if ids is None and isinstance(pool.get("tiers"), dict):
            ids = pool["tiers"].get(tier)
        if not isinstance(ids, (list, tuple)):
            raise RuntimeError(
                "split manifest subsets.pool has no {} tier".format(tier))
        return [str(x) for x in ids]
    return [
        str(e["curve_id"]) if isinstance(e, dict) else str(e)
        for e in pool
        if (not isinstance(e, dict))
        or int(e.get("rank", 0)) <= TIER_LIMITS[tier]
    ]


# ---------------------------------------------------------------------------
# WP5-CERT streaming mode (V5.4.1 §8.1 补证优先规则; prereg wp5_cert_…)
# ---------------------------------------------------------------------------
STREAM_REPORT_PATH = (PROJECT_ROOT / "outputs/pesn/v541_embed_50k/audits/"
                      "loader_v2_streaming_certificate.json")
STREAM_PREREG_PATH = (PROJECT_ROOT / "outputs/pesn/v541_embed_50k/audits/"
                      "wp5_cert_preregistration.json")
#: files whose sha256 is verified against data_manifest.json / idx binding
STREAM_HASH_FILES = ("sample_view.csv", "irc_node.csv", "irc_curve.csv",
                     "reaction_intent.csv", "irc_edge.csv",
                     "curve_quality.csv", "shard_index_v53mp.json")


def _vm_rss_kb() -> int:
    """Current VmRSS (not the high-water mark) from /proc/self/status."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _phase_stats(name: str, t0: float, rss_before_kb: int) -> dict[str, Any]:
    return {
        "phase": name,
        "wall_s": round(time.perf_counter() - t0, 1),
        "rss_before_kb": rss_before_kb,
        "rss_after_kb": _vm_rss_kb(),
        "ru_maxrss_kb": _rss_kb(),
    }


class StreamCertError(RuntimeError):
    """Typed refusal of the streaming certificate."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _stream_require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise StreamCertError(code, detail)


def _pool_probe_rows(pool_positions: np.ndarray,
                     rng: np.random.RandomState) -> np.ndarray:
    """First/last N_EDGE + ≥1% random rows OVER THE POOL ONLY (seed 17).

    Reuses the exact ``_probe_rows`` protocol but applied to the pool
    position subsequence, so first/last/random all filter the pool before
    selection — dev/confirm positions can never enter any probe.
    """
    n_pool = int(pool_positions.shape[0])
    local = _probe_rows(n_pool, rng)
    return np.asarray(pool_positions[local], dtype=np.int64)


def _verify_stream_hashes(tables_dir: Path, split_path: Path,
                          binding: Mapping[str, Any]) -> dict[str, Any]:
    """sha256 of every consumed table vs data_manifest.json (+ idx binding
    for the non-manifest artifacts).  Fail-closed, streamed hashing."""
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import _sha256

    manifest_path = tables_dir / "data_manifest.json"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = doc.get("files", {})
    entries = ({str(e["path"]): str(e["sha256"]) for e in files}
               if isinstance(files, list)
               else {str(k): str(v) for k, v in files.items()})
    manifest_sha = _sha256(manifest_path)
    out: dict[str, Any] = {
        "data_manifest_sha256": manifest_sha,
        "manifest_sha_matches_idx_binding":
            manifest_sha == str(binding.get("data_manifest_sha256", "")),
        "files": {},
    }
    _stream_require(out["manifest_sha_matches_idx_binding"], "EVIDENCE_MISSING",
                    "data_manifest.json sha drifted from idx_v2 binding")
    for name in STREAM_HASH_FILES:
        path = tables_dir / name
        _stream_require(path.is_file(), "EVIDENCE_MISSING", f"{name} missing")
        actual = _sha256(path)
        expected = entries.get(name)
        ok = expected is None or actual == expected
        out["files"][name] = {
            "sha256": actual, "manifest_sha256": expected, "match": ok}
        _stream_require(ok, "EVIDENCE_MISSING",
                        f"{name}: sha256 {actual[:12]}… != manifest "
                        f"{str(expected)[:12]}…")
    # split manifest + label-cache binding (not listed in data_manifest)
    split_sha = _sha256(split_path)
    out["split_manifest_sha256"] = split_sha
    out["split_matches_idx_binding"] = (
        split_sha == str(binding.get("split_manifest_sha256", "")))
    _stream_require(out["split_matches_idx_binding"], "EVIDENCE_MISSING",
                    "split_manifest sha drifted from idx_v2 binding")
    # label-cache npz/json: sha vs the idx binding's recorded build-time sha
    for key, name in (("source_labels_npz", "labels_cache_v53mp.npz"),
                      ("source_labels_json", "labels_cache_v53mp.json")):
        entry = binding.get(key)
        path = tables_dir / name
        _stream_require(path.is_file(), "EVIDENCE_MISSING", f"{name} missing")
        actual = _sha256(path)
        expected = str((entry or {}).get("sha256", ""))
        ok = (not expected) or actual == expected
        out[f"{name}_matches_idx_binding"] = ok
        out[f"{name}_sha256"] = actual
        _stream_require(ok, "EVIDENCE_MISSING",
                        f"{name}: sha256 {actual[:12]}… != idx binding "
                        f"{expected[:12]}…")
    lc_json = tables_dir / "labels_cache_v53mp.json"
    lc_doc = json.loads(lc_json.read_text(encoding="utf-8"))
    out["label_cache_binding_matches_manifest"] = (
        str(lc_doc.get("dataset_manifest_sha256")) == manifest_sha)
    _stream_require(out["label_cache_binding_matches_manifest"],
                    "EVIDENCE_MISSING",
                    "label cache is bound to a different data_manifest sha")
    out["label_cache_n_samples"] = int(lc_doc.get("n_samples", -1))
    out["label_cache_endpoint_windows"] = lc_doc.get("endpoint_windows", {})
    return out


def _record_field_mismatches(a: Any, b: Any) -> dict[str, int]:
    """Per-field comparison of two SampleRecordV3Lite-shaped records
    (the check1 exact/float32 semantics, factorised for reuse)."""
    fail: dict[str, int] = {}

    def _bump(key: str) -> None:
        fail[key] = fail.get(key, 0) + 1

    exact = {
        "sample_id": (a.sample_id, b.sample_id),
        "curve_id": (a.curve_id, b.curve_id),
        "node_index": (int(a.node_index), int(b.node_index)),
        "d0": (float(a.d0), float(b.d0)),
        "label_status": (a.label_status, b.label_status),
        "fold": (a.fold, b.fold),
        "target_node_index": (a.target_node_index, b.target_node_index),
        "atomic_numbers": (tuple(a.atomic_numbers), tuple(b.atomic_numbers)),
    }
    for field, (va, vb) in exact.items():
        if va != vb:
            _bump(field)
    for field in ("label_weight", "curve_weight", "sample_weight"):
        if np.float32(getattr(a, field)) != np.float32(getattr(b, field)):
            _bump(field)
    ta, tb = a.target_action_u, b.target_action_u
    if (ta is None) != (tb is None):
        _bump("target_action_u_None")
    elif ta is not None:
        aa = np.asarray(ta, dtype=np.float32)
        bb = np.asarray(tb, dtype=np.float32)
        if aa.shape != bb.shape or not np.array_equal(aa, bb):
            _bump("target_action_u")
    for field in _intent_mismatches(a.intent, b.intent):
        _bump(f"intent.{field}")
    return fail


def _stream_check_columns(view: Any, ds_v2: Any, columns: Any,
                          probe_positions: np.ndarray,
                          rng: np.random.RandomState,
                          ) -> tuple[dict[str, Any], dict[int, Any]]:
    """Phase 1a — compact stream vs idx_v2 columns over pool-only probes."""
    t0 = time.perf_counter()
    n1 = int(columns.n_rows)
    n2 = len(ds_v2)
    count_equal = n1 == n2
    records = view.probe_records(probe_positions, columns)
    field_fail: dict[str, int] = {}
    n_checked = 0
    for p in probe_positions:
        rec_v1 = records[int(p)]
        rec_v2 = ds_v2.samples[int(p)]
        n_checked += 1
        for key, cnt in _record_field_mismatches(rec_v1, rec_v2).items():
            field_fail[key] = field_fail.get(key, 0) + cnt
    passed = count_equal and not field_fail
    return {
        "passed": bool(passed),
        "n_samples_v1_stream": n1,
        "n_samples_v2": n2,
        "count_equal": bool(count_equal),
        "n_rows_checked": int(n_checked),
        "probe_positions_pool_only": True,
        "sample_field_failures": field_fail,
        "weight_curves_checked": int(columns.weight_curves_checked),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }, records


def _sid_rows(ids: np.ndarray, probe_sids: Sequence[str]) -> np.ndarray:
    """Map each probe sid to its row in the label-cache ``sample_ids``
    column, INDEPENDENT of the column's sort order.

    The 20k-build cache happens to be lexicographically sorted, but the
    delivered 50k-build cache is in curve/node numeric order (``:9:``
    precedes ``:10:``), so ``np.searchsorted`` is NOT valid here.  We
    stream the column in chunks and match only the probe sids (the probe
    set is ~1% of rows — bounded memory either way).  Result is aligned
    with ``probe_sids`` order; a missing sid is a typed refusal.
    """
    want = {str(s): i for i, s in enumerate(probe_sids)}
    _stream_require(len(want) == len(probe_sids), "EVIDENCE_MISSING",
                    "duplicate probe sids supplied to _sid_rows")
    rows: list[int | None] = [None] * len(probe_sids)
    n = int(ids.shape[0])
    chunk = 1 << 20
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        for j, sid in enumerate(ids[start:stop].tolist()):
            k = want.get(str(sid))
            if k is not None:
                rows[k] = start + j
        if all(r is not None for r in rows):
            break
    missing = [s for s, r in zip(probe_sids, rows) if r is None]
    if missing:
        raise StreamCertError(
            "EVIDENCE_MISSING",
            f"probe sid {missing[0]} not found in label cache")
    return np.asarray(rows, dtype=np.int64)


def _stream_check_labels(ds_v2: Any, labels_v2: Any, ep_v2: Any,
                         npz: Any, lc_doc: Mapping[str, Any],
                         probe_sids: list[str],
                         sid_to_pos: Mapping[str, int],
                         probe_curve_ids: list[str],
                         records: Mapping[int, Any],
                         rng: np.random.RandomState) -> dict[str, Any]:
    """Phase 1b — label-cache memmap columns vs v2 label memmaps, per probe
    row (no by_sample dict ever materialised on the v1 side)."""
    t0 = time.perf_counter()
    ids = npz.column("sample_ids")
    _stream_require(int(ids.shape[0]) == int(lc_doc["n_samples"]),
                    "EVIDENCE_MISSING", "label cache n_samples drift")
    # order-independent sid→row mapping (the 50k cache is NOT sorted by
    # sid; see _sid_rows) — fail-closed on any missing probe sid
    v1_rows_arr = _sid_rows(ids, probe_sids)
    pos_arr = np.asarray([sid_to_pos[s] for s in probe_sids], dtype=np.int64)

    label_fail: dict[str, int] = {}

    def _eq_col(name_v1: str, field: str, v2_col: Any,
                transform: Any = None) -> None:
        a = npz.column(name_v1)[v1_rows_arr]
        b = np.asarray(v2_col[pos_arr])
        if transform is not None:
            a = transform(a)
            b = transform(b)
        if a.dtype.kind == "f":
            bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        else:
            bad = a != b
        n_bad = int(np.count_nonzero(bad))
        if n_bad:
            label_fail[field] = label_fail.get(field, 0) + n_bad

    _eq_col("q", "v52.q", labels_v2.q_col)
    _eq_col("q_mask", "v52.q_mask", labels_v2.q_mask_col)
    _eq_col("stop", "v52.stop", labels_v2._stop)
    _eq_col("d_key_ts", "v52.d_key_ts", labels_v2._d_key_ts)
    _eq_col("ep_label", "ep.endpoint_label", ep_v2.ep_label_col)
    _eq_col("d_end", "ep.d_end_angstrom", ep_v2._d_end)
    _eq_col("d_key_ts_ep", "ep.d_key_ts_angstrom", ep_v2._d_key_ts_ep)
    _eq_col("bond_edit_complete", "ep.bond_edit_complete", ep_v2._bond_edit)
    _eq_col("geometry_legal", "ep.geometry_legal", ep_v2._geom_legal)
    _eq_col("ts_row", "ep.ts_row", ep_v2._ts_row)
    _eq_col("start_row", "ep.start_row", ep_v2._start_row)
    _eq_col("terminal_row", "ep.terminal_row", ep_v2._terminal_row)
    _eq_col("degenerate_curve", "ep.degenerate_curve", ep_v2._degenerate)
    _eq_col("energy_ev_within_window", "ep.energy_ev_within_window",
            ep_v2._energy_ev)

    # identity columns: v1 npz at the cache row vs the v1 probe record
    # (check 1 already tied those records to the v2 columns bit-exactly,
    # so this closes v1-cache ↔ v2-label-identity without residency)
    ident_fail: dict[str, int] = {}

    def _ident_bad(name_v1: str, field: str, getter: Any) -> None:
        col = npz.column(name_v1)[v1_rows_arr]
        n_bad = 0
        for i, row in enumerate(v1_rows_arr):
            rec = records[int(pos_arr[i])]
            if not getter(str(col[i]), rec):
                n_bad += 1
        if n_bad:
            ident_fail[field] = ident_fail.get(field, 0) + n_bad

    _ident_bad("curve_of", "curve_id", lambda v, r: v == str(r.curve_id))
    _ident_bad("node_index", "node_index",
               lambda v, r: int(v) == int(r.node_index))
    _ident_bad("intent_of", "intent_id",
               lambda v, r: v == str(r.intent.intent_id))
    _ident_bad("direction", "direction",
               lambda v, r: v == str(r.intent.direction))
    _ident_bad("fold", "fold",
               lambda v, r: ((None if int(v) < 0 else int(v)) == r.fold))
    _ident_bad("label_status", "label_status",
               lambda v, r: v == str(r.label_status))
    label_fail.update({f"id.{k}": v for k, v in ident_fail.items()})

    # per-curve label backbone over pool-only curve probes
    curve_fail: dict[str, int] = {}
    meta = {str(m["curve_id"]): m for m in lc_doc.get("curve_meta", [])}
    node_order = npz.column("node_order")
    arc_s = npz.column("arc_s")
    q_canon = npz.column("q_canonical")
    off = 0
    off_q = 0
    offsets: dict[str, tuple[int, int, bool]] = {}
    for m in lc_doc.get("curve_meta", []):
        k = int(m["n"])
        has_q = not bool(m["q_zero_arc_reason"])
        offsets[str(m["curve_id"])] = (off, off_q, has_q)
        off += k
        if has_q:
            off_q += k
    for cid in probe_curve_ids:
        start, q_start, has_q = offsets[cid]
        k = int(meta[cid]["n"])
        cb = labels_v2.curves[cid]
        if tuple(int(v) for v in node_order[start:start + k]) != \
                tuple(int(v) for v in cb.node_order):
            curve_fail["node_order"] = curve_fail.get("node_order", 0) + 1
        if tuple(map(float, arc_s[start:start + k])) != \
                tuple(map(float, cb.arc_s)):
            curve_fail["arc_s"] = curve_fail.get("arc_s", 0) + 1
        if has_q != (cb.q_canonical is not None):
            curve_fail["q_canonical_None"] = \
                curve_fail.get("q_canonical_None", 0) + 1
        elif has_q and \
                tuple(map(float, q_canon[q_start:q_start + k])) != \
                tuple(map(float, cb.q_canonical)):
            curve_fail["q_canonical"] = curve_fail.get("q_canonical", 0) + 1
        if int(meta[cid]["ts_index"]) != int(cb.ts_index) or \
                str(meta[cid]["q_zero_arc_reason"]) != str(cb.q_zero_arc_reason):
            curve_fail["meta"] = curve_fail.get("meta", 0) + 1

    # structural surface (cheap, memmap-backed)
    surface_ok = (
        ep_v2.positive_window
        == float(lc_doc["endpoint_windows"]["positive_window_angstrom"])
        and ep_v2.negative_window
        == float(lc_doc["endpoint_windows"]["negative_window_angstrom"])
        and ep_v2.n_rows == len(ds_v2)
    )
    # v2 label rows must cover every probe sid (lazy lookup)
    coverage_ok = all(s in labels_v2.by_sample and s in ep_v2.by_sample
                      for s in probe_sids)
    npz.release("sample_ids")
    passed = (not label_fail and not curve_fail and surface_ok
              and coverage_ok)
    return {
        "passed": bool(passed),
        "n_probe_sids": len(probe_sids),
        "n_probe_curves": len(probe_curve_ids),
        "label_field_failures": label_fail,
        "curve_failures": curve_fail,
        "label_surface_ok": bool(surface_ok),
        "v2_by_sample_coverage_ok": bool(coverage_ok),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def _stream_check_sampler(view: Any, columns: Any, ds_v2: Any,
                          pool_curve_ids: list[str]) -> dict[str, Any]:
    """Phase 2 — compact per-item v1 view vs v2 fast path (200 updates)."""
    t0 = time.perf_counter()
    samples_v1 = view.compact_view(columns)
    samples_v2 = ds_v2.samples
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
        "sampler_id": "v53mp_curve_dir_v1",
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


def _stream_check_features(view: Any, ds_v2: Any,
                           records: Mapping[int, Any],
                           feature_positions: np.ndarray) -> dict[str, Any]:
    """Phase 3 — collated features over pool-only probes (v1 records from
    the streaming probe pass + bounded shard-LRU geometry vs v2 geom-pack)."""
    from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: F401
        build_sample_features_v3, collate_v3,
    )
    t0 = time.perf_counter()
    array_fail: dict[str, int] = {}
    dtype_fail: dict[str, int] = {}
    n_compared = 0
    for p in feature_positions:
        p = int(p)
        rec_v1 = records[p]
        s1 = view.to_sample_v3(rec_v1)
        s2 = ds_v2.to_sample_v3(ds_v2.samples[p])
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
        "probe_positions_pool_only": True,
        "array_failures": array_fail,
        "dtype_failures": dtype_fail,
        "feature_level": FEATURE_LEVEL_V53MP,
        "geometry_lru_max_curves": int(view.cache_curves),
        "geometry_lru_peak_curves": int(view.frames_lru_size()),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def run_certificate_streaming(
    tables_dir: Path,
    report_path: Path = STREAM_REPORT_PATH,
    *,
    tier: str = "50k",
    prereg_path: Path = STREAM_PREREG_PATH,
    cgroup_evidence_dir: str = "",
) -> dict[str, Any]:
    """WP5-CERT one-shot streaming certificate (prereg wp5_cert_prereg…).

    Phases (serial, transient state freed between phases): columns →
    sampler → features.  v1 is NEVER fully resident; labels are compared
    per probe row via memmapped NumPy columns; the sampler uses the compact
    curve/direction view; features materialise probe records only with a
    bounded shard-LRU geometry cache.  All probes come from the 50k pool
    only.  Any typed failure writes passed=false and stops (no probe
    redefinition, no retry — attempt 1 of 1).
    """
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        ColumnarKeyAtomDatasetV3,
    )
    from pes2ts_core.pesn.offline.data.key_atom_stream_view import (
        NpzColumnMemmaps, StreamThinViewError, StreamThinViewV3,
    )

    tables_dir = Path(tables_dir)
    report_path = Path(report_path)
    prereg_path = Path(prereg_path)
    rng = np.random.RandomState(SEED)
    timings: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    phases: list[dict[str, Any]] = []
    rss_start = _rss_kb()
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    verdict: dict[str, Any] = {
        "schema": "v53mp_loader_v2_streaming_certificate_v1",
        "mode": "streaming",
        "workpackage": "WP5-CERT",
        "attempt": 1,
        "sampler_id": "v53mp_curve_dir_v1",
        "created_at": _utc_now(),
        "tables_dir": str(tables_dir),
        "tier": tier,
        "preregistration": str(prereg_path),
        "cgroup_evidence_dir": cgroup_evidence_dir or None,
        "protocol": PROTOCOL,
    }
    try:
        # --- prereg binding (fail-closed) -----------------------------------
        _stream_require(prereg_path.is_file(), "PREREG_MISMATCH",
                        f"{prereg_path} missing")
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        _stream_require(prereg.get("mode") == "streaming_phased",
                        "PREREG_MISMATCH", "prereg mode drift")
        _stream_require(prereg.get("tier") == tier, "PREREG_MISMATCH",
                        "prereg tier drift")
        _stream_require(prereg.get("attempt") == 1, "PREREG_MISMATCH",
                        "prereg attempt drift")
        proto = prereg.get("probe_protocol", {})
        _stream_require(int(proto.get("seed", -1)) == SEED,
                        "PREREG_MISMATCH", "prereg seed drift")
        _stream_require(bool(proto.get("pool_only")) is True,
                        "PREREG_MISMATCH", "prereg pool_only drift")
        _stream_require(int(proto.get("sampler_updates", -1)) == N_UPDATES,
                        "PREREG_MISMATCH", "prereg sampler_updates drift")
        _stream_require(int(proto.get("feature_probes", -1))
                        == N_FEATURE_PROBES, "PREREG_MISMATCH",
                        "prereg feature_probes drift")
        cg = prereg.get("cgroup", {})
        _stream_require(cg.get("memory_max") == "12G"
                        and cg.get("memory_high") == "10G"
                        and int(cg.get("swap_max", -1)) == 0,
                        "PREREG_MISMATCH", "prereg cgroup limits drift")
        _stream_require(
            Path(str(prereg.get("outputs", {}).get("certificate", "")))
            .name == report_path.name, "PREREG_MISMATCH",
            "prereg certificate output drift")
        verdict["preregistration_sha256"] = hashlib.sha256(
            prereg_path.read_bytes()).hexdigest()

        # --- idx_v2 (v2 memmap side) ----------------------------------------
        t0 = time.perf_counter()
        rss_before = _vm_rss_kb()
        _log("[stream-cert] opening v2 columnar dataset (memmap) …")
        ds_v2 = ColumnarKeyAtomDatasetV3(
            tables_dir, label_statuses=LABEL_STATUSES, verify_manifest=False)
        labels_v2, ep_v2 = ds_v2.label_tables()
        binding = ds_v2.idx_binding
        timings["v2_dataset_init_s"] = round(ds_v2.init_seconds, 1)
        phases.append(_phase_stats("v2_open", t0, rss_before))
        _log(f"    v2 open in {timings['v2_dataset_init_s']}s "
             f"(rss={_vm_rss_kb() // 1024} MB)")

        # --- hash verification of consumed files ----------------------------
        t0 = time.perf_counter()
        rss_before = _vm_rss_kb()
        _log("[stream-cert] verifying consumed-file hashes …")
        split_path = tables_dir / "split_manifest_v53mp.json"
        hashes = _verify_stream_hashes(tables_dir, split_path, binding)
        phases.append(_phase_stats("hash_verify", t0, rss_before))
        _log(f"    {len(hashes['files'])} file hashes verified "
             f"({phases[-1]['wall_s']}s)")

        # --- pool curve ids (pool-only probes by construction) ---------------
        split_doc = json.loads(split_path.read_text(encoding="utf-8"))
        pool_curve_ids = _select_pool_curve_ids(split_doc, tier=tier)
        dev_ids = {str(x) for x in split_doc["subsets"].get("dev", [])}
        confirm_ids = {str(x) for x in
                       split_doc["subsets"].get("confirm", [])}
        _stream_require(not (set(pool_curve_ids) & dev_ids)
                        and not (set(pool_curve_ids) & confirm_ids),
                        "EVIDENCE_MISSING",
                        "pool curves intersect dev/confirm subsets")
        _log(f"    pool[{tier}] = {len(pool_curve_ids)} curves")

        # --- Phase 1: columns (streaming thin view) -------------------------
        t0 = time.perf_counter()
        rss_before = _vm_rss_kb()
        _log("[stream-cert] phase 1a: building v1 streaming thin view …")
        # verify_manifest=True on the VIEW only re-hashes the ~1.8 GB of
        # geometry shards (fail-closed); the 55 GB data-manifest chain is
        # covered by the hash_verify phase instead.
        view = StreamThinViewV3(
            tables_dir, label_statuses=LABEL_STATUSES,
            verify_manifest=True, cache_curves=64)
        timings["v1_view_init_s"] = view.init_seconds
        _log(f"    view init {view.init_seconds}s "
             f"(rss={_vm_rss_kb() // 1024} MB)")
        _log("[stream-cert] phase 1a: streaming sample_view.csv pass 1 "
             "(compact columns) …")
        columns = view.stream_columns()
        timings["v1_stream_columns_s"] = columns.elapsed_s
        _log(f"    {columns.n_rows} surviving rows "
             f"({columns.elapsed_s}s, rss={_vm_rss_kb() // 1024} MB)")

        # pool positions from the compact columns (curve membership only)
        curve_row = {cid: i for i, cid in enumerate(view._curves.keys())}
        pool_rows = np.asarray([curve_row[c] for c in pool_curve_ids
                                if c in curve_row], dtype=np.int64)
        _stream_require(pool_rows.size > 0, "EVIDENCE_MISSING",
                        "no pool curves in the streamed tables")
        is_pool = np.isin(columns.curve_idx, pool_rows)
        pool_positions = np.flatnonzero(is_pool).astype(np.int64)
        _stream_require(int(pool_positions.shape[0]) > 0, "EVIDENCE_MISSING",
                        "no pool positions in the streamed tables")
        # exclude dev/confirm positions explicitly (belt and braces)
        nonpool_curve_rows = np.asarray(
            [i for i, cid in enumerate(view._curves.keys())
             if cid not in set(pool_curve_ids)], dtype=np.int64)
        if nonpool_curve_rows.size:
            n_nonpool_probed = int(np.count_nonzero(
                np.isin(columns.curve_idx[pool_positions],
                        nonpool_curve_rows)))
            _stream_require(n_nonpool_probed == 0, "EVIDENCE_MISSING",
                            "non-pool positions inside the pool mask")

        probe_positions = _pool_probe_rows(pool_positions, rng)
        # feature probes drawn UP FRONT (same seed-17 protocol over the
        # pool) so pass 2 materialises every needed record in ONE csv pass
        f_rng = np.random.RandomState(SEED)
        feature_positions = np.asarray(
            sorted(int(x) for x in f_rng.choice(
                pool_positions, size=min(N_FEATURE_PROBES,
                                         pool_positions.shape[0]),
                replace=False)), dtype=np.int64)
        _log(f"    pool positions={pool_positions.shape[0]} "
             f"probes={probe_positions.shape[0]} (pool-only)")
        checks["column_equality"], records = _stream_check_columns(
            view, ds_v2, columns,
            np.union1d(probe_positions, feature_positions), rng)
        _log(f"    passed={checks['column_equality']['passed']} "
             f"rows={checks['column_equality']['n_rows_checked']}")

        # labels: probe sids + their dataset positions (pool-only)
        probe_sids = [records[int(p)].sample_id for p in probe_positions]
        sid_to_pos = {s: int(p)
                      for s, p in zip(probe_sids, probe_positions)}
        # curve backbone probes: first/last 100 + ≥1% random POOL curves
        pool_only = [c for c in pool_curve_ids if c in curve_row]
        c_rng = np.random.RandomState(SEED)
        n_c = len(pool_only)
        n_c_rand = min(max(int(n_c * 0.01), 100), n_c)
        c_probe_local = sorted(set(
            list(range(min(100, n_c)))
            + list(range(max(0, n_c - 100), n_c))
            + [int(x) for x in c_rng.choice(
                n_c, size=n_c_rand, replace=False)]))
        probe_curve_ids = [pool_only[i] for i in c_probe_local]
        _log("[stream-cert] phase 1b: label columns (npz memmap vs v2) …")
        npz = NpzColumnMemmaps(tables_dir / "labels_cache_v53mp.npz")
        lc_full = json.loads(
            (tables_dir / "labels_cache_v53mp.json").read_text(
                encoding="utf-8"))
        checks["label_equality"] = _stream_check_labels(
            ds_v2, labels_v2, ep_v2, npz, lc_full, probe_sids, sid_to_pos,
            probe_curve_ids, records, rng)
        _log(f"    passed={checks['label_equality']['passed']} "
             f"sids={checks['label_equality']['n_probe_sids']}")
        phases.append(_phase_stats("columns", t0, rss_before))

        # --- Phase 2: sampler (compact view) --------------------------------
        t0 = time.perf_counter()
        rss_before = _vm_rss_kb()
        _log("[stream-cert] phase 2: sampler equality (compact view) …")
        checks["sampler_equality"] = _stream_check_sampler(
            view, columns, ds_v2, pool_curve_ids)
        _log(f"    passed={checks['sampler_equality']['passed']} "
             f"train={checks['sampler_equality']['train_positions']}")
        phases.append(_phase_stats("sampler", t0, rss_before))

        # --- Phase 3: features (probe records + shard LRU) ------------------
        t0 = time.perf_counter()
        rss_before = _vm_rss_kb()
        _log("[stream-cert] phase 3: feature equality (pool-only probes) …")
        checks["feature_equality"] = _stream_check_features(
            view, ds_v2, records, feature_positions)
        _log(f"    passed={checks['feature_equality']['passed']} "
             f"positions={checks['feature_equality']['positions']}")
        phases.append(_phase_stats("features", t0, rss_before))

        # free v2 before the verdict block (v2 must survive alone)
        del view, columns, records, npz
        gc.collect()
        rss_end = _rss_kb()

        idx_sha = ""
        curves_json = tables_dir / "idx_v2" / "curves.json"
        if curves_json.is_file():
            from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
                _sha256,
            )
            idx_sha = _sha256(curves_json)

        passed = all(c["passed"] for c in checks.values())
        verdict.update({
            "passed": bool(passed),
            "failure": None if passed else {
                "code": "EQUIVALENCE_MISMATCH",
                "detail": "one or more streaming checks failed",
            },
            "idx_curves_json_sha256": idx_sha,
            "hash_verification": hashes,
            "checks": checks,
            "phases": phases,
            "timings": timings,
            "resource": {
                "device": "cpu",
                "gpu_used": False,
                "wall_s": round(time.perf_counter() - wall0, 1),
                "cpu_s": round(time.process_time() - cpu0, 1),
                "ru_maxrss_kb": _rss_kb(),
            },
            "anon_rss_before_kb": rss_start,
            "anon_rss_after_kb": rss_end,
            "anon_rss_before_mib": rss_start // 1024,
            "anon_rss_after_mib": rss_end // 1024,
            "wall_s": round(time.perf_counter() - wall0, 1),
            "cpu_s": round(time.process_time() - cpu0, 1),
            "pool_probe_contract": {
                "pool_only": True,
                "pool_curves": len(pool_curve_ids),
                "pool_positions": int(pool_positions.shape[0]),
                "column_probes": int(probe_positions.shape[0]),
                "curve_probes": len(probe_curve_ids),
                "feature_probes": int(feature_positions.shape[0]),
            },
        })
    except StreamCertError as exc:
        verdict.update({
            "passed": False,
            "failure": {"code": exc.code, "detail": exc.detail},
            "checks": checks,          # partial evidence survives failure
            "phases": phases,
            "timings": timings,
            "wall_s": round(time.perf_counter() - wall0, 1),
            "cpu_s": round(time.process_time() - cpu0, 1),
        })
    except StreamThinViewError as exc:
        verdict.update({
            "passed": False,
            "failure": {"code": exc.code, "detail": exc.detail},
            "checks": checks,          # partial evidence survives failure
            "phases": phases,
            "timings": timings,
            "wall_s": round(time.perf_counter() - wall0, 1),
            "cpu_s": round(time.process_time() - cpu0, 1),
        })
    except Exception as exc:  # noqa: BLE001 — one-shot cert must always write
        verdict.update({
            "passed": False,
            "failure": {
                "code": "RUNTIME_ERROR",
                "detail": f"{type(exc).__name__}: {exc}",
            },
            "checks": checks,          # partial evidence survives failure
            "phases": phases,
            "timings": timings,
            "wall_s": round(time.perf_counter() - wall0, 1),
            "cpu_s": round(time.process_time() - cpu0, 1),
        })
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(verdict, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    verdict["report_path"] = str(report_path)
    return verdict


def run_certificate(tables_dir: Path, report_dir: Path, *,
                    tier: str = DEFAULT_TIER) -> dict[str, Any]:
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

    # pool curve ids: the frozen tier of the split manifest (default 20k) —
    # the driver's _tier_pool_curve_ids logic, now tier-parameterized
    split_path = tables_dir / "split_manifest_v53mp.json"
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    pool_curve_ids = _select_pool_curve_ids(split_doc, tier=tier)
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
        "tier": tier,
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
        description="Bitwise A/B certificate: loader v2 vs loader v1 (§4.2); "
                    "--mode streaming = WP5-CERT bounded 50k certificate")
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--tier", default=DEFAULT_TIER,
                        choices=sorted(TIER_LIMITS),
                        help="frozen tier to certify against subsets.pool "
                             "(default {} — byte-identical default; 50k "
                             "certifies a 50k build)".format(DEFAULT_TIER))
    parser.add_argument("--mode", default="full", choices=("full", "streaming"),
                        help="full = the original co-resident certificate "
                             "(default, unchanged); streaming = the "
                             "preregistered WP5-CERT phased certificate")
    parser.add_argument("--report-path", default=str(STREAM_REPORT_PATH),
                        help="streaming-mode report file "
                             "(default: v541 audits loader_v2_streaming_"
                             "certificate.json)")
    parser.add_argument("--prereg", default=str(STREAM_PREREG_PATH),
                        help="streaming-mode preregistration JSON "
                             "(fail-closed binding)")
    parser.add_argument("--cgroup-evidence-dir", default="",
                        help="streaming-mode cgroup evidence dir (recorded "
                             "as a path reference in the certificate; the "
                             "sampler CSVs/summary are written by "
                             "cgroup_scope.sh after the command exits)")
    args = parser.parse_args(argv)
    if args.mode == "streaming":
        verdict = run_certificate_streaming(
            Path(args.tables_dir), Path(args.report_path),
            tier=args.tier, prereg_path=Path(args.prereg),
            cgroup_evidence_dir=args.cgroup_evidence_dir)
    else:
        verdict = run_certificate(Path(args.tables_dir),
                                  Path(args.report_dir), tier=args.tier)
    print(json.dumps(verdict, indent=1, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
