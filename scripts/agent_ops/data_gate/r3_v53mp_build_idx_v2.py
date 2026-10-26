#!/usr/bin/env python3
"""Build the V5.3-MP loader-v2 columnar index (idx_v2/) — plan §4 / lane I1.

Reads the EXISTING frozen artifacts in ``tables_dir`` (default
``outputs/pesn/v53mp/data``):

* ``.light_index_v53mp.pkl``   — v1 light index (sample order = v1 samples);
* ``labels_cache_v53mp.npz``   — v52 + endpoint label cache (sorted by sid);
* ``labels_cache_v53mp.json``  — label-cache binding + per-curve meta;
* ``shard_index_v53mp.json``   — presence check only (geometry stays there);
* ``data_manifest.json``       — binding sha.

and emits ``<tables_dir>/idx_v2/`` (flat ``.npy`` per field,
``allow_pickle=False`` everywhere; see ``key_atom_dataset_v3_col`` for the
reader).  Sample order is EXACTLY the v1 light-pickle ``samples`` order —
the §4.2 certificate verifies this field-by-field against v1.

The builder fails closed (typed ``KeyAtomDatasetV3Error``) on:
missing/stale source, duplicate (curve, node, intent) identity, non-contiguous
node sets, sample_id format drift, weight-sum drift, label-cache count/coverage
mismatch.

Progress is logged to stderr; the final summary JSON is printed to stdout.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    LABEL_STATUSES,
    LIGHT_CACHE_PKL,
    KeyAtomDatasetV3Error,
    _require,
    _sha256,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (  # noqa: E402
    IDX2_DIRNAME,
    IDX2_SCHEMA,
    _pack_sid_key,
    _slot_of,
)
from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (  # noqa: E402
    CACHE_JSON,
    CACHE_NPZ,
)

SHARD_INDEX_FILENAME = "shard_index_v53mp.json"
SPLIT_DEFAULT = "split_manifest_v53mp.json"
D0_PRIMARY = 0.08
DIRECTIONS = ("forward", "reverse")

#: hash table load factor for the sample_id → row probe table.
SID_LOAD = 1.3


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _save_npy(path: Path, array: np.ndarray) -> dict:
    """Deterministic ``.npy`` write (allow_pickle=False) + size/sha record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.save(fh, array, allow_pickle=False)
    return {
        "sha256": _sha256(path),
        "dtype": str(array.dtype),
        "shape": [int(s) for s in array.shape],
        "bytes": int(path.stat().st_size),
    }


def _write_str_column(values: list[str], path: Path) -> dict:
    arr = np.asarray(values, dtype="U64")
    return _save_npy(path, arr)


def build_idx_v2(tables_dir: Path, out_dir: Path, *,
                 overwrite: bool = False, verify_sources: bool = True) -> dict:
    """Build ``idx_v2/``; returns the summary dict (also printed as JSON)."""
    t_start = time.perf_counter()
    tables_dir = Path(tables_dir)
    out = Path(out_dir)
    _require(tables_dir.is_dir(), "TABLES_DIR",
             f"{tables_dir} is not a directory")
    for name in (LIGHT_CACHE_PKL, CACHE_NPZ, CACHE_JSON,
                 SHARD_INDEX_FILENAME, "data_manifest.json"):
        _require((tables_dir / name).is_file(), "SOURCE_MISSING",
                 f"{tables_dir / name} not found — build the v1 light index "
                 "and label cache first")
    if out.exists() and any(out.iterdir()) and not overwrite:
        _require(False, "IDX2_EXISTS",
                 f"{out} is not empty; pass --overwrite to rebuild")

    # --- source identity (bound into curves.json) ----------------------------
    _log("[1/7] hashing sources …")
    manifest_sha = _sha256(tables_dir / "data_manifest.json")
    split_path = tables_dir / SPLIT_DEFAULT
    split_sha = _sha256(split_path) if split_path.is_file() else ""
    sources = {}
    for key, name in (("light_pickle", LIGHT_CACHE_PKL),
                      ("labels_npz", CACHE_NPZ),
                      ("labels_json", CACHE_JSON)):
        path = tables_dir / name
        stat = path.stat()
        sources[key] = {
            "path": name,
            "sha256": _sha256(path) if verify_sources else "",
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        _log(f"    {name}: {stat.st_size / 1e9:.2f} GB")

    # --- load v1 light index --------------------------------------------------
    _log("[2/7] loading v1 light pickle …")
    t0 = time.perf_counter()
    with (tables_dir / LIGHT_CACHE_PKL).open("rb") as fh:
        state = pickle.load(fh)
    for key in ("node_index", "curve_atomic", "curve_energy", "samples",
                "weight_totals"):
        _require(key in state, "SOURCE_SHAPE",
                 f"light pickle missing key {key!r}")
    samples = state["samples"]
    n_rows = len(samples)
    _require(n_rows > 0, "SOURCE_SHAPE", "light pickle has no samples")
    _log(f"    {n_rows} samples ({time.perf_counter() - t0:.1f}s)")

    # --- curve table (first-appearance order == v1 samples block order) ------
    _log("[4/7] curve + intent tables …")
    curve_ids: list[str] = []
    curve_row: dict[str, int] = {}
    for rec in samples:
        cid = str(rec.curve_id)
        if cid not in curve_row:
            curve_row[cid] = len(curve_ids)
            curve_ids.append(cid)
    n_curves = len(curve_ids)
    _require(n_curves < (1 << 21), "IDX2_SHAPE",
             "curve count exceeds idx_v2 packing limit 2^21")

    # per-curve meta from irc_curve.csv (ts_index / endpoints) + pickle state
    import csv as _csv
    irc_meta: dict[str, dict[str, str]] = {}
    with (tables_dir / "irc_curve.csv").open("r", encoding="utf-8",
                                             newline="") as fh:
        for row in _csv.DictReader(fh):
            irc_meta[str(row["curve_id"])] = row
    curve_meta: list[dict[str, Any]] = []
    node_offsets = np.zeros(n_curves + 1, dtype=np.int64)
    energies = np.full(
        sum(len(state["node_index"][c]) for c in curve_ids), np.nan,
        dtype=np.float64)
    for row, cid in enumerate(curve_ids):
        nodes = state["node_index"].get(cid)
        _require(bool(nodes), "SOURCE_SHAPE", f"curve {cid} has no nodes")
        _require(nodes == set(range(len(nodes))), "NODE_ORDER",
                 f"curve {cid} node set is not contiguous 0..n-1 "
                 f"({len(nodes)} nodes) — idx_v2 layout assumes it")
        n_nodes = len(nodes)
        node_offsets[row + 1] = node_offsets[row] + n_nodes
        atomic = state["curve_atomic"].get(cid) or ()
        _require(bool(atomic), "ATOMIC_NUMBERS_MISSING",
                 f"curve {cid} has no atomic_numbers in the light index")
        curve_energy = state["curve_energy"].get(cid) or {}
        has_energy = set(curve_energy) == set(nodes)
        if has_energy:
            start = int(node_offsets[row])
            for node, value in curve_energy.items():
                energies[start + int(node)] = float(value)
        meta_row = irc_meta.get(cid)
        ts_index = (int(meta_row["ts_index"]) if meta_row
                    and str(meta_row.get("ts_index", "")).strip() else n_nodes - 1)
        curve_meta.append({
            "n_nodes": n_nodes,
            "atomic_numbers": [int(a) for a in atomic],
            "ts_index": int(ts_index),
            "endpoint_a": int((meta_row or {}).get("endpoint_a_index") or 0),
            "endpoint_b": int((meta_row or {}).get("endpoint_b_index")
                              or (n_nodes - 1)),
            "has_energy": bool(has_energy),
            "weight_total": float(state["weight_totals"].get(cid, 0.0)),
        })

    # --- intents (first-appearance over samples) -----------------------------
    intent_ids: list[str] = []
    intent_row: dict[str, int] = {}
    intent_objs: list[Any] = []
    for rec in samples:
        iid = str(rec.intent.intent_id)
        if iid not in intent_row:
            intent_row[iid] = len(intent_ids)
            intent_ids.append(iid)
            intent_objs.append(rec.intent)
    n_intents = len(intent_ids)
    _require(n_intents < (1 << 21), "IDX2_SHAPE",
             "intent count exceeds idx_v2 packing limit 2^21")
    # intent-pair involution (T0.2) — same invariant v1 verifies at init
    for (cid, iid), intent in (( (o.curve_id, o.intent_id), o)
                               for o in intent_objs):
        if intent.intent_underdetermined:
            continue
        rev = intent_row.get(str(intent.reverse_of_intent_id))
        _require(rev is not None, "INTENT_REVERSE_MISSING",
                 f"intent {iid} has no reverse row "
                 f"{intent.reverse_of_intent_id!r}")
        back = intent_objs[rev]
        _require(back.reverse_of_intent_id == iid, "INTENT_INVOLUTION",
                 f"reverse(reverse({iid})) != {iid} (curve {cid})")
    _log(f"    {n_curves} curves, {n_intents} intents")

    # --- per-sample columns ---------------------------------------------------
    _log("[5/7] per-sample columns + actions …")
    t0 = time.perf_counter()
    curve_idx_col = np.empty(n_rows, dtype=np.int32)
    node_col = np.empty(n_rows, dtype=np.int32)
    intent_col = np.empty(n_rows, dtype=np.int32)
    fold_col = np.empty(n_rows, dtype=np.int8)
    status_col = np.empty(n_rows, dtype=np.int8)
    d0_col = np.empty(n_rows, dtype=np.float64)
    lw_col = np.empty(n_rows, dtype=np.float32)
    cw_col = np.empty(n_rows, dtype=np.float32)
    sw_col = np.empty(n_rows, dtype=np.float32)
    tnode_col = np.empty(n_rows, dtype=np.int32)
    aoff_col = np.empty(n_rows, dtype=np.int32)
    alen_col = np.empty(n_rows, dtype=np.int8)
    action_blocks: list[np.ndarray] = []
    action_total = 0

    status_table = sorted({str(r.label_status) for r in samples})
    status_code = {s: i for i, s in enumerate(status_table)}
    identity: set[tuple[int, int, int]] = set()
    seen_sid: set[str] = set()

    for i, rec in enumerate(samples):
        cid = str(rec.curve_id)
        c_row = curve_row[cid]
        iid = str(rec.intent.intent_id)
        i_row = intent_row[iid]
        node = int(rec.node_index)
        _require(node < (1 << 21), "IDX2_SHAPE",
                 f"sample {rec.sample_id}: node_index exceeds packing limit")
        key = (c_row, node, i_row)
        _require(key not in identity, "SAMPLE_ID_DUP",
                 f"duplicate (curve, node, intent) identity {key}")
        identity.add(key)
        d0 = float(rec.d0)
        expect_sid = f"{cid}:{node}:{iid}:{d0}"
        _require(str(rec.sample_id) == expect_sid, "SAMPLE_ID_FORMAT",
                 f"sample_id {rec.sample_id!r} != reconstructable "
                 f"{expect_sid!r} — idx_v2 cannot serve this table")
        _require(expect_sid not in seen_sid, "SAMPLE_ID_DUP",
                 f"duplicate sample_id {expect_sid!r}")
        seen_sid.add(expect_sid)
        fold = rec.fold
        _require(fold is None or 0 <= int(fold) <= 127, "IDX2_SHAPE",
                 f"sample {rec.sample_id}: fold {fold} out of int8 range")
        status = str(rec.label_status)
        _require(status in status_code, "IDX2_SHAPE",
                 f"sample {rec.sample_id}: unknown label_status {status!r}")
        target_node = rec.target_node_index
        _require(target_node is None or int(target_node) >= 0, "IDX2_SHAPE",
                 f"sample {rec.sample_id}: negative target_node_index")
        actions = rec.target_action_u
        if actions is None:
            alen = 0
            aoff = action_total
        else:
            alen = len(actions)
            _require(0 < alen <= 127, "IDX2_SHAPE",
                     f"sample {rec.sample_id}: |K|={alen} out of int8 range")
            _require(len(rec.intent.key_rows()) == alen, "TARGET_U_SHAPE",
                     f"sample {rec.sample_id}: target_action_u has {alen} "
                     f"rows != |K|={len(rec.intent.key_rows())}")
            aoff = action_total
            action_blocks.append(
                np.asarray(actions, dtype=np.float64).astype(np.float32))
            action_total += alen
        curve_idx_col[i] = c_row
        node_col[i] = node
        intent_col[i] = i_row
        fold_col[i] = -1 if fold is None else int(fold)
        status_col[i] = status_code[status]
        d0_col[i] = d0
        lw_col[i] = float(rec.label_weight)
        cw_col[i] = float(rec.curve_weight)
        sw_col[i] = float(rec.sample_weight)
        tnode_col[i] = -1 if target_node is None else int(target_node)
        aoff_col[i] = aoff
        alen_col[i] = alen
        if (i + 1) % 1_000_000 == 0:
            _log(f"    {i + 1}/{n_rows} …")
    del identity, seen_sid
    _require(bool((np.abs(d0_col - D0_PRIMARY) <= 1e-12).all()), "IDX2_D0",
             "light pickle carries rows outside the primary d0 — idx_v2 "
             "binding assumes a single d0")
    actions_arr = (np.concatenate(action_blocks) if action_blocks
                   else np.zeros((0, 3), dtype=np.float32))
    _require(actions_arr.shape == (action_total, 3), "IDX2_SHAPE",
             "action buffer shape mismatch")
    _log(f"    {n_rows} rows, {action_total} action rows "
         f"({time.perf_counter() - t0:.1f}s)")

    # The light pickle holds 14.7 M Python records (~31 GB at 50k).  All
    # per-sample columns are now built, so release it BEFORE opening the
    # ~10 GB label cache.  Peak becomes max(pickle, labels+arrays), not the
    # sum (the 50k host cannot hold both simultaneously without pagefile).
    _log("[5b/7] freeing light pickle before label cache …")
    del samples, state
    gc.collect()
    # CPython's pymalloc keeps freed arenas; return them to the OS so the
    # cgroup does not count the freed 30 GB pickle against the next phase.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    gc.collect()
    _log("    pickle freed; label cache follows")

    # --- load label cache -----------------------------------------------------
    _log("[5c/7] loading label cache …")
    t0 = time.perf_counter()
    lab_doc = json.loads((tables_dir / CACHE_JSON).read_text(encoding="utf-8"))
    _require(int(lab_doc["n_samples"]) == n_rows, "SOURCE_SHAPE",
             f"label cache n_samples {lab_doc['n_samples']} != "
             f"light pickle {n_rows}")
    _require(int(lab_doc["n_curves"]) == n_curves, "SOURCE_SHAPE",
             "label cache n_curves != light pickle curve count")
    data = np.load(tables_dir / CACHE_NPZ, allow_pickle=False)
    lab_sids = [str(s) for s in data["sample_ids"]]
    _require(len(lab_sids) == n_rows, "SOURCE_SHAPE",
             "label cache row count != sample count")
    sid_row = {sid: i for i, sid in enumerate(lab_sids)}
    _require(len(sid_row) == n_rows, "SOURCE_SHAPE",
             "duplicate sample_ids in label cache")
    _log(f"    {len(lab_sids)} label rows ({time.perf_counter() - t0:.1f}s)")

    # --- curve-weight invariant (ORA-2 #3, same tolerance as v1) -------------
    _log("[6/7] invariants + label reorder …")
    for row, cid in enumerate(curve_ids):
        total = curve_meta[row]["weight_total"]
        _require(abs(total - 1.0) <= 1e-9, "CURVE_WEIGHT_SUM",
                 f"curve {cid}: sum(curve_weight)={total!r} != 1.0")

    # --- sid lookup table (open addressing over packed keys) -----------------
    table_size = 1
    while table_size < n_rows * SID_LOAD:
        table_size <<= 1
    keys = np.full(table_size, -1, dtype=np.int64)
    vals = np.full(table_size, -1, dtype=np.int32)
    mask = table_size - 1
    for i in range(n_rows):
        key = _pack_sid_key(int(curve_idx_col[i]), int(node_col[i]),
                            int(intent_col[i]))
        slot = _slot_of(key, mask)
        while int(keys[slot]) != -1:
            slot = (slot + 1) & mask
        keys[slot] = key
        vals[slot] = i

    # --- label reorder (label cache is sorted by sid; idx follows samples) ---
    v52_q = np.empty(n_rows, dtype=np.float64)
    v52_q_mask = np.empty(n_rows, dtype=bool)
    v52_stop = np.empty(n_rows, dtype=np.int8)
    v52_dts = np.empty(n_rows, dtype=np.float64)
    ep_label = np.empty(n_rows, dtype=np.int8)
    ep_dend = np.empty(n_rows, dtype=np.float64)
    ep_dts = np.empty(n_rows, dtype=np.float64)
    ep_bond = np.empty(n_rows, dtype=bool)
    ep_geom = np.empty(n_rows, dtype=bool)
    ep_ts = np.empty(n_rows, dtype=bool)
    ep_start = np.empty(n_rows, dtype=bool)
    ep_term = np.empty(n_rows, dtype=bool)
    ep_degen = np.empty(n_rows, dtype=bool)
    ep_energy = np.empty(n_rows, dtype=np.int8)
    src_q = data["q"]
    src_qm = data["q_mask"]
    src_stop = data["stop"]
    src_dts = data["d_key_ts"]
    src_ep = data["ep_label"]
    src_dend = data["d_end"]
    src_dts_ep = data["d_key_ts_ep"]
    src_bond = data["bond_edit_complete"]
    src_geom = data["geometry_legal"]
    src_ts = data["ts_row"]
    src_start = data["start_row"]
    src_term = data["terminal_row"]
    src_degen = data["degenerate_curve"]
    src_energy = data["energy_ev_within_window"]
    used = np.zeros(len(lab_sids), dtype=bool)
    for i in range(n_rows):
        sid = f"{curve_ids[int(curve_idx_col[i])]}:{int(node_col[i])}:" \
              f"{intent_ids[int(intent_col[i])]}:{float(d0_col[i])}"
        j = sid_row[sid]
        used[j] = True
        v52_q[i] = src_q[j]
        v52_q_mask[i] = src_qm[j]
        v52_stop[i] = src_stop[j]
        v52_dts[i] = src_dts[j]
        ep_label[i] = src_ep[j]
        ep_dend[i] = src_dend[j]
        ep_dts[i] = src_dts_ep[j]
        ep_bond[i] = src_bond[j]
        ep_geom[i] = src_geom[j]
        ep_ts[i] = src_ts[j]
        ep_start[i] = src_start[j]
        ep_term[i] = src_term[j]
        ep_degen[i] = src_degen[j]
        ep_energy[i] = src_energy[j]
    _require(bool(used.all()), "LABEL_COVERAGE",
             f"label cache has {int((~used).sum())} rows not reachable from "
             "the light index")
    del (src_q, src_qm, src_stop, src_dts, src_ep, src_dend, src_dts_ep,
         src_bond, src_geom, src_ts, src_start, src_term, src_degen,
         src_energy, used)

    # --- per-curve label backbone, reordered to curve_ids order --------------
    _require(len(lab_doc["curve_meta"]) == n_curves, "SOURCE_SHAPE",
             "label cache curve_meta count != curve count")
    node_order_flat = data["node_order"]
    arc_s_flat = data["arc_s"]
    q_flat = data["q_canonical"]
    lab_n = np.zeros(n_curves, dtype=np.int64)
    lab_offset = np.zeros(n_curves + 1, dtype=np.int64)
    lab_q_offset = np.zeros(n_curves + 1, dtype=np.int64)
    lab_ts = np.zeros(n_curves, dtype=np.int64)
    lab_has_q = np.zeros(n_curves, dtype=bool)
    lab_reason: list[str] = []
    node_order_out = np.zeros(len(node_order_flat), dtype=np.int32)
    arc_s_out = np.zeros(len(arc_s_flat), dtype=np.float64)
    q_out = np.zeros(len(q_flat), dtype=np.float64)
    pos_out = pos_out_f = pos_out_q = 0
    # source ranges per curve row (label cache flattens in curve_meta order;
    # rekey by the idx curve_ids row so both orderings stay distinct)
    ranges: dict[int, tuple[int, int, int, int, bool, int, str]] = {}
    p = pf = pq = 0
    for meta in lab_doc["curve_meta"]:
        cid = str(meta["curve_id"])
        _require(cid in curve_row, "SOURCE_SHAPE",
                 f"label cache curve {cid} not in light index")
        i = curve_row[cid]
        k = int(meta["n"])
        has_q = not bool(meta["q_zero_arc_reason"])
        ranges[i] = (p, pf, pq, k, has_q, int(meta["ts_index"]),
                     str(meta["q_zero_arc_reason"]))
        p += k
        pf += k
        pq += k if has_q else 0
    _require(len(ranges) == n_curves, "SOURCE_SHAPE",
             "label cache curve set != light index curve set")
    for i, cid in enumerate(curve_ids):
        p0, pf0, pq0, k, has_q, ts, reason = ranges[i]
        # NOTE: label-cache "n" is irc_curve.n_nodes (the v52 builder's path
        # order length); it MAY differ from the light index node count on
        # real deliveries, so label curves carry their own offsets.
        node_order_out[pos_out:pos_out + k] = \
            node_order_flat[p0:p0 + k].astype(np.int32)
        arc_s_out[pos_out_f:pos_out_f + k] = arc_s_flat[pf0:pf0 + k]
        if has_q:
            q_out[pos_out_q:pos_out_q + k] = q_flat[pq0:pq0 + k]
        lab_n[i] = k
        lab_ts[i] = ts
        lab_has_q[i] = has_q
        lab_reason.append(reason)
        lab_offset[i + 1] = lab_offset[i] + k
        lab_q_offset[i + 1] = lab_q_offset[i] + (k if has_q else 0)
        pos_out += k
        pos_out_f += k
        pos_out_q += k if has_q else 0
    _require(pos_out == len(node_order_flat), "SOURCE_SHAPE",
             "label curve node_order flat length drift")
    del node_order_flat, arc_s_flat, q_flat, ranges

    # --- write idx_v2 ---------------------------------------------------------
    _log(f"[7/7] writing {out} …")
    if out.exists() and overwrite:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    sample_dir = out / "samples"
    files["samples/curve_idx.npy"] = _save_npy(
        sample_dir / "curve_idx.npy", curve_idx_col)
    files["samples/node_index.npy"] = _save_npy(
        sample_dir / "node_index.npy", node_col)
    files["samples/intent_idx.npy"] = _save_npy(
        sample_dir / "intent_idx.npy", intent_col)
    files["samples/fold.npy"] = _save_npy(sample_dir / "fold.npy", fold_col)
    files["samples/label_status.npy"] = _save_npy(
        sample_dir / "label_status.npy", status_col)
    files["samples/d0.npy"] = _save_npy(sample_dir / "d0.npy", d0_col)
    files["samples/label_weight.npy"] = _save_npy(
        sample_dir / "label_weight.npy", lw_col)
    files["samples/curve_weight.npy"] = _save_npy(
        sample_dir / "curve_weight.npy", cw_col)
    files["samples/sample_weight.npy"] = _save_npy(
        sample_dir / "sample_weight.npy", sw_col)
    files["samples/target_node.npy"] = _save_npy(
        sample_dir / "target_node.npy", tnode_col)
    files["samples/action_off.npy"] = _save_npy(
        sample_dir / "action_off.npy", aoff_col)
    files["samples/action_len.npy"] = _save_npy(
        sample_dir / "action_len.npy", alen_col)
    files["actions.npy"] = _save_npy(out / "actions.npy", actions_arr)

    it = out / "intents"
    # CSR over src bonds + tgt bonds sharing bond_i/bond_j/bond_order:
    # pass 1 appends ALL src bonds (row-contiguous), pass 2 ALL tgt bonds.
    src_off = np.zeros(n_intents + 1, dtype=np.int32)
    tgt_off = np.zeros(n_intents + 1, dtype=np.int32)
    bi: list[int] = []
    bj: list[int] = []
    bo: list[float] = []
    for row, obj in enumerate(intent_objs):
        for i, j, order in obj.g_src_bonds:
            bi.append(int(i)); bj.append(int(j)); bo.append(float(order))
        src_off[row + 1] = len(bi)
    tgt_off[0] = len(bi)                      # tgt block starts past src block
    for row, obj in enumerate(intent_objs):
        for i, j, order in obj.g_tgt_bonds:
            bi.append(int(i)); bj.append(int(j)); bo.append(float(order))
        tgt_off[row + 1] = len(bi)
    cv_off = np.zeros(n_intents + 1, dtype=np.int32)
    cv_deltas_out: list[float] = []
    cv_coords_out: list[int] = []
    cv_coords_off = np.zeros(n_intents + 1, dtype=np.int32)
    g_atoms_out: list[int] = []
    g_atoms_off = np.zeros(n_intents + 1, dtype=np.int32)
    reverse_of = np.full(n_intents, -1, dtype=np.int32)
    for row, obj in enumerate(intent_objs):
        cv_deltas_out.extend(float(v) for v in obj.cv_deltas)
        cv_off[row + 1] = len(cv_deltas_out)
        for a, b in obj.cv_coordinates:
            cv_coords_out.extend((int(a), int(b)))
        cv_coords_off[row + 1] = len(cv_coords_out)
        g_atoms_out.extend(int(a) for a in obj.g_atoms)
        g_atoms_off[row + 1] = len(g_atoms_out)
        rev = intent_row.get(str(obj.reverse_of_intent_id), -1)
        reverse_of[row] = rev
    files["intents/intent_ids.npy"] = _write_str_column(
        intent_ids, it / "intent_ids.npy")
    files["intents/curve_idx.npy"] = _save_npy(
        it / "curve_idx.npy",
        np.asarray([curve_row[obj.curve_id] for obj in intent_objs],
                   dtype=np.int32))
    files["intents/direction.npy"] = _save_npy(
        it / "direction.npy",
        np.asarray([0 if obj.direction == "forward" else 1
                    for obj in intent_objs], dtype=np.int8))
    files["intents/reverse_of.npy"] = _save_npy(
        it / "reverse_of.npy", reverse_of)
    files["intents/pair_ids.npy"] = _write_str_column(
        [str(obj.intent_pair_id) for obj in intent_objs],
        it / "pair_ids.npy")
    files["intents/g_src_off.npy"] = _save_npy(it / "g_src_off.npy", src_off)
    files["intents/g_tgt_off.npy"] = _save_npy(it / "g_tgt_off.npy", tgt_off)
    files["intents/bond_i.npy"] = _save_npy(
        it / "bond_i.npy", np.asarray(bi, dtype=np.int32))
    files["intents/bond_j.npy"] = _save_npy(
        it / "bond_j.npy", np.asarray(bj, dtype=np.int32))
    files["intents/bond_order.npy"] = _save_npy(
        it / "bond_order.npy", np.asarray(bo, dtype=np.float64))
    files["intents/cv_off.npy"] = _save_npy(it / "cv_off.npy", cv_off)
    files["intents/cv_deltas.npy"] = _save_npy(
        it / "cv_deltas.npy", np.asarray(cv_deltas_out, dtype=np.float64))
    files["intents/cv_coords.npy"] = _save_npy(
        it / "cv_coords.npy", np.asarray(cv_coords_out, dtype=np.int32))
    files["intents/cv_coords_off.npy"] = _save_npy(
        it / "cv_coords_off.npy", cv_coords_off)
    files["intents/g_atoms.npy"] = _save_npy(
        it / "g_atoms.npy", np.asarray(g_atoms_out, dtype=np.int32))
    files["intents/g_atoms_off.npy"] = _save_npy(
        it / "g_atoms_off.npy", g_atoms_off)
    files["intents/version_ids.npy"] = _write_str_column(
        [str(obj.intent_version) for obj in intent_objs],
        it / "version_ids.npy")
    files["intents/underdetermined.npy"] = _save_npy(
        it / "underdetermined.npy",
        np.asarray([bool(obj.intent_underdetermined) for obj in intent_objs],
                   dtype=bool))
    files["intents/cv_kind_ids.npy"] = _write_str_column(
        [str(obj.cv_kind) for obj in intent_objs], it / "cv_kind_ids.npy")

    lab_dir = out / "labels_v52"
    files["labels_v52/q.npy"] = _save_npy(lab_dir / "q.npy", v52_q)
    files["labels_v52/q_mask.npy"] = _save_npy(
        lab_dir / "q_mask.npy", v52_q_mask)
    files["labels_v52/stop.npy"] = _save_npy(lab_dir / "stop.npy", v52_stop)
    files["labels_v52/d_key_ts.npy"] = _save_npy(
        lab_dir / "d_key_ts.npy", v52_dts)
    ep_dir = out / "labels_ep"
    files["labels_ep/ep_label.npy"] = _save_npy(
        ep_dir / "ep_label.npy", ep_label)
    files["labels_ep/d_end.npy"] = _save_npy(ep_dir / "d_end.npy", ep_dend)
    files["labels_ep/d_key_ts_ep.npy"] = _save_npy(
        ep_dir / "d_key_ts_ep.npy", ep_dts)
    files["labels_ep/bond_edit_complete.npy"] = _save_npy(
        ep_dir / "bond_edit_complete.npy", ep_bond)
    files["labels_ep/geometry_legal.npy"] = _save_npy(
        ep_dir / "geometry_legal.npy", ep_geom)
    files["labels_ep/ts_row.npy"] = _save_npy(ep_dir / "ts_row.npy", ep_ts)
    files["labels_ep/start_row.npy"] = _save_npy(
        ep_dir / "start_row.npy", ep_start)
    files["labels_ep/terminal_row.npy"] = _save_npy(
        ep_dir / "terminal_row.npy", ep_term)
    files["labels_ep/degenerate_curve.npy"] = _save_npy(
        ep_dir / "degenerate_curve.npy", ep_degen)
    files["labels_ep/energy_ev_within_window.npy"] = _save_npy(
        ep_dir / "energy_ev_within_window.npy", ep_energy)
    lc_dir = out / "label_curves"
    files["label_curves/node_order.npy"] = _save_npy(
        lc_dir / "node_order.npy", node_order_out)
    files["label_curves/arc_s.npy"] = _save_npy(
        lc_dir / "arc_s.npy", arc_s_out)
    files["label_curves/q_canonical.npy"] = _save_npy(
        lc_dir / "q_canonical.npy", q_out)
    files["label_curves/offsets.npy"] = _save_npy(
        lc_dir / "offsets.npy", lab_offset)
    files["label_curves/q_offsets.npy"] = _save_npy(
        lc_dir / "q_offsets.npy", lab_q_offset)
    files["sid_lookup_keys.npy"] = _save_npy(
        out / "sid_lookup_keys.npy", keys)
    files["sid_lookup_vals.npy"] = _save_npy(out / "sid_lookup_vals.npy", vals)
    files["curve_node_energies.npy"] = _save_npy(
        out / "curve_node_energies.npy", energies)

    doc = {
        "schema": IDX2_SCHEMA,
        "created_at": _utc_now(),
        "builder_git_commit": _git_commit(),
        "core_index_bytes_per_sample": int(
            sum(np.dtype(t).itemsize for t in
                ("int32", "int32", "int32", "int8", "int8", "float64",
                 "float32", "float32", "float32", "int32", "int32", "int8"))),
        "binding": {
            "schema": IDX2_SCHEMA,
            "d0": D0_PRIMARY,
            "label_statuses": sorted(LABEL_STATUSES),
            "directions": sorted(DIRECTIONS),
            "fold": None,
            "n_samples": n_rows,
            "n_curves": n_curves,
            "n_intents": n_intents,
            "data_manifest_sha256": manifest_sha,
            "split_manifest_sha256": split_sha,
            "source_light_pickle": sources["light_pickle"],
            "source_labels_npz": sources["labels_npz"],
            "source_labels_json": sources["labels_json"],
            "shard_index_present": True,
        },
        "curve_ids": curve_ids,
        "curves": curve_meta,
        "label_status_table": status_table,
        "labels": {
            "endpoint_windows": {
                "positive_window_angstrom": float(
                    lab_doc["endpoint_windows"]["positive_window_angstrom"]),
                "negative_window_angstrom": float(
                    lab_doc["endpoint_windows"]["negative_window_angstrom"]),
            },
        },
        "label_curves": [
            {"lab_n": int(lab_n[i]), "ts_index": int(lab_ts[i]),
             "has_q": bool(lab_has_q[i]),
             "q_zero_arc_reason": lab_reason[i]}
            for i in range(n_curves)
        ],
        "files": files,
    }
    (out / "curves.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    total_bytes = sum(int(m["bytes"]) for m in files.values())
    summary = {
        "schema": IDX2_SCHEMA,
        "idx_dir": str(out),
        "n_samples": n_rows,
        "n_curves": n_curves,
        "n_intents": n_intents,
        "action_rows": action_total,
        "total_bytes": total_bytes,
        "bytes_per_sample": round(total_bytes / n_rows, 1),
        "core_index_bytes_per_sample":
            doc["core_index_bytes_per_sample"],
        "elapsed_s": round(time.perf_counter() - t_start, 1),
        "created_at": doc["created_at"],
        "builder_git_commit": doc["builder_git_commit"],
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_v53mp_build_idx_v2",
        description="Build the columnar memmap index (idx_v2/) for loader v2")
    parser.add_argument(
        "--tables-dir",
        default=str(PROJECT_ROOT / "outputs/pesn/v53mp/data"))
    parser.add_argument("--out", default=None,
                        help="default: <tables-dir>/idx_v2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify-sources", action="store_true",
                        help="skip source sha256 (size/mtime still bound)")
    args = parser.parse_args(argv)
    tables_dir = Path(args.tables_dir)
    out = Path(args.out) if args.out else tables_dir / IDX2_DIRNAME
    try:
        summary = build_idx_v2(
            tables_dir, out, overwrite=args.overwrite,
            verify_sources=not args.no_verify_sources)
    except KeyAtomDatasetV3Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
