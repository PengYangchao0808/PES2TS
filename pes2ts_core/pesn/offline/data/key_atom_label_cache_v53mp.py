"""V5.3-MP label cache: serialize the (deterministic) v52 + endpoint label
tables so the 20k-scale training runs do not recompute them per run.

Labels are pure functions of the frozen six tables; the cache is bound to
``data_manifest.json``'s sha256 (fail-closed: a rebuilt dataset with the same
path but different content is never served from a stale cache).  The cache
is a provenance-only accelerator — rebuilding from the tables must always
produce byte-identical label values (round-trip test).

Files (inside the tables dir):

* ``labels_cache_v53mp.npz`` — per-sample arrays + per-curve vectors;
* ``labels_cache_v53mp.json`` — schema, manifest sha, counts, windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
    CurveV52Labels,
    V52LabelTable,
    V52SampleLabel,
)
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
    EndpointLabelTableV1,
    EndpointSampleLabelV1,
)

CACHE_NPZ = "labels_cache_v53mp.npz"
CACHE_JSON = "labels_cache_v53mp.json"
CACHE_SCHEMA = "v53mp_label_cache_v1"


class LabelCacheError(RuntimeError):
    """Typed refusal of the label cache (stale / corrupt / mismatched)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise LabelCacheError(code, detail)


def save_label_cache(
    tables_dir: str | Path,
    v52: V52LabelTable,
    endpoint: EndpointLabelTableV1,
    *,
    dataset_manifest_sha256: str,
) -> dict:
    """Write the label cache (deterministic arrays; returns the json doc)."""
    tables_dir = Path(tables_dir)
    sample_ids = sorted(v52.by_sample)
    _require(sorted(endpoint.by_sample) == sample_ids, "CACHE_COVERAGE",
             "endpoint label table and v52 label table cover different "
             "sample sets")
    sid_index = {sid: i for i, sid in enumerate(sample_ids)}
    n = len(sample_ids)

    q = np.full(n, np.nan)
    q_mask = np.zeros(n, dtype=bool)
    stop = np.full(n, -2, dtype=np.int8)          # -2 = unset sentinel
    d_key_ts = np.full(n, np.nan)
    fold = np.full(n, -1, dtype=np.int64)
    label_status = np.zeros(n, dtype="U16")
    direction = np.zeros(n, dtype="U8")
    curve_of = np.zeros(n, dtype="U24")
    node_index = np.zeros(n, dtype=np.int64)
    intent_of = np.zeros(n, dtype="U40")
    for sid in sample_ids:
        i = sid_index[sid]
        lab = v52.by_sample[sid]
        if lab.q is not None:
            q[i] = float(lab.q)
        q_mask[i] = bool(lab.q_mask)
        stop[i] = int(lab.stop)
        d_key_ts[i] = float(lab.d_key_ts)
        fold[i] = -1 if lab.fold is None else int(lab.fold)
        label_status[i] = str(lab.label_status)
        direction[i] = str(lab.direction)
        curve_of[i] = str(lab.curve_id)
        node_index[i] = int(lab.node_index)
        intent_of[i] = str(lab.intent_id)

    ep_label = np.full(n, -2, dtype=np.int8)
    d_end = np.full(n, np.nan)
    bond_edit_complete = np.zeros(n, dtype=bool)
    geometry_legal = np.zeros(n, dtype=bool)
    ts_row = np.zeros(n, dtype=bool)
    start_row = np.zeros(n, dtype=bool)
    terminal_row = np.zeros(n, dtype=bool)
    degenerate_curve = np.zeros(n, dtype=bool)
    energy_ev_within_window = np.full(n, -1, dtype=np.int8)   # -1 = None
    d_key_ts_ep = np.full(n, np.nan)
    for sid in sample_ids:
        i = sid_index[sid]
        lab = endpoint.by_sample[sid]
        ep_label[i] = int(lab.endpoint_label)
        d_end[i] = float(lab.d_end_angstrom)
        bond_edit_complete[i] = bool(lab.bond_edit_complete)
        geometry_legal[i] = bool(lab.geometry_legal)
        ts_row[i] = bool(lab.ts_row)
        start_row[i] = bool(lab.start_row)
        terminal_row[i] = bool(lab.terminal_row)
        degenerate_curve[i] = bool(lab.degenerate_curve)
        ev = lab.energy_ev_within_window
        energy_ev_within_window[i] = -1 if ev is None else int(bool(ev))
        d_key_ts_ep[i] = float(getattr(lab, "d_key_ts_angstrom", np.nan))

    curve_ids = sorted(v52.curves)
    curve_meta = []
    node_orders = []
    arc_s_list = []
    q_canon_list = []
    for cid in curve_ids:
        c = v52.curves[cid]
        curve_meta.append({
            "curve_id": cid,
            "ts_index": int(c.ts_index),
            "q_zero_arc_reason": str(c.q_zero_arc_reason),
            "n": len(c.node_order),
        })
        node_orders.append(np.asarray(c.node_order, dtype=np.int64))
        arc_s_list.append(np.asarray(c.arc_s, dtype=np.float64))
        q_canon_list.append(
            np.asarray(c.q_canonical, dtype=np.float64)
            if c.q_canonical is not None else np.zeros(0, dtype=np.float64))

    np.savez(
        tables_dir / CACHE_NPZ,
        sample_ids=np.asarray(sample_ids, dtype="U64"),
        q=q, q_mask=q_mask, stop=stop, d_key_ts=d_key_ts,
        fold=fold, label_status=label_status, direction=direction,
        curve_of=curve_of, node_index=node_index, intent_of=intent_of,
        ep_label=ep_label, d_end=d_end,
        bond_edit_complete=bond_edit_complete, geometry_legal=geometry_legal,
        ts_row=ts_row, start_row=start_row, terminal_row=terminal_row,
        degenerate_curve=degenerate_curve,
        energy_ev_within_window=energy_ev_within_window,
        d_key_ts_ep=d_key_ts_ep,
        node_order=np.concatenate(node_orders) if node_orders else np.zeros(0, dtype=np.int64),
        arc_s=np.concatenate(arc_s_list) if arc_s_list else np.zeros(0),
        q_canonical=np.concatenate(q_canon_list) if q_canon_list else np.zeros(0),
    )
    doc = {
        "schema": CACHE_SCHEMA,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "n_samples": n,
        "n_curves": len(curve_ids),
        "curve_meta": curve_meta,
        "endpoint_windows": {
            "positive_window_angstrom": endpoint.positive_window,
            "negative_window_angstrom": endpoint.negative_window,
        },
        "label_schemas": ["pes2ts_e4_labels_v1 (v52 q/stop diagnostics)",
                          "endpoint_label_spec_v1"],
    }
    (tables_dir / CACHE_JSON).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return doc


def load_label_cache(
    tables_dir: str | Path,
    *,
    dataset_manifest_sha256: str,
) -> tuple[V52LabelTable, EndpointLabelTableV1]:
    """Load the label cache; fail-closed on absence/mismatch."""
    tables_dir = Path(tables_dir)
    json_path = tables_dir / CACHE_JSON
    npz_path = tables_dir / CACHE_NPZ
    if not json_path.is_file() or not npz_path.is_file():
        raise LabelCacheError("CACHE_MISSING", f"{tables_dir}")
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    _require(doc.get("schema") == CACHE_SCHEMA, "CACHE_SCHEMA",
             f"{doc.get('schema')!r}")
    _require(doc.get("dataset_manifest_sha256") == dataset_manifest_sha256,
             "CACHE_STALE",
             "cache is bound to a different data_manifest sha256")
    data = np.load(npz_path)
    sample_ids = [str(s) for s in data["sample_ids"]]
    n = len(sample_ids)
    _require(int(doc["n_samples"]) == n, "CACHE_COUNT", "sample count drift")

    by_sample: dict[str, V52SampleLabel] = {}
    q = data["q"]
    q_mask = data["q_mask"]
    stop = data["stop"]
    d_key_ts = data["d_key_ts"]
    fold = data["fold"]
    label_status = data["label_status"]
    direction = data["direction"]
    curve_of = data["curve_of"]
    node_index = data["node_index"]
    intent_of = data["intent_of"]
    for i, sid in enumerate(sample_ids):
        by_sample[sid] = V52SampleLabel(
            sample_id=sid,
            curve_id=str(curve_of[i]),
            node_index=int(node_index[i]),
            intent_id=str(intent_of[i]),
            direction=str(direction[i]),
            fold=(None if int(fold[i]) < 0 else int(fold[i])),
            label_status=str(label_status[i]),
            q=(None if np.isnan(q[i]) else float(q[i])),
            q_mask=bool(q_mask[i]),
            stop=int(stop[i]),
            d_key_ts=float(d_key_ts[i]),
        )

    curves: dict[str, CurveV52Labels] = {}
    node_order_flat = data["node_order"]
    arc_s_flat = data["arc_s"]
    q_flat = data["q_canonical"]
    pos = 0
    pos_f = 0
    pos_q = 0
    for meta in doc["curve_meta"]:
        k = int(meta["n"])
        no = tuple(int(v) for v in node_order_flat[pos:pos + k])
        arc = tuple(float(v) for v in arc_s_flat[pos_f:pos_f + k])
        has_q = not meta["q_zero_arc_reason"]
        qc = (tuple(float(v) for v in q_flat[pos_q:pos_q + k])
              if has_q else None)
        curves[meta["curve_id"]] = CurveV52Labels(
            curve_id=meta["curve_id"],
            ts_index=int(meta["ts_index"]),
            arc_s=arc,
            q_canonical=qc,
            q_zero_arc_reason=str(meta["q_zero_arc_reason"]),
            node_order=no,
        )
        pos += k
        pos_f += k
        pos_q += k if has_q else 0
    v52 = V52LabelTable(by_sample=by_sample, curves=curves)

    ep_label = data["ep_label"]
    d_end = data["d_end"]
    bond_edit_complete = data["bond_edit_complete"]
    geometry_legal = data["geometry_legal"]
    ts_row = data["ts_row"]
    start_row = data["start_row"]
    terminal_row = data["terminal_row"]
    degenerate_curve = data["degenerate_curve"]
    energy_ev = data["energy_ev_within_window"]
    d_key_ts_ep = data["d_key_ts_ep"]
    ep_by_sample: dict[str, EndpointSampleLabelV1] = {}
    for i, sid in enumerate(sample_ids):
        ev = int(energy_ev[i])
        ep_by_sample[sid] = EndpointSampleLabelV1(
            sample_id=sid,
            curve_id=str(curve_of[i]),
            node_index=int(node_index[i]),
            intent_id=str(intent_of[i]),
            direction=str(direction[i]),
            fold=(None if int(fold[i]) < 0 else int(fold[i])),
            label_status=str(label_status[i]),
            endpoint_label=int(ep_label[i]),
            d_end_angstrom=float(d_end[i]),
            bond_edit_complete=bool(bond_edit_complete[i]),
            geometry_legal=bool(geometry_legal[i]),
            ts_row=bool(ts_row[i]),
            start_row=bool(start_row[i]),
            terminal_row=bool(terminal_row[i]),
            degenerate_curve=bool(degenerate_curve[i]),
            energy_ev_within_window=(None if ev < 0 else bool(ev)),
            d_key_ts_angstrom=float(d_key_ts_ep[i]),
        )
    windows = doc["endpoint_windows"]
    endpoint = EndpointLabelTableV1(
        by_sample=ep_by_sample,
        positive_window=float(windows["positive_window_angstrom"]),
        negative_window=float(windows["negative_window_angstrom"]),
    )
    return v52, endpoint


__all__ = [
    "CACHE_NPZ",
    "CACHE_JSON",
    "CACHE_SCHEMA",
    "LabelCacheError",
    "save_label_cache",
    "load_label_cache",
]
