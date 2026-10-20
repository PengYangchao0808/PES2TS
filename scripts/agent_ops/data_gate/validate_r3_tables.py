"""Data Gate CLI for the 30h R3 tables: machine-checkable T0.1 / T0.4 tests.

Tests (ids per 30h plan §7.4, statuses per §7.3):

===========  ================================================================
``T0.1a``   FK integrity — node→curve, edge→curve/node, sample_view→curve /
            node / intent (plus intent & quality→curve). No orphans.
``T0.1b``   TS uniqueness per curve + TS two-side node order legality flag
            (canonical A → TS → B ordering, progress A≤0≤B).
``T0.1c``   No non-finite coordinates (NaN/±inf/missing) in ``irc_node`` and
            ``sample_view``.
``T0.4a``   ``split_manifest_r3.json``: the same ``curve_id`` never appears in
            two folds; likewise near-duplicate groups when that column is
            present.
``T0.4b``   ``data_manifest.json`` / ``teacher_manifest.json`` minimal
            structure: referenced files exist, ``sha256`` fields present
            (64-hex).
===========  ================================================================

Statuses: ``pass | fail | blocked | not_run``.

* ``fail``   — the check ran and found violations.
* ``blocked``— a required input (table/manifest) is missing or unusable, so
  the check cannot run to completion.
* ``not_run``— the test was excluded via ``--tests``.
* internal errors are reported as ``fail`` with an error note (a crashing
  gate must never be reported as a pass).

Bad rows are **never dropped**: every violating row is listed in
``tests[].bad_rows`` (capped at ``MAX_EVIDENCE_ROWS``) while ``counts`` stay
exact. Each test carries an ``evidence`` pointer into the report JSON plus
``inputs`` (the source files it read).

Exit code 0 iff every executed test passed.

Usage::

    python -m scripts.agent_ops.data_gate.validate_r3_tables \
        --tables-dir DIR --report OUT.json [--tests T0.1a ...]

Dry-run scope: no truth-data access, no ACP/engine calls, stdlib + pandas +
numpy only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # package execution: python -m scripts.agent_ops.data_gate...
    from scripts.agent_ops.data_gate import r3_table_schema as schema
except ImportError:  # direct execution: python .../validate_r3_tables.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import r3_table_schema as schema  # type: ignore

TEST_IDS: Tuple[str, ...] = ("T0.1a", "T0.1b", "T0.1c", "T0.4a", "T0.4b")
STATUS_VALUES: Tuple[str, ...] = ("pass", "fail", "blocked", "not_run")

MAX_EVIDENCE_ROWS = 50
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SIDE_EPS = 1e-6

# Single-column FK relations: (child, child_cols, parent, parent_cols).
COLUMN_FK: Tuple[Tuple[str, Tuple[str, ...], str, Tuple[str, ...]], ...] = (
    ("irc_node", ("curve_id",), "irc_curve", ("curve_id",)),
    ("irc_edge", ("curve_id",), "irc_curve", ("curve_id",)),
    ("reaction_intent", ("curve_id",), "irc_curve", ("curve_id",)),
    ("curve_quality", ("curve_id",), "irc_curve", ("curve_id",)),
    ("sample_view", ("curve_id",), "irc_curve", ("curve_id",)),
    ("sample_view", ("intent_id",), "reaction_intent", ("intent_id",)),
)

ALL_FK = COLUMN_FK + schema.COMPOSITE_FK


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _relation_label(rel: Tuple[str, Tuple[str, ...], str, Tuple[str, ...]]) -> str:
    child, ccols, parent, pcols = rel
    return (
        f"{child}.({'+'.join(ccols)}) -> {parent}.({'+'.join(pcols)})"
    )


def _canon_str(series: pd.Series) -> pd.Series:
    return series.map(lambda v: "<NULL>" if pd.isna(v) else str(v))


def _canon_idx(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")

    def _fmt(v: Any) -> str:
        if pd.isna(v):
            return "<NULL>"
        fv = float(v)
        return str(int(fv)) if fv.is_integer() else repr(fv)

    return num.map(_fmt)


def _result(
    test_id: str,
    status: str,
    *,
    inputs: Sequence[Path] = (),
    counts: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
    bad_rows: Optional[List[Dict[str, Any]]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    assert status in STATUS_VALUES, status
    out: Dict[str, Any] = {
        "id": test_id,
        "status": status,
        "counts": counts or {},
        "bad_rows": bad_rows or [],
        "details": details or {},
        "inputs": [str(p) for p in inputs],
        "evidence": None,  # filled in by run_validation
    }
    if note:
        out["note"] = note
    return out


def _blocked(test_id: str, reason: str, inputs: Sequence[Path] = ()) -> Dict[str, Any]:
    return _result(test_id, "blocked", inputs=inputs, note=reason,
                   counts={"n_violations": 0})


class _Tables:
    """Lazily-loaded table cache; never raises, records load errors instead."""

    def __init__(self, tables_dir: Path) -> None:
        self.dir = Path(tables_dir)
        self.frames: Dict[str, pd.DataFrame] = {}
        self.load_errors: Dict[str, str] = {}
        self.missing_columns: Dict[str, List[str]] = {}
        self.inputs: Dict[str, Path] = {}
        for name, tsch in schema.TABLES.items():
            path = self.dir / tsch.filename
            self.inputs[name] = path
            if not path.is_file():
                self.load_errors[name] = f"missing table file: {path}"
                continue
            try:
                df = pd.read_csv(path)
            except Exception as exc:  # unreadable / malformed CSV
                self.load_errors[name] = f"unreadable table {path}: {exc}"
                continue
            missing = [c for c in tsch.required_columns if c not in df.columns]
            if missing:
                self.missing_columns[name] = missing
            self.frames[name] = df

    def problem(self, *names: str) -> Optional[str]:
        """Blocking reason when a table file is missing/unreadable, else None."""
        for name in names:
            if name in self.load_errors:
                return self.load_errors[name]
        return None

    def column_problem(
        self, need: Dict[str, Sequence[str]]
    ) -> Optional[Dict[str, List[str]]]:
        """Schema violations among *present* tables: {table: missing columns}."""
        out: Dict[str, List[str]] = {}
        for name, cols in need.items():
            df = self.frames.get(name)
            if df is None:
                continue  # missing file is a load problem, not a schema one
            missing = [c for c in cols if c not in df.columns]
            if missing:
                out[name] = missing
        return out or None

    def schema_issue_rows(
        self, miss: Dict[str, List[str]], relation: str = "schema"
    ) -> List[Dict[str, Any]]:
        return [
            {"table": table, "relation": relation, "column": col,
             "reason": "missing_required_column"}
            for table, cols in sorted(miss.items())
            for col in cols
        ]

    def input_paths(self, *names: str) -> List[Path]:
        return [self.inputs[n] for n in names if n in self.inputs]


# ---------------------------------------------------------------------------
# T0.1a — FK integrity
# ---------------------------------------------------------------------------

def _check_relation(
    tables: _Tables, rel: Tuple[str, Tuple[str, ...], str, Tuple[str, ...]]
) -> Tuple[str, List[Dict[str, Any]], Optional[str], List[Dict[str, Any]]]:
    """Returns (state, bad_rows, load_reason, schema_rows).

    state is ``"checked"`` | ``"load"`` (table file missing/unreadable →
    blocked) | ``"schema"`` (required column absent in an existing file →
    fail).
    """
    child, ccols, parent, pcols = rel
    reason = tables.problem(child, parent)
    if reason is not None:
        return "load", [], reason, []
    cdf, pdf = tables.frames[child], tables.frames[parent]
    miss: Dict[str, List[str]] = {}
    for name, cols, df in ((child, ccols, cdf), (parent, pcols, pdf)):
        absent = [c for c in cols if c not in df.columns]
        if absent:
            miss[name] = absent
    if miss:
        return "schema", [], None, [
            {"table": t, "relation": _relation_label(rel), "column": c,
             "reason": "missing_required_column"}
            for t, cols in sorted(miss.items())
            for c in cols
        ]
    child_keys = list(zip(*[_key_series(cdf, c) for c in ccols]))
    parent_keys = set(zip(*[_key_series(pdf, c) for c in pcols]))
    bad: List[Dict[str, Any]] = []
    label = _relation_label(rel)
    for pos, key in enumerate(child_keys):
        if key not in parent_keys:
            bad.append({
                "table": child,
                "row_index": int(pos),
                "relation": label,
                "value": "|".join(key),
            })
    return "checked", bad, None, []


def _key_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Canonical string keys: ids as-is, integer indices zero-padded-free."""
    if col.endswith("_id") or col in ("curve_id", "intent_id", "sample_id",
                                      "canonical_irc_id"):
        return _canon_str(df[col])
    # numeric index columns (node_index, src_node_index, ...)
    if pd.api.types.is_numeric_dtype(df[col]):
        return _canon_idx(df[col])
    return _canon_str(df[col])


def _test_t0_1a(tables: _Tables, report_path: Path) -> Dict[str, Any]:
    inputs = tables.input_paths(*schema.TABLE_ORDER)
    checked: List[str] = []
    unchecked: List[Dict[str, str]] = []
    schema_relations: List[str] = []
    bad_rows: List[Dict[str, Any]] = []
    # whole-schema conformance: every present table must carry its required
    # columns (this also covers primary-key columns such as sample_id).
    schema_rows: List[Dict[str, Any]] = []
    for name, tsch in schema.TABLES.items():
        df = tables.frames.get(name)
        if df is None:
            continue  # missing/unreadable file → blocked path below
        for col in tsch.required_columns:
            if col not in df.columns:
                schema_rows.append({
                    "table": name,
                    "relation": f"schema:{name}.required_columns",
                    "column": col,
                    "reason": "missing_required_column",
                })
    n_bad = 0
    for rel in ALL_FK:
        label = _relation_label(rel)
        state, rows, load_reason, miss_rows = _check_relation(tables, rel)
        if state == "checked":
            checked.append(label)
            n_bad += len(rows)
            bad_rows.extend(rows)
        elif state == "schema":
            schema_relations.append(label)
        else:
            unchecked.append({"relation": label, "reason": load_reason or "unknown"})
    counts = {
        "n_relations_total": len(ALL_FK),
        "n_relations_checked": len(checked),
        "n_relations_schema_invalid": len(schema_relations),
        "n_relations_unchecked_missing_input": len(unchecked),
        "n_orphan_rows": n_bad,
        "n_schema_violations": len(schema_rows),
        "n_bad_rows_reported": len(bad_rows[:MAX_EVIDENCE_ROWS]),
    }
    reported = (bad_rows + schema_rows)[:MAX_EVIDENCE_ROWS]
    details = {
        "checked_relations": checked,
        "unchecked_relations": unchecked,
    }
    if n_bad > 0 or schema_rows:
        status = "fail"
        parts = []
        if n_bad:
            parts.append(f"{n_bad} orphan FK row(s)")
        if schema_rows:
            parts.append(f"{len(schema_rows)} schema violation(s)")
        note = "; ".join(parts) + " — nothing dropped, see bad_rows"
    elif unchecked:
        status = "blocked"
        note = ("uncheckable relations: "
                + "; ".join(f"{u['relation']} ({u['reason']})" for u in unchecked))
    else:
        status = "pass"
        note = None
    return _result(
        "T0.1a", status, inputs=inputs, counts=counts,
        details=details, bad_rows=reported, note=note,
    )


# ---------------------------------------------------------------------------
# T0.1b — TS uniqueness + two-side order legality
# ---------------------------------------------------------------------------

def _side_order_violations(
    nodes: pd.DataFrame, ts_index: int, eps: float = SIDE_EPS
) -> List[Dict[str, Any]]:
    bad: List[Dict[str, Any]] = []
    for _, row in nodes.iterrows():
        ni = pd.to_numeric(pd.Series([row["node_index"]]), errors="coerce").iloc[0]
        prog = pd.to_numeric(pd.Series([row["progress"]]), errors="coerce").iloc[0]
        if pd.isna(ni):
            bad.append({"node_index": None, "reason": "non_numeric_node_index"})
            continue
        ni = int(ni)
        if pd.isna(prog) or not np.isfinite(float(prog)):
            bad.append({"node_index": ni, "reason": "non_finite_progress"})
            continue
        prog = float(prog)
        if ni < ts_index and prog > eps:
            bad.append({"node_index": ni, "reason": "side_A_progress_positive",
                        "progress": prog})
        elif ni > ts_index and prog < -eps:
            bad.append({"node_index": ni, "reason": "side_B_progress_negative",
                        "progress": prog})
    return bad


_T0_1B_NEED: Dict[str, Tuple[str, ...]] = {
    "irc_curve": ("curve_id", "ts_index"),
    "irc_node": ("curve_id", "node_index", "progress", "is_ts"),
}


def _schema_fail(
    test_id: str, need: Dict[str, Sequence[str]], tables: _Tables,
    miss: Dict[str, List[str]],
) -> Dict[str, Any]:
    rows = tables.schema_issue_rows(miss, relation=f"{test_id}:schema")
    return _result(
        test_id, "fail", inputs=tables.input_paths(*need),
        counts={"n_violations": len(rows),
                "n_schema_violations": len(rows),
                "n_bad_rows_reported": len(rows)},
        details={"missing_columns": miss},
        bad_rows=rows,
        note=f"schema violation: missing required columns {miss}",
    )


def _test_t0_1b(tables: _Tables, report_path: Path) -> Dict[str, Any]:
    inputs = tables.input_paths("irc_curve", "irc_node")
    reason = tables.problem("irc_curve", "irc_node")
    if reason is not None:
        return _blocked("T0.1b", reason, inputs=inputs)
    miss = tables.column_problem(_T0_1B_NEED)
    if miss:
        return _schema_fail("T0.1b", _T0_1B_NEED, tables, miss)
    curve_df = tables.frames["irc_curve"]
    node_df = tables.frames["irc_node"]
    node_df = node_df.copy()
    node_df["_cid"] = _canon_str(node_df["curve_id"])
    node_df["_ni"] = pd.to_numeric(node_df["node_index"], errors="coerce")
    node_df["_is_ts"] = pd.to_numeric(node_df["is_ts"], errors="coerce").fillna(0)
    grouped = dict(list(node_df.groupby("_cid", sort=False)))

    violations: List[Dict[str, Any]] = []
    per_curve: Dict[str, Any] = {}
    n_dup = n_missing = n_illegal = n_mismatch = 0
    for _, crow in curve_df.iterrows():
        cid = str(crow["curve_id"])
        ts_index_raw = pd.to_numeric(
            pd.Series([crow["ts_index"]]), errors="coerce"
        ).iloc[0]
        ts_index = int(ts_index_raw) if not pd.isna(ts_index_raw) else None
        nodes = grouped.get(cid)
        if nodes is None or len(nodes) == 0:
            violations.append({"curve_id": cid, "reason": "curve_without_nodes"})
            per_curve[cid] = {"n_nodes": 0, "ts_side_order_legal": False}
            n_missing += 1
            continue
        ts_nodes = sorted(
            int(v) for v in nodes.loc[nodes["_is_ts"] > 0, "_ni"].dropna().tolist()
        )
        reasons: List[str] = []
        if len(ts_nodes) == 0:
            reasons.append("missing_ts")
            n_missing += 1
        elif len(ts_nodes) > 1:
            reasons.append(f"duplicate_ts:{ts_nodes}")
            n_dup += 1
        elif ts_index is not None and ts_nodes[0] != ts_index:
            reasons.append(f"ts_index_mismatch:node={ts_nodes[0]},curve.ts_index={ts_index}")
            n_mismatch += 1
        # two-side order legality relative to the curve's canonical TS index
        ref_ts = ts_index if ts_index is not None else (
            ts_nodes[0] if len(ts_nodes) == 1 else None
        )
        side_bad: List[Dict[str, Any]] = []
        if ref_ts is not None:
            side_bad = _side_order_violations(nodes, ref_ts)
        if side_bad:
            reasons.append("ts_side_order_illegal")
            n_illegal += 1
        per_curve[cid] = {
            "n_nodes": int(len(nodes)),
            "ts_nodes": ts_nodes,
            "ts_index_curve": ts_index,
            "ts_side_order_legal": not side_bad,
            "side_order_violations": side_bad[:MAX_EVIDENCE_ROWS],
        }
        for r in reasons:
            violations.append({"curve_id": cid, "reason": r})
    counts = {
        "n_curves": int(len(curve_df)),
        "n_curves_duplicate_ts": n_dup,
        "n_curves_missing_ts": n_missing,
        "n_curves_ts_index_mismatch": n_mismatch,
        "n_curves_ts_side_order_illegal": n_illegal,
        "n_violations": len(violations),
        "n_bad_rows_reported": len(violations[:MAX_EVIDENCE_ROWS]),
    }
    status = "fail" if violations else "pass"
    return _result(
        "T0.1b", status, inputs=inputs, counts=counts,
        details={"per_curve": per_curve},
        bad_rows=violations[:MAX_EVIDENCE_ROWS],
        note=None if not violations else
        f"{len(violations)} TS violation(s); see bad_rows",
    )


# ---------------------------------------------------------------------------
# T0.1c — finite coordinates
# ---------------------------------------------------------------------------

_T0_1C_NEED: Dict[str, Tuple[str, ...]] = {
    "irc_node": ("x", "y", "z"),
    "sample_view": ("x", "y", "z"),
}


def _test_t0_1c(tables: _Tables, report_path: Path) -> Dict[str, Any]:
    inputs = tables.input_paths("irc_node", "sample_view")
    reason = tables.problem("irc_node", "sample_view")
    if reason is not None:
        return _blocked("T0.1c", reason, inputs=inputs)
    miss = tables.column_problem(_T0_1C_NEED)
    if miss:
        return _schema_fail("T0.1c", _T0_1C_NEED, tables, miss)
    bad_rows: List[Dict[str, Any]] = []
    per_table: Dict[str, Dict[str, int]] = {}
    n_bad = 0
    for table, key_cols in (("irc_node", ("curve_id", "node_index")),
                            ("sample_view", ("sample_id",))):
        df = tables.frames[table]
        per_table[table] = {}
        for col in ("x", "y", "z"):
            vals = pd.to_numeric(df[col], errors="coerce")
            finite = np.isfinite(vals.to_numpy(dtype="float64", na_value=np.nan))
            n_col = int((~finite).sum())
            per_table[table][col] = n_col
            n_bad += n_col
            for pos in np.nonzero(~finite)[0].tolist():
                entry: Dict[str, Any] = {
                    "table": table,
                    "row_index": int(pos),
                    "column": col,
                    "value": str(df[col].iloc[pos]),
                }
                for kc in key_cols:
                    if kc in df.columns:
                        entry[kc] = str(df[kc].iloc[pos])
                bad_rows.append(entry)
    counts = {
        "n_tables_checked": 2,
        "n_non_finite_coordinates": n_bad,
        "n_bad_rows_reported": len(bad_rows[:MAX_EVIDENCE_ROWS]),
        "per_table_per_column": per_table,
    }
    status = "fail" if n_bad else "pass"
    return _result(
        "T0.1c", status, inputs=inputs, counts=counts,
        bad_rows=bad_rows[:MAX_EVIDENCE_ROWS],
        note=None if not n_bad else
        f"{n_bad} non-finite coordinate(s); rows reported, never dropped",
    )


# ---------------------------------------------------------------------------
# T0.4a — split manifest fold leakage
# ---------------------------------------------------------------------------

def _split_manifest_path(tables: _Tables) -> Tuple[Optional[Path], Optional[str]]:
    primary = tables.dir / schema.SPLIT_MANIFEST_FILENAME
    if primary.is_file():
        return primary, None
    fallback = tables.dir / schema.SPLIT_MANIFEST_FALLBACK_FILENAME
    if fallback.is_file():
        return fallback, (
            f"{schema.SPLIT_MANIFEST_FILENAME} absent; using fallback "
            f"{schema.SPLIT_MANIFEST_FALLBACK_FILENAME} (plan §3.1 vs §7.1 name "
            "discrepancy, see schema_notes)"
        )
    return None, (
        f"missing {schema.SPLIT_MANIFEST_FILENAME} (and fallback "
        f"{schema.SPLIT_MANIFEST_FALLBACK_FILENAME}) in {tables.dir}"
    )


def _fold_mapping(data: Any) -> Tuple[Optional[Dict[str, List[Any]]], str, Optional[str]]:
    """Normalize supported manifest shapes to {curve_id: [fold, ...]}."""
    assignments: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        cf = data.get("curve_folds")
        if isinstance(cf, dict):
            for cid, fold in cf.items():
                assignments.append({"curve_id": cid, "fold": fold})
            return _group_folds(assignments), "curve_folds", None
        folds = data.get("folds")
        if isinstance(folds, dict):
            for fold, cids in folds.items():
                if not isinstance(cids, list):
                    return None, "folds", f"folds[{fold!r}] is not a list"
                for cid in cids:
                    assignments.append({"curve_id": cid, "fold": fold})
            return _group_folds(assignments), "folds", None
        raw = data.get("assignments")
        if isinstance(raw, list):
            assignments = _normalize_assignments(raw)
            if assignments is None:
                return None, "assignments", "assignment records lack curve_id/fold"
            return _group_folds(assignments), "assignments", None
        return None, "unknown", (
            "no recognized fold structure "
            "(expected one of: curve_folds, folds, assignments)"
        )
    if isinstance(data, list):
        assignments = _normalize_assignments(data)
        if assignments is None:
            return None, "assignment_list", "assignment records lack curve_id/fold"
        return _group_folds(assignments), "assignment_list", None
    return None, "unknown", "top-level JSON must be an object or a list"


def _normalize_assignments(raw: List[Any]) -> Optional[List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or "curve_id" not in item or "fold" not in item:
            return None
        out.append(item)
    return out


def _group_folds(assignments: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    mapping: Dict[str, List[Any]] = {}
    for a in assignments:
        mapping.setdefault(str(a["curve_id"]), []).append(a["fold"])
    return mapping


def _near_dup_groups(data: Any) -> Optional[Dict[str, List[str]]]:
    if isinstance(data, dict):
        ng = data.get("near_dup_groups")
        if isinstance(ng, dict):
            return {str(k): [str(c) for c in v] for k, v in ng.items()
                    if isinstance(v, list)}
    # assignment-level "near_dup_group" column
    records: List[Any] = []
    if isinstance(data, dict) and isinstance(data.get("assignments"), list):
        records = data["assignments"]
    elif isinstance(data, list):
        records = data
    groups: Dict[str, List[str]] = {}
    for item in records:
        if isinstance(item, dict) and "near_dup_group" in item:
            groups.setdefault(str(item["near_dup_group"]), []).append(
                str(item.get("curve_id"))
            )
    return groups or None


def _test_t0_4a(tables: _Tables, report_path: Path) -> Dict[str, Any]:
    path, path_note = _split_manifest_path(tables)
    if path is None:
        return _blocked("T0.4a", path_note or "split manifest missing",
                        inputs=[tables.dir / schema.SPLIT_MANIFEST_FILENAME])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _result(
            "T0.4a", "fail", inputs=[path],
            counts={"n_violations": 1, "n_bad_rows_reported": 1},
            bad_rows=[{"manifest": str(path), "reason": f"unparseable_json: {exc}"}],
            note="split manifest is not valid JSON",
        )
    mapping, shape, shape_err = _fold_mapping(data)
    if mapping is None:
        return _result(
            "T0.4a", "fail", inputs=[path],
            counts={"n_violations": 1, "n_bad_rows_reported": 1,
                    "manifest_shape": shape},
            bad_rows=[{"manifest": str(path), "reason": shape_err or "unknown"}],
            note="split manifest structure not recognized",
            details={"manifest": str(path), "shape": shape},
        )
    bad_rows: List[Dict[str, Any]] = []
    for cid, folds in sorted(mapping.items()):
        distinct = list(dict.fromkeys(str(f) for f in folds))
        if len(distinct) > 1:
            bad_rows.append({
                "curve_id": cid,
                "folds": distinct,
                "reason": "curve_in_two_folds",
            })
    groups = _near_dup_groups(data)
    n_groups = 0
    n_group_viol = 0
    if groups:
        n_groups = len(groups)
        for gid, cids in sorted(groups.items()):
            seen: List[str] = []
            for cid in cids:
                for f in mapping.get(str(cid), []):
                    fs = str(f)
                    if fs not in seen:
                        seen.append(fs)
            if len(seen) > 1:
                n_group_viol += 1
                bad_rows.append({
                    "near_dup_group": gid,
                    "curve_ids": cids,
                    "folds": seen,
                    "reason": "near_dup_group_split_across_folds",
                })
    n_bad = len(bad_rows)
    counts = {
        "n_curves_in_manifest": len(mapping),
        "n_fold_values": len({str(f) for fs in mapping.values() for f in fs}),
        "n_near_dup_groups": n_groups,
        "n_violations": n_bad,
        "n_bad_rows_reported": len(bad_rows[:MAX_EVIDENCE_ROWS]),
    }
    details: Dict[str, Any] = {
        "manifest": str(path),
        "manifest_shape": shape,
        "near_dup_column_present": bool(groups),
    }
    if path_note:
        details["path_note"] = path_note
    note = None
    if n_bad:
        note = f"{n_bad} split leakage violation(s); see bad_rows"
    elif not groups:
        note = "near_dup_group column not present in manifest: near-dup check skipped"
    return _result("T0.4a", "fail" if n_bad else "pass", inputs=[path],
                   counts=counts, details=details,
                   bad_rows=bad_rows[:MAX_EVIDENCE_ROWS], note=note)


# ---------------------------------------------------------------------------
# T0.4b — data/teacher manifest structure
# ---------------------------------------------------------------------------

def _manifest_entries(data: Any) -> Tuple[Optional[List[Tuple[str, Any]]], Optional[str]]:
    if not isinstance(data, dict):
        return None, "top-level JSON must be an object with a 'files' member"
    files = data.get(schema.MANIFEST_ENTRY_LIST_KEY)
    if isinstance(files, list):
        entries: List[Tuple[str, Any]] = []
        for i, item in enumerate(files):
            if not isinstance(item, dict):
                return None, f"files[{i}] is not an object"
            entries.append((f"files[{i}]", item))
        return entries, None
    if isinstance(files, dict):
        return [(f"files[{k!r}]", {schema.MANIFEST_SHA_KEY: v, "path": k})
                for k, v in files.items()], None
    return None, f"missing or invalid '{schema.MANIFEST_ENTRY_LIST_KEY}' member"


def _check_manifest(path: Path) -> List[Dict[str, Any]]:
    bad: List[Dict[str, Any]] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [{"manifest": str(path), "reason": f"unparseable_json: {exc}"}]
    entries, err = _manifest_entries(data)
    if entries is None:
        return [{"manifest": str(path), "reason": err or "unrecognized structure"}]
    for label, item in entries:
        sha = item.get(schema.MANIFEST_SHA_KEY)
        if not isinstance(sha, str) or not SHA256_RE.match(sha):
            bad.append({
                "manifest": str(path),
                "entry": label,
                "reason": "missing_or_malformed_sha256",
                "value": str(sha),
            })
        rel = item.get("path")
        if not isinstance(rel, str) or not rel:
            bad.append({"manifest": str(path), "entry": label,
                        "reason": "missing_path_field"})
            continue
        target = Path(rel)
        if not target.is_absolute():
            target = path.parent / target
        if not target.is_file():
            bad.append({
                "manifest": str(path),
                "entry": label,
                "reason": "referenced_file_missing",
                "value": rel,
            })
    if not entries:
        bad.append({"manifest": str(path), "reason": "files list is empty"})
    return bad


def _test_t0_4b(tables: _Tables, report_path: Path) -> Dict[str, Any]:
    paths = [tables.dir / name for name in schema.REQUIRED_MANIFESTS]
    present = [p for p in paths if p.is_file()]
    missing = [p for p in paths if not p.is_file()]
    bad_rows: List[Dict[str, Any]] = []
    for p in present:
        bad_rows.extend(_check_manifest(p))
    counts = {
        "n_manifests_expected": len(paths),
        "n_manifests_present": len(present),
        "n_manifests_missing": len(missing),
        "n_violations": len(bad_rows),
        "n_bad_rows_reported": len(bad_rows[:MAX_EVIDENCE_ROWS]),
    }
    details = {
        "manifests": {p.name: str(p) for p in present},
        "missing": [str(p) for p in missing],
    }
    inputs = paths
    if bad_rows:
        return _result("T0.4b", "fail", inputs=inputs, counts=counts,
                       details=details, bad_rows=bad_rows[:MAX_EVIDENCE_ROWS],
                       note=f"{len(bad_rows)} manifest violation(s); see bad_rows")
    if missing:
        return _blocked(
            "T0.4b",
            "missing manifest(s): " + ", ".join(p.name for p in missing),
            inputs=inputs,
        )
    return _result("T0.4b", "pass", inputs=inputs, counts=counts, details=details)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

_RUNNERS = {
    "T0.1a": _test_t0_1a,
    "T0.1b": _test_t0_1b,
    "T0.1c": _test_t0_1c,
    "T0.4a": _test_t0_4a,
    "T0.4b": _test_t0_4b,
}


def run_validation(
    tables_dir: Path,
    report_path: Path,
    only_tests: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run all (or the selected) gate tests, write the JSON report, return it."""
    tables_dir = Path(tables_dir)
    report_path = Path(report_path)
    tables = _Tables(tables_dir)
    only = set(only_tests) if only_tests else None

    tests: List[Dict[str, Any]] = []
    for i, tid in enumerate(TEST_IDS):
        if only is not None and tid not in only:
            res = _result(tid, "not_run", note=f"excluded by --tests filter")
        else:
            try:
                res = _RUNNERS[tid](tables, report_path)
            except Exception as exc:  # a crashing gate must not pass
                res = _result(tid, "fail",
                              note=f"internal error: {type(exc).__name__}: {exc}")
        res["evidence"] = f"{report_path}#/tests/{i}"
        tests.append(res)

    summary = Counter(t["status"] for t in tests)
    all_pass = bool(tests) and all(t["status"] == "pass" for t in tests)
    report: Dict[str, Any] = {
        "report_type": "pes2ts_r3_data_gate",
        "validator_version": schema.VALIDATOR_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
        "schema_source": list(schema.SCHEMA_SOURCE),
        "schema_notes": list(schema.SCHEMA_NOTES),
        "tables_dir": str(tables_dir),
        "report_path": str(report_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_ids": list(TEST_IDS),
        "tests": tests,
        "summary": {
            "total": len(tests),
            "pass": summary.get("pass", 0),
            "fail": summary.get("fail", 0),
            "blocked": summary.get("blocked", 0),
            "not_run": summary.get("not_run", 0),
            "all_pass": all_pass,
        },
        "exit_code": 0 if all_pass else 1,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_r3_tables",
        description=(
            "Data Gate validator for the 30h R3 tables "
            "(T0.1a/b/c FK+TS+finite, T0.4a/b split+manifests)."
        ),
    )
    parser.add_argument("--tables-dir", required=True,
                        help="directory holding the six CSV tables + 3 JSON manifests")
    parser.add_argument("--report", required=True,
                        help="output JSON report path")
    parser.add_argument("--tests", nargs="+", choices=list(TEST_IDS), default=None,
                        help="run only these test ids (others are reported not_run)")
    args = parser.parse_args(argv)

    report = run_validation(args.tables_dir, args.report, args.tests)
    s = report["summary"]
    print(
        f"[r3-data-gate] {report['exit_code']}: "
        f"pass={s['pass']} fail={s['fail']} blocked={s['blocked']} "
        f"not_run={s['not_run']} -> {args.report}"
    )
    for t in report["tests"]:
        print(f"  {t['id']}: {t['status']}"
              + (f" ({t['note']})" if t.get("note") else ""))
    return int(report["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
