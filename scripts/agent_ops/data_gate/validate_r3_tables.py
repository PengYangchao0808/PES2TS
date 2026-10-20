"""Data Gate CLI for the 30h R3 tables: machine-checkable T0.1–T0.5 gates.

Round 2 (WP-D2): implements the **ORA-2** blockers against schema
``r3_6table_v2`` (PD-9 flat ``delta_xyz`` encoding, intent pairing, label /
weight invariants, d0/gauge rules, truth-access audit).

Tests (ids: T0.1*/*T0.4* per 30h plan §7.4; T0.2a/b, T0.3a/b refine T0.2/T0.3;
T0.5 comes from ORA-2):

===========  ================================================================
``T0.1a``   FK integrity — node→curve, edge→curve/node/intent,
            sample_view→curve/node/intent (plus intent & quality→curve, and
            reverse_of self-FK is owned by T0.2a). No orphans. Also
            count/weight self-consistency: unique ``(curve_id, intent_id)``
            on reaction_intent, ``sum(curve_weight) == 1.0`` per
            ``(curve_id, d0)`` (ORA-2 #3 — bidirectional augmentation must
            not double count curves), ``feature_sha``/``label_sha`` are
            64-hex.
``T0.1b``   TS uniqueness per curve + TS two-side node order legality flag
            (canonical A → TS → B ordering via ``ts_index``, PD-9 ts
            authority; progress A≤0≤B, eps 1e-6).
``T0.1c``   No non-finite coordinates in ``irc_node`` / ``sample_view``;
            **PD-9 flat encoding checks** on ``irc_edge`` (ORA-2 #1):
            ``units == "angstrom"``, shape parses as ``[M,3]``,
            ``len(delta_xyz_flat) == prod(shape)``, offsets ``[start,end)``
            with ``end-start == len(flat)``, starts non-decreasing in file
            order (first row start==0), ``delta_xyz_sha256`` equals
            ``sha256(struct.pack('<{M*3}d', *flat))``, ``atom_map_sha256``
            is 64-hex.
``T0.2a``   Intent involution (ORA-2 #2): each curve has exactly 2 intents
            forming one pair (forward+reverse, shared ``intent_pair_id``)
            and ``reverse_of_intent_id`` chains satisfy
            ``reverse(reverse(x)) == x`` with mutual, non-self reverses —
            curves flagged ``intent_underdetermined`` are exempt.
``T0.2b``   Reversed single-step teacher consistency (ORA-2 #4): for every
            edge (i→j) under the forward intent there is the mirrored edge
            (j→i) under the reverse intent whose ``delta_xyz_flat`` is the
            negation within ``1e-6`` (PD-9), after the same-reference-frame
            flag (``same_ref_frame``) holds both ways.
``T0.3a``   d0 gate (ORA-2 #5): ``d0 == 0.08`` for all main-task rows;
            any other d0 must carry ``sensitivity_only=true``; at least one
            main row must exist.
``T0.3b``   gauge gate (ORA-2 #5): ``gauge_ref ∈ {local_frame,
            unresolved_masked}``, ``label_status ∈ {ok, masked, terminal,
            conflict}``, and ``unresolved_masked`` ⇒ ``label_status ==
            masked`` (never a silent lab frame).
``T0.4a``   ``split_manifest_r3.json``: the same ``curve_id`` never appears in
            two folds; likewise near-duplicate groups when that column is
            present.
``T0.4b``   ``data_manifest.json`` / ``teacher_manifest.json`` minimal
            structure: referenced files exist, ``sha256`` fields present
            (64-hex).
``T0.5``    truth-access audit (ORA-2 #6, constitution R1): every entry of
            the truth access log must name an accessor inside
            ``TRUTH_ACCESSOR_ALLOWLIST`` (``truth_reader`` only).  With an
            explicit ``--truth-log`` a missing log is **blocked**
            (fail-closed, never silently green).  Without the option the
            log is looked up at ``<tables-dir>/truth_access_log.jsonl`` then
            ``docs/agent_runs/truth_access_log.jsonl`` (cwd-relative); if
            neither exists the audit is ``not_run``.
===========  ================================================================

Statuses: ``pass | fail | blocked | not_run``.

* ``fail``   — the check ran and found violations.
* ``blocked``— a required input (table/manifest, or an explicitly requested
  truth log) is missing or unusable, so the check cannot run.
* ``not_run``— the test was excluded via ``--tests``, or T0.5 found no log
  to audit without an explicit request.
* internal errors are reported as ``fail`` with an error note (a crashing
  gate must never be reported as a pass).

Bad rows are **never dropped**: every violating row is listed in
``tests[].bad_rows`` (capped at ``MAX_EVIDENCE_ROWS``) while ``counts`` stay
exact. Each test carries an ``evidence`` pointer into the report JSON plus
``inputs`` (the source files it read).

Exit code 0 iff every executed test passed.

Usage::

    python3 -m scripts.agent_ops.data_gate.validate_r3_tables \\
        --tables-dir DIR --report OUT.json \\
        [--tests T0.1a ...] [--truth-log PATH]

Dry-run scope: no truth-data access, no ACP/engine calls, stdlib + pandas +
numpy only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # package execution: python3 -m scripts.agent_ops.data_gate...
    from scripts.agent_ops.data_gate import r3_table_schema as schema
except ImportError:  # direct execution: python3 .../validate_r3_tables.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import r3_table_schema as schema  # type: ignore

TEST_IDS: Tuple[str, ...] = (
    "T0.1a",
    "T0.1b",
    "T0.1c",
    "T0.2a",
    "T0.2b",
    "T0.3a",
    "T0.3b",
    "T0.4a",
    "T0.4b",
    "T0.5",
)
STATUS_VALUES: Tuple[str, ...] = ("pass", "fail", "blocked", "not_run")

MAX_EVIDENCE_ROWS = 50
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SHAPE_RE = re.compile(r"^\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*$")
SIDE_EPS = schema.DELTA_EPS

# Single-column FK relations: (child, child_cols, parent, parent_cols).
# NOTE: reaction_intent.reverse_of_intent_id is deliberately NOT listed here —
# exempt (intent_underdetermined) rows may leave it empty; T0.2a owns both the
# exemption rule and reverse-chain resolution.
COLUMN_FK: Tuple[Tuple[str, Tuple[str, ...], str, Tuple[str, ...]], ...] = (
    ("irc_node", ("curve_id",), "irc_curve", ("curve_id",)),
    ("irc_edge", ("curve_id",), "irc_curve", ("curve_id",)),
    ("irc_edge", ("intent_id",), "reaction_intent", ("intent_id",)),
    ("reaction_intent", ("curve_id",), "irc_curve", ("curve_id",)),
    ("curve_quality", ("curve_id",), "irc_curve", ("curve_id",)),
    ("sample_view", ("curve_id",), "irc_curve", ("curve_id",)),
    ("sample_view", ("intent_id",), "reaction_intent", ("intent_id",)),
)

ALL_FK = COLUMN_FK + schema.COMPOSITE_FK

PD9_EDGE_COLUMNS: Tuple[str, ...] = (
    "delta_xyz_flat",
    "delta_xyz_shape",
    "delta_xyz_offsets",
    "atom_map_sha256",
    "delta_xyz_sha256",
    "units",
    "same_ref_frame",
)

TRUTH_LOG_CANDIDATES_NOTE = (
    f"explicit --truth-log, else <tables-dir>/{schema.TRUTH_LOG_TABLES_DIR_NAME}, "
    f"else {schema.TRUTH_LOG_DEFAULT_SUBPATH} (cwd-relative)"
)


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


def _scalar_num(v: Any) -> Optional[float]:
    """Numeric coercion of a single CSV cell (None if not numeric)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(
        v, (bool, np.bool_)
    ):
        f = float(v)
        return f if math.isfinite(f) else None
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


_TRUE_TOKENS = {"true", "1", "yes", "y"}
_FALSE_TOKENS = {"false", "0", "no", "n", ""}


def _parse_bool(v: Any) -> Optional[bool]:
    """Parse a CSV cell into bool; None if unparseable (NaN counts as None)."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        if float(v) in (0.0, 1.0):
            return bool(v)
        return None
    s = str(v).strip().lower()
    if s in _TRUE_TOKENS:
        return True
    if s in _FALSE_TOKENS:
        return False
    return None


def _parse_flat_list(v: Any) -> Tuple[Optional[List[float]], Optional[str]]:
    """Parse delta_xyz_flat: JSON list of finite floats."""
    try:
        data = json.loads(str(v))
    except Exception as exc:
        return None, f"unparsable_json: {exc}"
    if not isinstance(data, list):
        return None, "not_a_list"
    out: List[float] = []
    for item in data:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None, "non_numeric_entry"
        f = float(item)
        if not math.isfinite(f):
            return None, "non_finite_entry"
        out.append(f)
    return out, None


def _parse_offsets(v: Any) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
    """Parse delta_xyz_offsets: JSON [start, end) with int-valued entries."""
    try:
        data = json.loads(str(v))
    except Exception as exc:
        return None, f"unparsable_json: {exc}"
    if not isinstance(data, list) or len(data) != 2:
        return None, "not_a_2_element_list"
    start, end = data
    for item in (start, end):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None, "non_numeric_entry"
        if float(item) != int(item):
            return None, "non_integer_entry"
    return (int(start), int(end)), None


def _parse_shape(v: Any) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
    m = SHAPE_RE.match(str(v))
    if not m:
        return None, "shape_not_[M,3]"
    m_dim, three = int(m.group(1)), int(m.group(2))
    if three != 3:
        return None, "second_dim_not_3"
    if m_dim < 1:
        return None, "empty_shape"
    return (m_dim, three), None


def _sha_flat(values: Sequence[float]) -> str:
    """PD-9 binding sha: IEEE-754 float64 little-endian bytes of the list."""
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


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


def _not_run(test_id: str, reason: str) -> Dict[str, Any]:
    return _result(test_id, "not_run", note=reason, counts={"n_violations": 0})


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


def _key_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Canonical string keys: ids as-is, integer indices zero-padded-free."""
    if col.endswith("_id") or col in ("curve_id", "intent_id", "sample_id",
                                      "canonical_irc_id"):
        return _canon_str(df[col])
    # numeric index columns (node_index, src_node_index, ...)
    if pd.api.types.is_numeric_dtype(df[col]):
        return _canon_idx(df[col])
    return _canon_str(df[col])


# ---------------------------------------------------------------------------
# T0.1a — FK integrity + count/weight self-consistency
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


def _check_curve_weights(sv_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], int]:
    """sum(curve_weight) per (curve_id, d0) group must equal CURVE_WEIGHT_TARGET."""
    bad: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str], List[Optional[float]]] = defaultdict(list)
    for _, row in sv_df.iterrows():
        cid = str(row["curve_id"])
        d0 = _scalar_num(row["d0"])
        d0k = "<NON_FINITE>" if d0 is None else f"{d0:.9f}"
        groups[(cid, d0k)].append(_scalar_num(row["curve_weight"]))
    for (cid, d0k), weights in sorted(groups.items()):
        if any(w is None or not math.isfinite(w) for w in weights):
            bad.append({
                "table": "sample_view",
                "relation": "curve_weight_sum",
                "curve_id": cid,
                "d0": d0k,
                "reason": "non_finite_curve_weight",
                "n_samples": len(weights),
            })
            continue
        total = math.fsum(w for w in weights if w is not None)
        if abs(total - schema.CURVE_WEIGHT_TARGET) > schema.WEIGHT_SUM_TOL:
            bad.append({
                "table": "sample_view",
                "relation": "curve_weight_sum",
                "curve_id": cid,
                "d0": d0k,
                "sum_curve_weight": total,
                "target": schema.CURVE_WEIGHT_TARGET,
                "n_samples": len(weights),
                "reason": "curve_weight_sum_mismatch",
            })
    return bad, len(groups)


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

    # --- count/weight self-consistency (ORA-2 #2/#3) ---
    integrity_rows: List[Dict[str, Any]] = []
    n_weight_groups = 0
    sv_df = tables.frames.get("sample_view")
    if sv_df is not None and {"curve_id", "d0", "curve_weight"} <= set(sv_df.columns):
        weight_bad, n_weight_groups = _check_curve_weights(sv_df)
        integrity_rows.extend(weight_bad)
    ri_df = tables.frames.get("reaction_intent")
    if ri_df is not None and {"curve_id", "intent_id"} <= set(ri_df.columns):
        pairs = list(zip(_canon_str(ri_df["curve_id"]), _canon_str(ri_df["intent_id"])))
        for key, n in sorted(Counter(pairs).items()):
            if n > 1:
                integrity_rows.append({
                    "table": "reaction_intent",
                    "row_index": None,
                    "relation": "unique(curve_id,intent_id)",
                    "value": "|".join(key),
                    "n_duplicates": n,
                    "reason": "duplicate_intent_key",
                })
    n_sha_bad = 0
    if sv_df is not None:
        for col in ("feature_sha", "label_sha"):
            if col not in sv_df.columns:
                continue
            for pos, val in enumerate(sv_df[col].tolist()):
                s = "" if val is None or (isinstance(val, float) and math.isnan(val)) else str(val)
                if not SHA256_RE.match(s):
                    n_sha_bad += 1
                    integrity_rows.append({
                        "table": "sample_view",
                        "row_index": int(pos),
                        "relation": "sha_format",
                        "column": col,
                        "value": s[:80],
                        "reason": "bad_sha_format",
                    })

    counts = {
        "n_relations_total": len(ALL_FK),
        "n_relations_checked": len(checked),
        "n_relations_schema_invalid": len(schema_relations),
        "n_relations_unchecked_missing_input": len(unchecked),
        "n_orphan_rows": n_bad,
        "n_schema_violations": len(schema_rows),
        "n_weight_groups": n_weight_groups,
        "n_weight_violations": sum(1 for r in integrity_rows
                                   if "curve_weight" in r["reason"]),
        "n_intent_key_violations": sum(1 for r in integrity_rows
                                       if r["reason"] == "duplicate_intent_key"),
        "n_sha_format_violations": n_sha_bad,
        "n_integrity_violations": len(integrity_rows),
        "n_bad_rows_reported": len((bad_rows + schema_rows + integrity_rows)[:MAX_EVIDENCE_ROWS]),
    }
    reported = (bad_rows + schema_rows + integrity_rows)[:MAX_EVIDENCE_ROWS]
    details = {
        "checked_relations": checked,
        "unchecked_relations": unchecked,
        "curve_weight_target": schema.CURVE_WEIGHT_TARGET,
        "weight_sum_tol": schema.WEIGHT_SUM_TOL,
    }
    parts = []
    if n_bad:
        parts.append(f"{n_bad} orphan FK row(s)")
    if schema_rows:
        parts.append(f"{len(schema_rows)} schema violation(s)")
    if integrity_rows:
        parts.append(f"{len(integrity_rows)} integrity violation(s)")
    if parts:
        status = "fail"
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
        ni = _scalar_num(row["node_index"])
        prog = _scalar_num(row["progress"])
        if ni is None:
            bad.append({"node_index": None, "reason": "non_numeric_node_index"})
            continue
        ni = int(ni)
        if prog is None:
            bad.append({"node_index": ni, "reason": "non_finite_progress"})
            continue
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


def _test_t0_1b(tables: _Tables, report_path: Path) -> Dict[str, Any]:
    inputs = tables.input_paths("irc_curve", "irc_node")
    reason = tables.problem("irc_curve", "irc_node")
    if reason is not None:
        return _blocked("T0.1b", reason, inputs=inputs)
    miss = tables.column_problem(_T0_1B_NEED)
    if miss:
        return _schema_fail("T0.1b", _T0_1B_NEED, tables, miss)
    curve_df = tables.frames["irc_curve"]
    node_df = tables.frames["irc_node"].copy()
    node_df["_cid"] = _canon_str(node_df["curve_id"])
    node_df["_ni"] = pd.to_numeric(node_df["node_index"], errors="coerce")
    node_df["_is_ts"] = pd.to_numeric(node_df["is_ts"], errors="coerce").fillna(0)
    grouped = dict(list(node_df.groupby("_cid", sort=False)))

    violations: List[Dict[str, Any]] = []
    per_curve: Dict[str, Any] = {}
    n_dup = n_missing = n_illegal = n_mismatch = 0
    for _, crow in curve_df.iterrows():
        cid = str(crow["curve_id"])
        ts_index = _scalar_num(crow["ts_index"])
        ts_index = int(ts_index) if ts_index is not None else None
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
# T0.1c — finite coordinates + PD-9 flat-encoding integrity
# ---------------------------------------------------------------------------

_T0_1C_NEED: Dict[str, Tuple[str, ...]] = {
    "irc_node": ("x", "y", "z"),
    "sample_view": ("x", "y", "z"),
    "irc_edge": PD9_EDGE_COLUMNS,
}


def _check_pd9_edges(edge_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """PD-9 per-row checks: units, shape/length, offsets, delta sha (ORA-2 #1)."""
    bad: List[Dict[str, Any]] = []
    by_reason: Counter = Counter()
    prev_start: Optional[int] = None
    for pos, row in edge_df.iterrows():
        key = {
            "table": "irc_edge",
            "row_index": int(pos),
            "curve_id": str(row.get("curve_id")),
            "intent_id": str(row.get("intent_id")),
            "edge_index": str(row.get("edge_index")),
        }

        def _vio(reason: str, **extra: Any) -> None:
            by_reason[reason] += 1
            bad.append({**key, "reason": reason, **extra})

        if str(row.get("units")) != schema.CANONICAL_UNITS:
            _vio("units_not_angstrom", value=str(row.get("units")))
        shape, shape_err = _parse_shape(row.get("delta_xyz_shape"))
        if shape is None:
            _vio("bad_delta_shape", value=str(row.get("delta_xyz_shape")),
                 detail=shape_err)
        flat, flat_err = _parse_flat_list(row.get("delta_xyz_flat"))
        if flat is None:
            _vio("bad_delta_flat", detail=flat_err)
        if shape is not None and flat is not None and len(flat) != shape[0] * shape[1]:
            _vio("flat_length_ne_shape",
                 n_flat=len(flat), shape=list(shape),
                 prod_shape=shape[0] * shape[1])
        offsets, off_err = _parse_offsets(row.get("delta_xyz_offsets"))
        if offsets is None:
            _vio("bad_delta_offsets", value=str(row.get("delta_xyz_offsets")),
                 detail=off_err)
        else:
            start, end = offsets
            if start < 0 or end < start:
                _vio("negative_or_inverted_offsets", offsets=list(offsets))
            elif flat is not None and end - start != len(flat):
                _vio("offset_span_ne_flat_length",
                     offsets=list(offsets), n_flat=len(flat))
            if start == 0 and prev_start is not None and prev_start == 0:
                _vio("offset_restart_at_zero", offsets=list(offsets))
            if prev_start is not None and start < prev_start:
                _vio("offsets_not_monotone", offsets=list(offsets),
                     prev_start=prev_start)
            prev_start = start
        if flat is not None:
            declared = str(row.get("delta_xyz_sha256"))
            if not SHA256_RE.match(declared):
                _vio("bad_delta_sha_format", value=declared[:80])
            elif _sha_flat(flat) != declared.lower():
                _vio("delta_sha_mismatch",
                     declared=declared[:16], recomputed=_sha_flat(flat)[:16])
        amap = str(row.get("atom_map_sha256"))
        if not SHA256_RE.match(amap):
            _vio("bad_atom_map_sha_format", value=amap[:80])
    return bad, dict(by_reason)


def _test_t0_1c(tables: _Tables, report_path: Path) -> Dict[str, Any]:
    inputs = tables.input_paths("irc_node", "sample_view", "irc_edge")
    reason = tables.problem("irc_node", "sample_view", "irc_edge")
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
    # PD-9 flat encoding (ORA-2 #1)
    pd9_bad, pd9_by_reason = _check_pd9_edges(tables.frames["irc_edge"])
    n_bad += len(pd9_bad)
    bad_rows.extend(pd9_bad)
    counts = {
        "n_tables_checked": 3,
        "n_non_finite_coordinates": len(bad_rows) - len(pd9_bad),
        "n_edges_pd9_checked": int(len(tables.frames["irc_edge"])),
        "n_pd9_violations": len(pd9_bad),
        "pd9_violations_by_reason": pd9_by_reason,
        "n_bad_rows_reported": len(bad_rows[:MAX_EVIDENCE_ROWS]),
        "per_table_per_column": per_table,
    }
    status = "fail" if n_bad else "pass"
    return _result(
        "T0.1c", status, inputs=inputs, counts=counts,
        details={"delta_sha_encoding": "sha256(struct.pack('<{n}d', *flat))",
                 "units_literal": schema.CANONICAL_UNITS},
        bad_rows=bad_rows[:MAX_EVIDENCE_ROWS],
        note=None if not n_bad else
        f"{n_bad} coordinate/PD-9 violation(s); rows reported, never dropped",
    )


# ---------------------------------------------------------------------------
# T0.2a — intent involution (reverse(reverse(x)) == x) + pairing
# ---------------------------------------------------------------------------

_T0_2A_NEED: Dict[str, Tuple[str, ...]] = {
    "reaction_intent": ("intent_id", "curve_id", "direction",
                        "reverse_of_intent_id", "intent_pair_id",
                        "intent_underdetermined"),
}


def _test_t0_2a(tables: _Tables, report_path: Path) -> Dict[str, Any]:
    inputs = tables.input_paths("reaction_intent")
    reason = tables.problem("reaction_intent")
    if reason is not None:
        return _blocked("T0.2a", reason, inputs=inputs)
    miss = tables.column_problem(_T0_2A_NEED)
    if miss:
        return _schema_fail("T0.2a", _T0_2A_NEED, tables, miss)
    ri = tables.frames["reaction_intent"]
    by_curve: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pos, row in ri.iterrows():
        by_curve[str(row["curve_id"])].append({
            "row_index": int(pos),
            "intent_id": str(row["intent_id"]),
            "direction": str(row["direction"]).strip().lower(),
            "reverse_of": ("" if pd.isna(row["reverse_of_intent_id"])
                           else str(row["reverse_of_intent_id"]).strip()),
            "pair_id": ("" if pd.isna(row["intent_pair_id"])
                        else str(row["intent_pair_id"]).strip()),
            "underdetermined": _parse_bool(row["intent_underdetermined"]),
        })
    violations: List[Dict[str, Any]] = []
    per_curve: Dict[str, Any] = {}
    n_exempt = n_unpaired = 0
    for cid, intents in sorted(by_curve.items()):
        reasons: List[str] = []
        if any(i["underdetermined"] for i in intents):
            n_exempt += 1
            per_curve[cid] = {"exempt_intent_underdetermined": True,
                              "n_intents": len(intents)}
            continue
        if len(intents) != 2:
            n_unpaired += 1
            violations.append({
                "curve_id": cid,
                "reason": f"intent_pair_count:{len(intents)}",
                "intent_ids": [i["intent_id"] for i in intents],
            })
            per_curve[cid] = {"exempt_intent_underdetermined": False,
                              "n_intents": len(intents)}
            continue
        a, b = intents
        if a["pair_id"] != b["pair_id"] or not a["pair_id"]:
            reasons.append("intent_pair_id_mismatch")
        if {a["direction"], b["direction"]} != {"forward", "reverse"}:
            reasons.append("pair_direction_invalid")
        ids = {i["intent_id"] for i in intents}
        for i in intents:
            r = i["reverse_of"]
            if not r:
                reasons.append(f"reverse_of_missing:{i['intent_id']}")
            elif r == i["intent_id"]:
                reasons.append(f"self_reverse:{i['intent_id']}")
            elif r not in ids:
                reasons.append(f"reverse_of_unknown:{i['intent_id']}->{r}")
        if not any(x.startswith(("reverse_of_unknown", "reverse_of_missing",
                                 "self_reverse")) for x in reasons):
            # both targets resolve inside the pair: verify reverse(reverse(x))==x
            fwd = {i["intent_id"]: i for i in intents}
            for i in intents:
                r = fwd.get(i["reverse_of"])
                if r is not None and r["reverse_of"] != i["intent_id"]:
                    reasons.append(f"involution_broken:{i['intent_id']}")
        for r in reasons:
            violations.append({"curve_id": cid, "reason": r,
                               "intent_ids": sorted(ids)})
        per_curve[cid] = {
            "exempt_intent_underdetermined": False,
            "n_intents": 2,
            "intent_ids": sorted(ids),
            "pair_ok": not reasons,
        }
    counts = {
        "n_curves_with_intents": len(by_curve),
        "n_curves_exempt_underdetermined": n_exempt,
        "n_curves_unpaired": n_unpaired,
        "n_violations": len(violations),
        "n_bad_rows_reported": len(violations[:MAX_EVIDENCE_ROWS]),
    }
    note = None
    if violations:
        note = f"{len(violations)} intent involution/pairing violation(s); see bad_rows"
    elif n_exempt:
        note = (f"{n_exempt} curve(s) exempt via intent_underdetermined; "
                "pairing not required for them")
    return _result(
        "T0.2a", "fail" if violations else "pass", inputs=inputs,
        counts=counts, details={"per_curve": per_curve},
        bad_rows=violations[:MAX_EVIDENCE_ROWS], note=note,
    )


# ---------------------------------------------------------------------------
# T0.2b — reversed single-step teacher consistency
# ---------------------------------------------------------------------------

_T0_2B_NEED: Dict[str, Tuple[str, ...]] = {
    "irc_edge": ("curve_id", "intent_id", "edge_index", "src_node_index",
                 "dst_node_index", "delta_xyz_flat", "same_ref_frame"),
    "reaction_intent": ("intent_id", "curve_id", "direction"),
}


def _test_t0_2b(tables: _Tables, report_path: Path) -> Dict[str, Any]:
    inputs = tables.input_paths("irc_edge", "reaction_intent")
    reason = tables.problem("irc_edge", "reaction_intent")
    if reason is not None:
        return _blocked("T0.2b", reason, inputs=inputs)
    miss = tables.column_problem(_T0_2B_NEED)
    if miss:
        return _schema_fail("T0.2b", _T0_2B_NEED, tables, miss)
    ri = tables.frames["reaction_intent"]
    edges = tables.frames["irc_edge"]

    intent_dir: Dict[Tuple[str, str], str] = {}
    intents_by_curve: Dict[str, Dict[str, str]] = defaultdict(dict)
    for _, row in ri.iterrows():
        d = str(row["direction"]).strip().lower()
        intent_dir[str(row["intent_id"])] = d
        intents_by_curve[str(row["curve_id"])][d] = str(row["intent_id"])

    edges_by_intent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pos, row in edges.iterrows():
        flat, flat_err = _parse_flat_list(row.get("delta_xyz_flat"))
        edges_by_intent[str(row["intent_id"])].append({
            "row_index": int(pos),
            "edge_index": str(row.get("edge_index")),
            "src": _scalar_num(row.get("src_node_index")),
            "dst": _scalar_num(row.get("dst_node_index")),
            "flat": flat,
            "flat_err": flat_err,
            "same_ref": _parse_bool(row.get("same_ref_frame")),
        })

    violations: List[Dict[str, Any]] = []
    n_pairs = n_curve_edges = n_skipped = 0
    curves = sorted({str(c) for c in edges["curve_id"].tolist()}) if len(edges) else []
    for cid in curves:
        dirs = intents_by_curve.get(cid, {})
        fwd_id, rev_id = dirs.get("forward"), dirs.get("reverse")
        if fwd_id is None or rev_id is None or fwd_id == rev_id:
            n_skipped += 1
            continue
        n_pairs += 1
        fwd_rows = edges_by_intent.get(fwd_id, [])
        rev_rows = edges_by_intent.get(rev_id, [])
        rev_by_ends: Dict[Tuple[Optional[float], Optional[float]], List[Dict[str, Any]]] = defaultdict(list)
        for r in rev_rows:
            rev_by_ends[(r["src"], r["dst"])].append(r)  # mirror of fwd (dst,src)
        fwd_by_ends: Dict[Tuple[Optional[float], Optional[float]], List[Dict[str, Any]]] = defaultdict(list)
        for f in fwd_rows:
            fwd_by_ends[(f["src"], f["dst"])].append(f)

        def _vio(reason: str, row: Dict[str, Any], **extra: Any) -> None:
            violations.append({
                "curve_id": cid, "reason": reason,
                "fwd_intent_id": fwd_id, "rev_intent_id": rev_id,
                "edge_index": row["edge_index"], "row_index": row["row_index"],
                **extra,
            })

        n_curve_edges += len(fwd_rows)
        for f in fwd_rows:
            if f["src"] is None or f["dst"] is None:
                _vio("bad_endpoint_index", f)
                continue
            cands = rev_by_ends.get((f["dst"], f["src"]), [])
            if not cands:
                _vio("missing_reverse_edge", f)
                continue
            r = cands[0]
            if f["same_ref"] is not True or r["same_ref"] is not True:
                _vio("reference_frame_mismatch", f,
                     rev_edge_index=r["edge_index"],
                     same_ref_frame_fwd=f["same_ref"],
                     same_ref_frame_rev=r["same_ref"])
                continue
            if f["flat"] is None or r["flat"] is None:
                _vio("delta_unparsable", f, rev_edge_index=r["edge_index"])
                continue
            if len(f["flat"]) != len(r["flat"]):
                _vio("reverse_length_mismatch", f, rev_edge_index=r["edge_index"],
                     n_fwd=len(f["flat"]), n_rev=len(r["flat"]))
                continue
            worst = max((abs(a + b) for a, b in zip(f["flat"], r["flat"])),
                        default=0.0)
            if worst > schema.DELTA_EPS:
                _vio("delta_not_negated", f, rev_edge_index=r["edge_index"],
                     max_abs_sum=worst, eps=schema.DELTA_EPS)
        # reverse rows must all be mirrored by a forward row (no orphan reverse)
        for r in rev_rows:
            if r["src"] is None or r["dst"] is None:
                _vio("bad_endpoint_index", r)
                continue
            if (r["dst"], r["src"]) not in fwd_by_ends:
                violations.append({
                    "curve_id": cid, "reason": "missing_forward_edge",
                    "fwd_intent_id": fwd_id, "rev_intent_id": rev_id,
                    "edge_index": r["edge_index"], "row_index": r["row_index"],
                })
    counts = {
        "n_curves_with_edges": len(curves),
        "n_intent_pairs_checked": n_pairs,
        "n_curves_skipped_unpaired": n_skipped,
        "n_forward_edges_checked": n_curve_edges,
        "n_violations": len(violations),
        "n_bad_rows_reported": len(violations[:MAX_EVIDENCE_ROWS]),
        "negation_eps": schema.DELTA_EPS,
    }
    note = None
    if violations:
        note = (f"{len(violations)} reversed-teacher violation(s) "
                "(mirror edge missing / delta not negated / frame flag); "
                "see bad_rows")
    elif n_skipped:
        note = (f"{n_skipped} curve(s) skipped: no unique forward+reverse "
                "intent pair (pairing completeness is owned by T0.2a)")
    return _result(
        "T0.2b", "fail" if violations else "pass", inputs=inputs,
        counts=counts, bad_rows=violations[:MAX_EVIDENCE_ROWS], note=note,
    )


# ---------------------------------------------------------------------------
# T0.3a — d0 gate
# ---------------------------------------------------------------------------

_T0_3A_NEED: Dict[str, Tuple[str, ...]] = {
    "sample_view": ("sample_id", "d0", "sensitivity_only"),
}


def _test_t0_3a(tables: _Tables, report_path: Path) -> Dict[str, Any]:
    inputs = tables.input_paths("sample_view")
    reason = tables.problem("sample_view")
    if reason is not None:
        return _blocked("T0.3a", reason, inputs=inputs)
    miss = tables.column_problem(_T0_3A_NEED)
    if miss:
        return _schema_fail("T0.3a", _T0_3A_NEED, tables, miss)
    sv = tables.frames["sample_view"]
    violations: List[Dict[str, Any]] = []
    n_main = n_sens = n_bad_d0 = 0
    for pos, row in sv.iterrows():
        sid = str(row["sample_id"])
        d0 = _scalar_num(row["d0"])
        flag = _parse_bool(row["sensitivity_only"])
        if d0 is None:
            n_bad_d0 += 1
            violations.append({"sample_id": sid, "row_index": int(pos),
                               "d0": str(row["d0"]),
                               "reason": "non_finite_d0"})
            continue
        if flag is None:
            violations.append({"sample_id": sid, "row_index": int(pos),
                               "sensitivity_only": str(row["sensitivity_only"]),
                               "reason": "bad_sensitivity_flag"})
            continue
        is_primary = abs(d0 - schema.PRIMARY_D0_ANGSTROM) <= schema.D0_TOL
        if is_primary:
            if not flag:
                n_main += 1
        else:
            if flag:
                n_sens += 1
            else:
                violations.append({
                    "sample_id": sid, "row_index": int(pos), "d0": d0,
                    "reason": "unflagged_nonprimary_d0",
                })
    if n_main == 0:
        violations.append({
            "reason": "no_main_task_rows",
            "primary_d0": schema.PRIMARY_D0_ANGSTROM,
            "n_rows": int(len(sv)),
        })
    counts = {
        "n_rows": int(len(sv)),
        "n_main_d0_rows": n_main,
        "n_sensitivity_rows": n_sens,
        "n_nonfinite_d0": n_bad_d0,
        "primary_d0": schema.PRIMARY_D0_ANGSTROM,
        "n_violations": len(violations),
        "n_bad_rows_reported": len(violations[:MAX_EVIDENCE_ROWS]),
    }
    note = None
    if violations:
        note = (f"{len(violations)} d0 violation(s): main rows must use "
                f"d0={schema.PRIMARY_D0_ANGSTROM}, other d0 needs "
                "sensitivity_only=true; see bad_rows")
    return _result(
        "T0.3a", "fail" if violations else "pass", inputs=inputs,
        counts=counts, bad_rows=violations[:MAX_EVIDENCE_ROWS], note=note,
    )


# ---------------------------------------------------------------------------
# T0.3b — gauge gate
# ---------------------------------------------------------------------------

_T0_3B_NEED: Dict[str, Tuple[str, ...]] = {
    "sample_view": ("sample_id", "gauge_ref", "label_status"),
}


def _test_t0_3b(tables: _Tables, report_path: Path) -> Dict[str, Any]:
    inputs = tables.input_paths("sample_view")
    reason = tables.problem("sample_view")
    if reason is not None:
        return _blocked("T0.3b", reason, inputs=inputs)
    miss = tables.column_problem(_T0_3B_NEED)
    if miss:
        return _schema_fail("T0.3b", _T0_3B_NEED, tables, miss)
    sv = tables.frames["sample_view"]
    violations: List[Dict[str, Any]] = []
    n_unresolved = 0
    for pos, row in sv.iterrows():
        sid = str(row["sample_id"])
        gauge = "" if pd.isna(row["gauge_ref"]) else str(row["gauge_ref"]).strip()
        status = "" if pd.isna(row["label_status"]) else str(row["label_status"]).strip()
        if gauge not in schema.GAUGE_REF_VALUES:
            violations.append({"sample_id": sid, "row_index": int(pos),
                               "gauge_ref": gauge, "reason": "bad_gauge_ref"})
        if status not in schema.LABEL_STATUS_VALUES:
            violations.append({"sample_id": sid, "row_index": int(pos),
                               "label_status": status, "reason": "bad_label_status"})
        if gauge == "unresolved_masked":
            n_unresolved += 1
            if status != "masked":
                violations.append({
                    "sample_id": sid, "row_index": int(pos),
                    "gauge_ref": gauge, "label_status": status,
                    "reason": "unresolved_gauge_not_masked",
                })
    counts = {
        "n_rows": int(len(sv)),
        "n_unresolved_gauge_rows": n_unresolved,
        "gauge_ref_values": list(schema.GAUGE_REF_VALUES),
        "label_status_values": list(schema.LABEL_STATUS_VALUES),
        "n_violations": len(violations),
        "n_bad_rows_reported": len(violations[:MAX_EVIDENCE_ROWS]),
    }
    note = None
    if violations:
        note = (f"{len(violations)} gauge/label violation(s): unresolved gauge "
                "must be masked (never a silent lab frame); see bad_rows")
    return _result(
        "T0.3b", "fail" if violations else "pass", inputs=inputs,
        counts=counts, bad_rows=violations[:MAX_EVIDENCE_ROWS], note=note,
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
# T0.5 — truth-access audit (ORA-2 #6 / constitution R1)
# ---------------------------------------------------------------------------

def _resolve_truth_log(
    tables: _Tables, truth_log: Optional[Path]
) -> Tuple[Optional[Path], bool, Optional[str]]:
    """Returns (path, strict, note). path=None → nothing to audit."""
    if truth_log is not None:
        return Path(truth_log), True, None
    in_tables = tables.dir / schema.TRUTH_LOG_TABLES_DIR_NAME
    if in_tables.is_file():
        return in_tables, False, f"resolved via <tables-dir>/{schema.TRUTH_LOG_TABLES_DIR_NAME}"
    fallback = Path(schema.TRUTH_LOG_DEFAULT_SUBPATH)
    if fallback.is_file():
        return fallback, False, f"resolved via {schema.TRUTH_LOG_DEFAULT_SUBPATH}"
    return None, False, None


def _test_t0_5(
    tables: _Tables, report_path: Path, truth_log: Optional[Path] = None
) -> Dict[str, Any]:
    path, strict, resolve_note = _resolve_truth_log(tables, truth_log)
    if path is None:
        if strict:
            return _blocked(
                "T0.5",
                f"requested truth log missing: {truth_log} (fail-closed; "
                f"resolved via {TRUTH_LOG_CANDIDATES_NOTE})",
                inputs=[Path(truth_log)],  # type: ignore[arg-type]
            )
        return _not_run(
            "T0.5",
            "no truth access log to audit; pass --truth-log PATH to enforce "
            f"(searched {TRUTH_LOG_CANDIDATES_NOTE})",
        )
    if not path.is_file():
        # strict requested path that vanished between resolve and read
        return _blocked("T0.5", f"truth log missing: {path}", inputs=[path])
    entries = 0
    bad_rows: List[Dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        entries += 1
        try:
            obj = json.loads(line)
        except Exception as exc:
            bad_rows.append({"log": str(path), "line": lineno,
                             "reason": f"unparseable_entry: {exc}"})
            continue
        if not isinstance(obj, dict):
            bad_rows.append({"log": str(path), "line": lineno,
                             "reason": "entry_not_an_object"})
            continue
        accessor = obj.get("accessor", obj.get("accessor_module"))
        if not isinstance(accessor, str) or not accessor.strip():
            bad_rows.append({"log": str(path), "line": lineno,
                             "reason": "missing_accessor_field",
                             "keys": sorted(obj)[:10]})
            continue
        if accessor.strip() not in schema.TRUTH_ACCESSOR_ALLOWLIST:
            bad_rows.append({"log": str(path), "line": lineno,
                             "reason": "accessor_not_allowlisted",
                             "accessor": accessor,
                             "allowlist": list(schema.TRUTH_ACCESSOR_ALLOWLIST)})
    counts = {
        "n_entries": entries,
        "n_violations": len(bad_rows),
        "n_bad_rows_reported": len(bad_rows[:MAX_EVIDENCE_ROWS]),
        "allowlist": list(schema.TRUTH_ACCESSOR_ALLOWLIST),
    }
    details = {"log": str(path), "strict": strict}
    if resolve_note:
        details["resolution"] = resolve_note
    note = None
    if entries == 0:
        note = "empty truth access log: no truth reads recorded"
    if bad_rows:
        note = (f"{len(bad_rows)} truth-access violation(s): only allow-listed "
                "truth_reader accessors may read truth; see bad_rows")
    return _result(
        "T0.5", "fail" if bad_rows else "pass", inputs=[path],
        counts=counts, details=details, bad_rows=bad_rows[:MAX_EVIDENCE_ROWS],
        note=note,
    )


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

_RUNNERS = {
    "T0.1a": _test_t0_1a,
    "T0.1b": _test_t0_1b,
    "T0.1c": _test_t0_1c,
    "T0.2a": _test_t0_2a,
    "T0.2b": _test_t0_2b,
    "T0.3a": _test_t0_3a,
    "T0.3b": _test_t0_3b,
    "T0.4a": _test_t0_4a,
    "T0.4b": _test_t0_4b,
}


def run_validation(
    tables_dir: Path,
    report_path: Path,
    only_tests: Optional[Sequence[str]] = None,
    truth_log: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all (or the selected) gate tests, write the JSON report, return it."""
    tables_dir = Path(tables_dir)
    report_path = Path(report_path)
    tables = _Tables(tables_dir)
    only = set(only_tests) if only_tests else None

    tests: List[Dict[str, Any]] = []
    for i, tid in enumerate(TEST_IDS):
        if only is not None and tid not in only:
            res = _not_run(tid, "excluded by --tests filter")
        else:
            try:
                if tid == "T0.5":
                    res = _test_t0_5(tables, report_path, truth_log=truth_log)
                else:
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
        "truth_log": str(truth_log) if truth_log is not None else None,
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
            "(T0.1a/b/c FK+TS+PD-9, T0.2a/b intent involution + reversed "
            "teacher, T0.3a/b d0+gauge, T0.4a/b split+manifests, "
            "T0.5 truth-access audit)."
        ),
    )
    parser.add_argument("--tables-dir", required=True,
                        help="directory holding the six CSV tables + 3 JSON manifests")
    parser.add_argument("--report", required=True,
                        help="output JSON report path")
    parser.add_argument("--tests", nargs="+", choices=list(TEST_IDS), default=None,
                        help="run only these test ids (others are reported not_run)")
    parser.add_argument("--truth-log", default=None,
                        help="truth_access_log.jsonl to audit (T0.5); a missing "
                             "file is BLOCKLED fail-closed. Without this option "
                             "the log is searched at <tables-dir>/"
                             f"{schema.TRUTH_LOG_TABLES_DIR_NAME} then "
                             f"{schema.TRUTH_LOG_DEFAULT_SUBPATH}.")
    args = parser.parse_args(argv)

    report = run_validation(args.tables_dir, args.report, args.tests,
                            truth_log=args.truth_log)
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
