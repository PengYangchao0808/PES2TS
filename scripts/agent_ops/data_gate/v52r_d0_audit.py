#!/usr/bin/env python3
"""V5.2-R stage D0 audit: endpoint-stop labels + provenance (gate V5.2R-D0-LABELS).

Read-only over the PD-1 six-table delivery (``outputs/pesn/30h_campaign/data``);
writes ONLY the D0 evidence pack (``evidence_pack.json`` + ``D0_AUDIT.md``)
under ``--out`` (default ``outputs/pesn/v52r/d0``).  No training, no model
loads, no sealed-fold metric use — folds 3/4 are COUNTED for leakage
disclosure only, never used for any selection (V5.2-R plan §3 D0; campaign
``pes2ts_v52r_20261021.yaml`` ``split_contract``).

Implements the D0 evidence list of ``docs/plans/PES2TS_V52R_全路径导航与
端点停止修复方案_v1_20261021.md`` §3 D0 and contract
``docs/contracts/PES2TS_FullPathNavigationContract_v1.md`` §5/§7/§11,
against the acceptance numbers of the V5.2 task-correction audit
(``docs/reports/PES2TS_V52_任务纠偏审计_20261021.md`` §1/§6):

1. label-table totals + window sensitivity (0.08/0.10/0.12/0.15 × 0.30);
2. per-curve positive min/median/max (expect 9/32/90 at 0.10/0.30);
3. 200 endpoint-row role audit (92 arriving terminal→1, 90 departing ok→0);
4. start-FP (expect 0) and TS-FP (expect exactly the 2 flagged degenerates;
   reported with/without-degenerate calibers);
5. terminal-inclusion proof (1123 no_future_contact rows in the Stop table —
   the core V5.2 bug fix);
6. intent-contrast table (arriving→1 / departing→0 at both endpoints);
7. leakage check: one fold per curve, dev folds 0–2 only (30 assigned / 28
   with ok rows), folds 3/4 sealed & untouched;
8. topology-consistency proxy over positive rows (per-cv-pair distance
   deviation from the target-endpoint value; >0.25 Å rows listed);
9. provenance (git HEAD, manifest sha256s, label schema + spec sha256,
   config sha256) and the four V5.2R-D0-LABELS gate checks.

Outputs are deterministic for fixed inputs (sorted keys, no timestamps).

Usage::

    python scripts/agent_ops/data_gate/v52r_d0_audit.py \\
        [--tables-dir outputs/pesn/30h_campaign/data] \\
        [--out outputs/pesn/v52r/d0] \\
        [--strict]          # exit 1 when the gate verdict is FAIL
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    DATA_MANIFEST_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    KeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (  # noqa: E402
    DEGENERATE_TS_CURVES,
    ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM,
    ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM,
    SCHEMA,
    EndpointStopLabelTable,
    build_endpoint_stop_labels,
)

SCHEMA_ID = "pes2ts_v52r_d0_evidence_v1"
DEFAULT_TABLES_DIR = REPO_ROOT / "outputs/pesn/30h_campaign/data"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/pesn/v52r/d0"
CAMPAIGN_CONFIG = REPO_ROOT / "config/campaigns/pes2ts_v52r_20261021.yaml"
ENDPOINT_MODULE = (REPO_ROOT /
                   "pes2ts_core/pesn/offline/data/key_atom_endpoint_labels_v1.py")
GATE_ID = "V5.2R-D0-LABELS"

#: Sensitivity grid of the candidate positive window (w_neg frozen 0.30).
WINDOW_GRID = (0.08, 0.10, 0.12, 0.15)
W_NEG = ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM

#: Acceptance targets from the V5.2 task-correction audit 20261021.
EXPECT_TOTAL_ROWS = 12034
EXPECT_POS = 1781
EXPECT_NEG = 8399
EXPECT_MASK = 1854
EXPECT_CURVE_POS = (9, 32.0, 90)          # min / median / max @ 0.10/0.30
EXPECT_START_FP = 0
EXPECT_TS_FP_CURVES = tuple(sorted(DEGENERATE_TS_CURVES))
EXPECT_TERMINAL_ROWS = 1123
EXPECT_ENDPOINT_ROWS = 200
EXPECT_ARRIVING_TERMINAL_LAB1 = 92
EXPECT_DEPARTING_OK_LAB0 = 90
EXPECT_DEV_CURVES = 30
EXPECT_DEV_CURVES_WITH_OK = 28
TOPOLOGY_DEV_TOL_ANGSTROM = 0.25


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _round(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


# --------------------------------------------------------------------- audits


def build_base_tables(
    dataset: KeyAtomDatasetV3,
) -> tuple[EndpointStopLabelTable, dict[float, EndpointStopLabelTable]]:
    """Primary-window table + per-w_pos sensitivity tables (w_neg frozen)."""
    v52 = dataset.v52_labels()
    base = build_endpoint_stop_labels(dataset, v52)
    grid = {
        w: build_endpoint_stop_labels(
            dataset, v52, positive_window=w, negative_window=W_NEG)
        for w in WINDOW_GRID}
    return base, grid


def audit_label_stats(
    base: EndpointStopLabelTable,
    grid: dict[float, EndpointStopLabelTable],
) -> dict[str, Any]:
    per_curve_pos = sorted(base.positive_counts_by_curve().values())
    windows: dict[str, Any] = {}
    for w in WINDOW_GRID:
        t = grid[w]
        cpos = sorted(t.positive_counts_by_curve().values())
        windows[f"{w:.2f}/{W_NEG:.2f}"] = {
            "positive": t.n_positive,
            "negative": t.n_negative,
            "mask": t.n_mask,
            "prevalence": _round(t.n_positive / max(1, t.n_rows)),
            "per_curve_positive_min": cpos[0],
            "per_curve_positive_median": _round(statistics.median(cpos)),
            "per_curve_positive_max": cpos[-1],
        }
    return {
        "schema": SCHEMA,
        "n_rows": base.n_rows,
        "n_curves": len(base.curves),
        "n_positive": base.n_positive,
        "n_negative": base.n_negative,
        "n_mask": base.n_mask,
        "move_status_counts": {
            s: base.count_move_status(s)
            for s in ("ok", "no_future_contact", "masked", "conflict")},
        "per_curve_positive_min": per_curve_pos[0],
        "per_curve_positive_median": _round(
            statistics.median(per_curve_pos)),
        "per_curve_positive_max": per_curve_pos[-1],
        "windows": windows,
    }


def audit_endpoint_roles(base: EndpointStopLabelTable) -> dict[str, Any]:
    """Role audit of the 200 endpoint rows (50 curves × 2 ends × 2 intents)."""
    breakdown: Counter = Counter()
    for lab in base.by_sample.values():
        c = base.curves[lab.curve_id]
        if lab.node_index not in (c.endpoint_a_node, c.endpoint_b_node):
            continue
        arriving = ((lab.direction == "forward"
                     and lab.node_index == c.endpoint_b_node)
                    or (lab.direction == "reverse"
                        and lab.node_index == c.endpoint_a_node))
        role = "arriving_target" if arriving else "departing_start"
        breakdown[(role, lab.move_status, lab.endpoint_stop_label)] += 1
    rows = [
        {"role": role, "move_status": status, "endpoint_stop_label": label,
         "count": n}
        for (role, status, label), n in sorted(
            breakdown.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))]
    total = sum(r["count"] for r in rows)
    arriving_terminal_1 = sum(
        r["count"] for r in rows if r["role"] == "arriving_target"
        and r["move_status"] == "no_future_contact"
        and r["endpoint_stop_label"] == 1)
    departing_ok_0 = sum(
        r["count"] for r in rows if r["role"] == "departing_start"
        and r["move_status"] == "ok" and r["endpoint_stop_label"] == 0)
    return {
        "n_endpoint_rows": total,
        "breakdown": rows,
        "arriving_terminal_labeled_1": arriving_terminal_1,
        "departing_ok_labeled_0": departing_ok_0,
    }


def audit_false_positives(
    base: EndpointStopLabelTable,
    dataset: KeyAtomDatasetV3,
) -> dict[str, Any]:
    """Start-endpoint FP + TS FP (with/without-degenerate calibers)."""
    start_rows: list = []
    ts_rows: list = []
    for lab in base.by_sample.values():
        c = base.curves[lab.curve_id]
        if lab.node_index == c.ts_index:
            ts_rows.append(lab)
        is_start = ((lab.direction == "forward"
                     and lab.node_index == c.endpoint_a_node)
                    or (lab.direction == "reverse"
                        and lab.node_index == c.endpoint_b_node))
        if is_start:
            start_rows.append(lab)
    start_fp = sorted(
        (l.curve_id, l.direction, _round(l.d_end_angstrom))
        for l in start_rows if l.endpoint_stop_label == 1)
    start_d_ends = sorted(l.d_end_angstrom for l in start_rows)
    ts_fp_rows = sorted(
        (l.curve_id, l.direction, _round(l.d_end_angstrom),
         bool(l.degenerate_flag))
        for l in ts_rows if l.endpoint_stop_label == 1)
    ts_fp_curves = sorted({r[0] for r in ts_fp_rows})
    ts_fp_unflagged = [r for r in ts_fp_rows if not r[3]]
    ts_fp_without_degenerate = [
        r for r in ts_fp_rows
        if not (r[0] in DEGENERATE_TS_CURVES)]
    ts_d_ends = sorted(l.d_end_angstrom for l in ts_rows)
    return {
        "start": {
            "n_rows": len(start_rows),
            "false_positive_count": len(start_fp),
            "false_positives": start_fp,
            "d_end_min": _round(start_d_ends[0]) if start_d_ends else None,
            "d_end_median": _round(
                statistics.median(start_d_ends)) if start_d_ends else None,
            "d_end_max": _round(start_d_ends[-1]) if start_d_ends else None,
        },
        "ts": {
            "n_rows": len(ts_rows),
            "false_positive_count": len(ts_fp_rows),
            "false_positives": [
                {"curve_id": c, "direction": d, "d_end_angstrom": v,
                 "degenerate_flag": f}
                for c, d, v, f in ts_fp_rows],
            "false_positive_curves": ts_fp_curves,
            "expected_degenerate_curves": list(EXPECT_TS_FP_CURVES),
            "false_positives_without_degenerate": len(
                ts_fp_without_degenerate),
            "unflagged_false_positives": ts_fp_unflagged,
            "degenerate_flagged_rows": sum(
                1 for l in base.by_sample.values() if l.degenerate_flag),
            "d_end_median": _round(statistics.median(ts_d_ends))
            if ts_d_ends else None,
        },
    }


def audit_terminal_inclusion(base: EndpointStopLabelTable) -> dict[str, Any]:
    terminal = [
        l for l in base.by_sample.values()
        if l.move_status == "no_future_contact"]
    return {
        "n_terminal_rows_in_stop_table": len(terminal),
        "n_terminal_positive": sum(
            1 for l in terminal if l.endpoint_stop_label == 1),
        "n_terminal_negative": sum(
            1 for l in terminal if l.endpoint_stop_label == 0),
        "n_terminal_mask": sum(
            1 for l in terminal if l.endpoint_stop_label == -1),
        "n_conflict_rows_in_stop_table": sum(
            1 for l in base.by_sample.values() if l.move_status == "conflict"),
        "n_masked_rows_in_stop_table": base.count_move_status("masked"),
        "n_all_rows": base.n_rows,
    }


def audit_intent_contrast(base: EndpointStopLabelTable) -> dict[str, Any]:
    """Arriving intent → label 1; departing intent → label 0; per endpoint."""
    by_key: dict[tuple[str, int, str], Any] = {}
    for lab in base.by_sample.values():
        by_key[(lab.curve_id, lab.node_index, lab.direction)] = lab
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for cid in sorted(base.curves):
        c = base.curves[cid]
        for node, name in ((c.endpoint_a_node, "A"), (c.endpoint_b_node, "B")):
            fwd = by_key.get((cid, node, "forward"))
            rev = by_key.get((cid, node, "reverse"))
            fwd_lab = fwd.endpoint_stop_label if fwd else None
            rev_lab = rev.endpoint_stop_label if rev else None
            # at the target endpoint B: forward arrives (1), reverse departs (0)
            expect_fwd = 1 if name == "B" else 0
            expect_rev = 0 if name == "B" else 1
            ok = fwd_lab == expect_fwd and rev_lab == expect_rev
            rows.append({
                "curve_id": cid, "endpoint": name, "node_index": node,
                "forward_label": fwd_lab, "reverse_label": rev_lab,
                "expect_forward": expect_fwd, "expect_reverse": expect_rev,
                "contrast_ok": ok})
            if not ok:
                violations.append(rows[-1])
    return {
        "n_curve_endpoints": len(rows),
        "n_violations": len(violations),
        "violations": violations,
        "rows": rows,
    }


def audit_leakage(
    tables_dir: Path,
    dataset: KeyAtomDatasetV3,
) -> dict[str, Any]:
    """One fold per curve; dev folds 0–2 disclosure; sealed folds counted."""
    split_path = tables_dir / SPLIT_MANIFEST_FILENAME
    doc = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = doc.get("assignments", [])
    curve_folds: dict[str, list[int]] = {}
    for a in assignments:
        curve_folds.setdefault(str(a["curve_id"]), []).append(int(a["fold"]))
    multi = {c: fs for c, fs in curve_folds.items() if len(set(fs)) != 1}
    dup = {c: fs for c, fs in curve_folds.items() if len(fs) > 1}
    folds_of = dataset.folds
    dev_curves = sorted(c for c, f in folds_of.items() if f in (0, 1, 2))
    sealed_curves = sorted(c for c, f in folds_of.items() if f in (3, 4))
    ok_curves = {
        rec.curve_id for rec in dataset.samples if rec.label_status == "ok"}
    dev_with_ok = sorted(c for c in dev_curves if c in ok_curves)
    per_fold = Counter(folds_of.values())
    return {
        "n_curves_in_manifest": len(curve_folds),
        "curves_with_multiple_folds": multi,
        "duplicate_assignments": dup,
        "single_fold_per_curve": not multi and not dup,
        "per_fold_curve_count": {str(f): per_fold.get(f, 0)
                                 for f in sorted(per_fold)},
        "dev_fold_curve_count": len(dev_curves),
        "dev_fold_curves_with_ok_rows": len(dev_with_ok),
        "dev_fold_curves": dev_curves,
        "sealed_fold_curve_count": len(sealed_curves),
        "sealed_fold_curves": sealed_curves,
        "sealed_folds_untouched": True,   # counted only; never a selection input
    }


def audit_topology_proxy(
    dataset: KeyAtomDatasetV3,
    base: EndpointStopLabelTable,
) -> dict[str, Any]:
    """Bond-edit-completion proxy over positive rows (contract §5.3 rule 5).

    For each positive row (``d_end <= w_pos``) and each of the intent's
    ``cv_spec.coordinates`` pairs ``(i, j)``, the deviation
    ``|‖x_i−x_j‖(X_row) − ‖x_i−x_j‖(X_target_endpoint)|`` is computed in the
    native frame (no alignment — same verified reference frame per PD-9).
    Rows with ANY pair deviating more than 0.25 Å from the endpoint value
    are listed as geometrically-near-but-topologically-incomplete.
    """
    mean_devs: list[float] = []
    max_devs: list[float] = []
    outliers: list[dict[str, Any]] = []
    n_no_cv = 0
    for rec in dataset.samples:
        lab = base.by_sample.get(rec.sample_id)
        if lab is None or lab.endpoint_stop_label != 1:
            continue
        pairs = rec.intent.cv_coordinates
        if not pairs:
            n_no_cv += 1
            continue
        frame = np.asarray(
            dataset.frame_xyz(rec.curve_id, rec.node_index),
            dtype=np.float64)
        end = np.asarray(
            dataset.frame_xyz(rec.curve_id, lab.target_endpoint_node),
            dtype=np.float64)
        devs = []
        for i, j in pairs:
            d_row = float(np.linalg.norm(frame[i] - frame[j]))
            d_endp = float(np.linalg.norm(end[i] - end[j]))
            devs.append(abs(d_row - d_endp))
        mean_devs.append(float(np.mean(devs)))
        max_devs.append(float(np.max(devs)))
        if max(devs) > TOPOLOGY_DEV_TOL_ANGSTROM:
            outliers.append({
                "sample_id": rec.sample_id,
                "curve_id": rec.curve_id,
                "node_index": int(rec.node_index),
                "direction": rec.intent.direction,
                "n_pairs": len(pairs),
                "mean_dev_angstrom": _round(np.mean(devs)),
                "max_dev_angstrom": _round(np.max(devs)),
                "pair_devs_angstrom": [_round(d) for d in devs],
            })
    mean_devs.sort()
    max_devs.sort()

    def _q(vals: list[float], frac: float) -> float | None:
        if not vals:
            return None
        idx = min(len(vals) - 1, int(round(frac * (len(vals) - 1))))
        return _round(vals[idx])

    return {
        "n_positive_rows": base.n_positive,
        "n_rows_scored": len(mean_devs),
        "n_rows_without_cv_pairs": n_no_cv,
        "tolerance_angstrom": TOPOLOGY_DEV_TOL_ANGSTROM,
        "per_row_mean_dev": {
            "min": _q(mean_devs, 0.0),
            "p25": _q(mean_devs, 0.25),
            "median": _q(mean_devs, 0.5),
            "p75": _q(mean_devs, 0.75),
            "max": _q(mean_devs, 1.0),
        },
        "per_row_max_dev_median": _q(max_devs, 0.5),
        "n_outlier_rows": len(outliers),
        "outliers": outliers,
    }


def collect_provenance(tables_dir: Path) -> dict[str, Any]:
    data_manifest = tables_dir / DATA_MANIFEST_FILENAME
    teacher_manifest = tables_dir / "teacher_manifest.json"
    split_manifest = tables_dir / SPLIT_MANIFEST_FILENAME
    dataset_sha = ""
    if data_manifest.is_file():
        doc = json.loads(data_manifest.read_text(encoding="utf-8"))
        dataset_sha = str(doc.get("dataset_sha256", ""))
    prov = {
        "code_commit": git_head(),
        "data_manifest_sha256": sha256_file(data_manifest)
        if data_manifest.is_file() else "",
        "teacher_manifest_sha256": sha256_file(teacher_manifest)
        if teacher_manifest.is_file() else "",
        "split_sha256": sha256_file(split_manifest)
        if split_manifest.is_file() else "",
        "label_schema": SCHEMA,
        "label_spec_sha256": sha256_file(ENDPOINT_MODULE)
        if ENDPOINT_MODULE.is_file() else "",
        "config_sha256": sha256_file(CAMPAIGN_CONFIG)
        if CAMPAIGN_CONFIG.is_file() else "",
        "dataset_sha256": dataset_sha,
        "tables_dir": str(tables_dir),
    }
    required = ("code_commit", "data_manifest_sha256",
                "teacher_manifest_sha256", "split_sha256", "label_schema",
                "label_spec_sha256", "config_sha256")
    prov["required_fields_present"] = all(prov.get(k, "") for k in required)
    prov["missing_required_fields"] = [
        k for k in required if not prov.get(k)]
    return prov


def evaluate_gate(
    label_stats: dict[str, Any],
    fp: dict[str, Any],
    terminal: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Gate ``V5.2R-D0-LABELS`` (campaign yaml): four checks + overall."""
    checks = [
        {
            "id": "endpoint_positives_in_stop_table",
            "expect": "stop table contains endpoint positives "
                      "(incl. terminal rows)",
            "actual": (f"{label_stats['n_positive']} positives; "
                       f"{terminal['n_terminal_positive']} of them "
                       "no_future_contact"),
            "pass": (label_stats["n_positive"] > 0
                     and terminal["n_terminal_positive"] > 0
                     and terminal["n_terminal_rows_in_stop_table"]
                     == EXPECT_TERMINAL_ROWS),
        },
        {
            "id": "start_false_positives_zero",
            "expect": f"start FP == {EXPECT_START_FP}",
            "actual": str(fp["start"]["false_positive_count"]),
            "pass": fp["start"]["false_positive_count"] == EXPECT_START_FP,
        },
        {
            "id": "ts_fp_only_flagged_degenerates",
            "expect": ("TS FP only on flagged degenerate curves "
                       f"{list(EXPECT_TS_FP_CURVES)}; unflagged FP == 0"),
            "actual": (f"curves={fp['ts']['false_positive_curves']}, "
                       f"unflagged={len(fp['ts']['unflagged_false_positives'])}"),
            "pass": (not fp["ts"]["unflagged_false_positives"]
                     and set(fp["ts"]["false_positive_curves"])
                     <= set(EXPECT_TS_FP_CURVES)),
        },
        {
            "id": "provenance_fields_present",
            "expect": "7 required run-record provenance fields non-empty",
            "actual": ("all present" if provenance["required_fields_present"]
                       else f"missing {provenance['missing_required_fields']}"),
            "pass": bool(provenance["required_fields_present"]),
        },
    ]
    overall = all(c["pass"] for c in checks)
    return {"gate_id": GATE_ID, "checks": checks,
            "verdict": "PASS" if overall else "FAIL"}


# ------------------------------------------------------------------ reporting


def render_markdown(pack: dict[str, Any]) -> str:
    stats = pack["label_table_stats"]
    roles = pack["endpoint_role_audit"]
    fp = pack["false_positive_audit"]
    term = pack["terminal_inclusion"]
    contrast = pack["intent_contrast"]
    leak = pack["leakage_checks"]
    topo = pack["topology_consistency_proxy"]
    prov = pack["provenance"]
    gate = pack["gate"]

    def mark(ok: bool) -> str:
        return "PASS" if ok else "**FAIL**"

    base_w = stats["windows"]["0.10/0.30"]
    lines: list[str] = []
    a = lines.append
    a("# V5.2-R D0 审计报告 — Endpoint Stop 标签与 provenance")
    a("")
    a(f"> 生成: `scripts/agent_ops/data_gate/v52r_d0_audit.py` · "
      f"schema `{stats['schema']}` · gate `{gate['gate_id']}`")
    a(f"> 数据: `{prov['tables_dir']}` · commit `{prov['code_commit'][:12]}`"
      f" · dataset_sha256 `{prov['dataset_sha256'][:12]}…`")
    a(f"> 主候选窗口: w_pos=0.10 Å / w_neg={W_NEG:.2f} Å（D1 敏感性后冻结）")
    a("")
    a("## 1. 标签表统计")
    a("")
    a(f"- 总行数: **{stats['n_rows']}** (期望 {EXPECT_TOTAL_ROWS}) — "
      f"{mark(stats['n_rows'] == EXPECT_TOTAL_ROWS)}；曲线数 "
      f"{stats['n_curves']}")
    a(f"- move_status: {stats['move_status_counts']}")
    a("")
    a("| 窗口 (pos/neg) | 正例 | 负例 | Mask | 每曲线正例 min/med/max |")
    a("|---|---:|---:|---:|---|")
    for key, w in stats["windows"].items():
        star = " ← 主候选" if key.startswith("0.10") else ""
        a(f"| {key}{star} | {w['positive']} | {w['negative']} | "
          f"{w['mask']} | {w['per_curve_positive_min']} / "
          f"{w['per_curve_positive_median']} / "
          f"{w['per_curve_positive_max']} |")
    ok_main = (base_w["positive"] == EXPECT_POS
               and base_w["negative"] == EXPECT_NEG
               and base_w["mask"] == EXPECT_MASK)
    a("")
    a(f"- 主窗口 0.10/{W_NEG:.2f}: 正例 {base_w['positive']} (期望 "
      f"{EXPECT_POS}) / 负例 {base_w['negative']} (期望 {EXPECT_NEG}) / "
      f"Mask {base_w['mask']} (期望 {EXPECT_MASK}) — {mark(ok_main)}")
    ok_pc = (stats["per_curve_positive_min"],
             stats["per_curve_positive_median"],
             stats["per_curve_positive_max"]) == EXPECT_CURVE_POS
    a(f"- 每曲线正例 min/med/max = {stats['per_curve_positive_min']} / "
      f"{stats['per_curve_positive_median']} / "
      f"{stats['per_curve_positive_max']} (期望 9 / 32 / 90) — {mark(ok_pc)}")
    a("")
    a("## 2. 端点行角色审计（200 行 = 50 曲线 × 2 端点 × 2 intent）")
    a("")
    a("| 角色 | move_status | endpoint_stop_label | 行数 |")
    a("|---|---|---:|---:|")
    for r in roles["breakdown"]:
        a(f"| {r['role']} | {r['move_status']} | "
          f"{r['endpoint_stop_label']} | {r['count']} |")
    a("")
    a(f"- 到达目标端点的 terminal 行 → label 1: "
      f"**{roles['arriving_terminal_labeled_1']}** (期望 "
      f"{EXPECT_ARRIVING_TERMINAL_LAB1}) — "
      f"{mark(roles['arriving_terminal_labeled_1'] == EXPECT_ARRIVING_TERMINAL_LAB1)}")
    a(f"- 从端点出发的 ok 行 → label 0: "
      f"**{roles['departing_ok_labeled_0']}** (期望 "
      f"{EXPECT_DEPARTING_OK_LAB0}) — "
      f"{mark(roles['departing_ok_labeled_0'] == EXPECT_DEPARTING_OK_LAB0)}")
    a(f"- 端点行总数 {roles['n_endpoint_rows']} (期望 {EXPECT_ENDPOINT_ROWS})"
      f" — {mark(roles['n_endpoint_rows'] == EXPECT_ENDPOINT_ROWS)}")
    a("")
    a("## 3. 起点 / TS 误正例")
    a("")
    s = fp["start"]
    a(f"- 起点行 {s['n_rows']}：误正例 **{s['false_positive_count']}** "
      f"(期望 {EXPECT_START_FP}) — "
      f"{mark(s['false_positive_count'] == EXPECT_START_FP)}；"
      f"d_end min/median/max = {s['d_end_min']} / {s['d_end_median']} / "
      f"{s['d_end_max']} Å")
    t = fp["ts"]
    a(f"- TS 行 {t['n_rows']}：误正例 **{t['false_positive_count']}** → "
      f"curves {t['false_positive_curves']} (期望 2 行 / 恰为 "
      f"{list(EXPECT_TS_FP_CURVES)})")
    a(f"  - 含退化口径: {t['false_positive_count']}；排除退化曲线口径: "
      f"{t['false_positives_without_degenerate']}；degenerate_flag 行数 "
      f"{t['degenerate_flagged_rows']}（2 曲线 × 2 intent = 4 行）")
    a(f"  - 未 flag 的 TS 误正例: {len(t['unflagged_false_positives'])} — "
      f"{mark(not t['unflagged_false_positives'])}")
    for r in t["false_positives"]:
        a(f"  - `{r['curve_id']}` {r['direction']} d_end={r['d_end_angstrom']} Å"
          f" degenerate={r['degenerate_flag']}")
    a("")
    a("## 4. terminal 纳入证明（核心修复）")
    a("")
    a(f"- Stop 表内 no_future_contact 行: "
      f"**{term['n_terminal_rows_in_stop_table']}** (期望 "
      f"{EXPECT_TERMINAL_ROWS}) — "
      f"{mark(term['n_terminal_rows_in_stop_table'] == EXPECT_TERMINAL_ROWS)}")
    a(f"- 其中 label 1/0/Mask = {term['n_terminal_positive']} / "
      f"{term['n_terminal_negative']} / {term['n_terminal_mask']}")
    a(f"- masked/conflict 行同样纳入（几何标签独立于 Move eligibility）: "
      f"masked={term['n_masked_rows_in_stop_table']}, "
      f"conflict={term['n_conflict_rows_in_stop_table']}, "
      f"总行 {term['n_all_rows']}")
    a("")
    a("## 5. intent 对照表（到达→1 / 出发→0）")
    a("")
    a(f"- 曲线端点检查点: {contrast['n_curve_endpoints']}；违规 "
      f"**{contrast['n_violations']}** — "
      f"{mark(contrast['n_violations'] == 0)}")
    a("")
    a("| curve | A(fwd/rev) | B(fwd/rev) | ok |")
    a("|---|---|---|:--:|")
    by_curve: dict[str, dict[str, Any]] = {}
    for r in contrast["rows"]:
        by_curve.setdefault(r["curve_id"], {})[r["endpoint"]] = r
    for cid in sorted(by_curve):
        ra = by_curve[cid].get("A")
        rb = by_curve[cid].get("B")
        a(f"| {cid} | {ra['forward_label'] if ra else '-'}/"
          f"{ra['reverse_label'] if ra else '-'} | "
          f"{rb['forward_label'] if rb else '-'}/"
          f"{rb['reverse_label'] if rb else '-'} | "
          f"{'✓' if (ra and rb and ra['contrast_ok'] and rb['contrast_ok']) else '✗'} |")
    a("")
    a("## 6. 泄漏检查（split_manifest_r3.json）")
    a("")
    a(f"- 每曲线单一折: {leak['single_fold_per_curve']}；清单曲线数 "
      f"{leak['n_curves_in_manifest']}；重复分配 "
      f"{len(leak['duplicate_assignments'])}")
    a(f"- 每折曲线数: {leak['per_fold_curve_count']}")
    a(f"- dev folds 0–2: **{leak['dev_fold_curve_count']}** 条分配 (期望 "
      f"{EXPECT_DEV_CURVES})，其中有 ok 行 "
      f"**{leak['dev_fold_curves_with_ok_rows']}** 条 (期望 "
      f"{EXPECT_DEV_CURVES_WITH_OK}) — "
      f"{mark(leak['dev_fold_curve_count'] == EXPECT_DEV_CURVES and leak['dev_fold_curves_with_ok_rows'] == EXPECT_DEV_CURVES_WITH_OK)}")
    a(f"- sealed folds 3/4: {leak['sealed_fold_curve_count']} 条曲线，"
      "仅计数披露、未用于任何选择（untouched）")
    a("")
    a("## 7. 拓扑一致性 spot 审计（正例 cv-pair 距离代理）")
    a("")
    md = topo["per_row_mean_dev"]
    a(f"- 正例行 {topo['n_positive_rows']}，可评分 {topo['n_rows_scored']}"
      f"（无 cv_pair: {topo['n_rows_without_cv_pairs']}）")
    a(f"- 每行 mean |Δd_pair| (Å) 分布: min {md['min']} / p25 {md['p25']} / "
      f"median {md['median']} / p75 {md['p75']} / max {md['max']}")
    a(f"- 每行 max |Δd_pair| 中位: {topo['per_row_max_dev_median']} Å")
    a(f"- 任一 pair 偏离 > {topo['tolerance_angstrom']} Å 的行: "
      f"**{topo['n_outlier_rows']}**")
    for r in topo["outliers"][:20]:
        a(f"  - `{r['sample_id']}` ({r['curve_id']} node {r['node_index']} "
          f"{r['direction']}) mean={r['mean_dev_angstrom']} "
          f"max={r['max_dev_angstrom']} pairs={r['pair_devs_angstrom']}")
    if len(topo["outliers"]) > 20:
        a(f"  - … 共 {len(topo['outliers'])} 行，完整列表见 evidence_pack.json")
    a("")
    a("## 8. provenance")
    a("")
    for k in ("code_commit", "data_manifest_sha256",
              "teacher_manifest_sha256", "split_sha256", "label_schema",
              "label_spec_sha256", "config_sha256", "dataset_sha256"):
        v = str(prov.get(k, ""))
        a(f"- `{k}`: `{v[:16] + '…' if len(v) > 16 else v}`")
    a(f"- 七项齐全: {mark(prov['required_fields_present'])}")
    a("")
    a("## 9. Gate 判定（V5.2R-D0-LABELS）")
    a("")
    a("| 检查 | 期望 | 实测 | 判定 |")
    a("|---|---|---|:--:|")
    for c in gate["checks"]:
        a(f"| {c['id']} | {c['expect']} | {c['actual']} | {mark(c['pass'])} |")
    a("")
    a(f"**Overall: {gate['verdict']}**")
    a("")
    return "\n".join(lines)


# ----------------------------------------------------------------------- main


def run(tables_dir: Path, out_dir: Path) -> dict[str, Any]:
    dataset = KeyAtomDatasetV3(
        tables_dir,
        label_statuses=("ok", "terminal", "masked", "conflict"))
    base, grid = build_base_tables(dataset)
    label_stats = audit_label_stats(base, grid)
    roles = audit_endpoint_roles(base)
    fp = audit_false_positives(base, dataset)
    terminal = audit_terminal_inclusion(base)
    contrast = audit_intent_contrast(base)
    leak = audit_leakage(tables_dir, dataset)
    topo = audit_topology_proxy(dataset, base)
    prov = collect_provenance(tables_dir)
    gate = evaluate_gate(label_stats, fp, terminal, prov)
    pack: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "label_schema": SCHEMA,
        "gate": gate,
        "label_table_stats": label_stats,
        "endpoint_role_audit": roles,
        "false_positive_audit": fp,
        "terminal_inclusion": terminal,
        "intent_contrast": contrast,
        "leakage_checks": leak,
        "topology_consistency_proxy": topo,
        "provenance": prov,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evidence_pack.json").write_text(
        json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "D0_AUDIT.md").write_text(
        render_markdown(pack), encoding="utf-8")
    return pack


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR,
                    help="PD-1 six-table directory")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                    help="output directory for the evidence pack")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when the gate verdict is FAIL")
    args = ap.parse_args(argv)
    pack = run(args.tables_dir, args.out)
    gate = pack["gate"]
    print(f"[{gate['gate_id']}] verdict: {gate['verdict']}")
    for c in gate["checks"]:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['id']}: {c['actual']}")
    print(f"evidence: {args.out / 'evidence_pack.json'}")
    print(f"report:   {args.out / 'D0_AUDIT.md'}")
    if args.strict and gate["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
