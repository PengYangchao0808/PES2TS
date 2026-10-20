#!/usr/bin/env python3
"""V5.1a R1: per-pair error attribution over the frozen V5 OOF cache (CPU).

Lane: campaign V5_1A_R1_REPAIR — pair-error-attribution diagnostics.

Reads ``outputs/pesn/v5_1/diagnostics/oof_cache_v5.pkl`` (5088 OOF records,
arm A05 seed 17, five formal folds; READ-ONLY) and the frozen bucket
reference ``outputs/pesn/v5_1/diagnostics/g53_buckets.json`` (READ-ONLY),
then computes per-pair EXACT finite-step Δd under teacher and prediction:

    dd_x = ||(pos_j + x_j) − (pos_i + x_i)|| − ||pos_j − pos_i||

for every record pair (records enumerate only ``pair_edit != 0`` pairs,
identical to ``evaluate_g53`` node-axis segmentation) via
``pes2ts_core.pesn.offline.evaluation.key_atom_protocols.exact_delta_d``
(imported, never re-typed).  Buckets follow the frozen half-open edges
dead < 0.005 ≤ weak < 0.01 ≤ medium < 0.02 ≤ strong (a pair EXACTLY on an
edge belongs to the stronger bucket; abstention |dd_pred| < 0.005 counts
incorrect in the main sign denominator — G5.3).

Before anything is written, per-bucket pair COUNTS (and the integer
sub-counts n_sign_correct / n_abstained / forming / breaking / flips) are
asserted EQUAL to the frozen ``g53_buckets.json`` values (|Δn| == 0).

Outputs (deterministic, re-runnable, overwritten in place):

* ``pair_attribution.json``   schema ``pesn_v5_1_r1_pair_attribution_v1``
* ``pair_error_report_data.md`` draft data tables (sections a–f)

Sections (plan R1 lane spec):

a) reconciliation table vs the frozen buckets (n, sign acc, abstention,
   amplitude ratio per bucket + SRE block);
b) per-bucket residual stats: dd_pred−dd_true mean/median/q25/q75/rms and
   |dd_pred|/|dd_true| ratio median/q25/q75; sign flips forming vs breaking;
c) STRONG-bucket decomposition of sum(err²) into sign-wrong / under-shoot
   (sign-correct, |dd_pred| < 0.5·|dd_true|) / adequate-magnitude residual /
   abstentions (mutually exclusive, precedence documented in output), plus
   per-cluster error-share vs pair-share excess (top 8);
d) DEAD-bucket spurious-change profile (|dd_pred| quantiles + tail
   fractions, worst-8 clusters by mean |dd_pred|);
e) frame-level G5.3 SRE recomputed EXACTLY like ``evaluate_g53`` and
   decomposed by bucket of each pair (per_bucket_mean_contribution must
   match the frozen sre block within 1e-6);
f) near-TS (near_ts ≤ 0.20) vs far-TS split of strong-bucket sign accuracy
   and dead-bucket spurious rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION: Final[str] = "pesn_v5_1_r1_pair_attribution_v1"

#: Frozen bucket edges (g53_buckets.json thresholds block; half-open,
#: stronger bucket on the edge).
BUCKET_EDGES: Final[tuple[float, float, float]] = (0.005, 0.01, 0.02)
SIGN_THRESHOLD: Final[float] = 0.005
SRE_MIN_TOTAL_CHANGE: Final[float] = 0.01
NEAR_TS_CUT: Final[float] = 0.20
SRE_RECONCILE_TOL: Final[float] = 1e-6
#: Under-shoot cut for the strong-bucket decomposition (category ii).
UNDERSHOOT_FACTOR: Final[float] = 0.5

BUCKET_LABELS: Final[tuple[str, ...]] = (
    "dead_lt_0.005", "weak_0.005_0.01", "medium_0.01_0.02", "strong_ge_0.02")

DEFAULT_CACHE = (PROJECT_ROOT / "outputs/pesn/v5_1/diagnostics"
                 / "oof_cache_v5.pkl")
DEFAULT_FROZEN = (PROJECT_ROOT / "outputs/pesn/v5_1/diagnostics"
                  / "g53_buckets.json")
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/pesn/v5_1_r1/pair_errors"


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def import_with_retry(module_name: str, *, retries: int = 3,
                      delay_s: float = 30.0):
    """Import a core module, retrying transient failures (concurrent lane
    edits to ``key_atom_bc.py`` can briefly break imports)."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return __import__(module_name, fromlist=["*"])
        except Exception as exc:  # noqa: BLE001 — diagnostic lane: retry any
            last = exc
            if attempt < retries - 1:
                print(f"[r1_pair] import {module_name} failed "
                      f"({exc!r}); retrying in {delay_s:.0f}s "
                      f"({attempt + 1}/{retries})", file=sys.stderr)
                time.sleep(delay_s)
    raise ImportError(f"cannot import {module_name} after "
                      f"{retries} attempts: {last!r}")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def quantile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def bucket_of(abs_dd: float) -> int:
    """Frozen half-open bucketing: [0, e0) [e0, e1) [e1, e2) [e2, ∞)."""
    x = abs(float(abs_dd))
    if x < BUCKET_EDGES[0]:
        return 0
    if x < BUCKET_EDGES[1]:
        return 1
    if x < BUCKET_EDGES[2]:
        return 2
    return 3


def sign_correct(dd_true: float, dd_pred: float) -> bool:
    """G5.3 main-denominator correctness (abstention never free)."""
    return (np.sign(dd_pred) == np.sign(dd_true)
            and abs(dd_pred) >= SIGN_THRESHOLD)


# ---------------------------------------------------------------------------
# per-pair enumeration (exact Δd via the frozen protocol helper)
# ---------------------------------------------------------------------------

def pair_table(records: Sequence[dict], exact_delta_d) -> list[dict]:
    """Flatten OOF records into per-pair rows with exact dd_true/dd_pred."""
    rows: list[dict] = []
    for r_idx, r in enumerate(records):
        pairs = r["pairs"]
        if not pairs:
            continue
        dd_true = exact_delta_d(r["pos"], r["key_rows"], pairs, r["y"])
        dd_pred = exact_delta_d(r["pos"], r["key_rows"], pairs, r["u"])
        near = float(r.get("near_ts", 1.0))
        for l, (i, j) in enumerate(pairs):
            ddt = float(dd_true[l])
            ddp = float(dd_pred[l])
            rows.append({
                "group_id": str(r["group_id"]),
                "fold": int(r["fold"]),
                "branch": int(r["branch"]),
                "pair": (int(i), int(j)),
                "pair_edit": float(r["pair_edit"][l]),
                "dd_true": ddt,
                "dd_pred": ddp,
                "bucket": bucket_of(abs(ddt)),
                "near_ts": near,
                "is_near": near <= NEAR_TS_CUT,
                "abstained": abs(ddp) < SIGN_THRESHOLD,
                "sign_correct": sign_correct(ddt, ddp),
                "sre_frame_id": r_idx,
            })
    return rows


# ---------------------------------------------------------------------------
# a) reconciliation vs frozen g53_buckets.json
# ---------------------------------------------------------------------------

#: Integer fields that must match EXACTLY (hard assertions).
#: Continuous fields are compared with a tolerance and reported as deltas.
FROZEN_INT_FIELDS: Final[tuple[str, ...]] = (
    "n_pairs", "n_sign_correct", "n_abstained", "n_predicted",
    "n_forming", "n_breaking", "sign_flips_forming", "sign_flips_breaking")
FROZEN_FLOAT_FIELDS: Final[tuple[str, ...]] = (
    "sign_accuracy", "abstention_rate", "sign_accuracy_where_predicted",
    "amplitude_ratio_mean")


def bucketize(rows: Sequence[dict]) -> dict:
    """Per-bucket counters mirroring diagnose_v51_g53_buckets.bucketize."""
    acc = {b: {"n_pairs": 0, "n_sign_correct": 0, "n_abstained": 0,
               "n_predicted": 0, "n_predicted_sign_correct": 0,
               "abs_dd_true_sum": 0.0, "abs_dd_pred_sum": 0.0,
               "n_forming": 0, "n_breaking": 0,
               "sign_flips_forming": 0, "sign_flips_breaking": 0}
           for b in range(4)}
    for row in rows:
        b = row["bucket"]
        a = acc[b]
        a["n_pairs"] += 1
        a["n_abstained"] += int(row["abstained"])
        if not row["abstained"]:
            a["n_predicted"] += 1
            a["n_predicted_sign_correct"] += int(
                np.sign(row["dd_pred"]) == np.sign(row["dd_true"]))
        a["n_sign_correct"] += int(row["sign_correct"])
        a["abs_dd_true_sum"] += abs(row["dd_true"])
        a["abs_dd_pred_sum"] += abs(row["dd_pred"])
        if row["pair_edit"] > 0:
            a["n_forming"] += 1
            if (not row["abstained"]) and np.sign(row["dd_pred"]) > 0:
                a["sign_flips_forming"] += 1
        elif row["pair_edit"] < 0:
            a["n_breaking"] += 1
            if (not row["abstained"]) and np.sign(row["dd_pred"]) < 0:
                a["sign_flips_breaking"] += 1
    out = {}
    for b in range(4):
        a = acc[b]
        n = a["n_pairs"]
        out[str(b)] = {
            "label": BUCKET_LABELS[b],
            "n_pairs": n,
            "n_sign_correct": a["n_sign_correct"],
            "n_abstained": a["n_abstained"],
            "n_predicted": a["n_predicted"],
            "sign_accuracy": (a["n_sign_correct"] / n) if n else None,
            "abstention_rate": (a["n_abstained"] / n) if n else None,
            "sign_accuracy_where_predicted": (
                a["n_predicted_sign_correct"] / a["n_predicted"]
                if a["n_predicted"] else None),
            "amplitude_ratio_mean": (
                a["abs_dd_pred_sum"] / a["abs_dd_true_sum"]
                if a["abs_dd_true_sum"] > 0 else None),
            "n_forming": a["n_forming"],
            "n_breaking": a["n_breaking"],
            "sign_flips_forming": a["sign_flips_forming"],
            "sign_flips_breaking": a["sign_flips_breaking"],
        }
    return out


def reconcile(ours: dict, frozen: dict, n_rows: int) -> dict:
    """Per-bucket reconciliation; hard-asserts integer equality per the
    R1 lane contract (|your_n − frozen_n| == 0 per bucket)."""
    comparison: dict[str, dict] = {}
    hard_failures: list[str] = []
    for b in range(4):
        key = str(b)
        mine, ref = ours[key], frozen["buckets"][key]
        entry: dict = {"label": BUCKET_LABELS[b], "int_fields": {},
                       "float_fields": {}}
        for f in FROZEN_INT_FIELDS:
            delta = int(mine[f]) - int(ref[f])
            entry["int_fields"][f] = {"ours": int(mine[f]),
                                      "frozen": int(ref[f]), "delta": delta}
            if delta != 0:
                hard_failures.append(f"bucket {b} field {f}: |delta|={abs(delta)}")
        for f in FROZEN_FLOAT_FIELDS:
            mv, rv = mine.get(f), ref.get(f)
            delta = (None if mv is None or rv is None
                     else float(mv) - float(rv))
            entry["float_fields"][f] = {"ours": mv, "frozen": rv,
                                        "delta": delta}
        comparison[key] = entry
    if int(n_rows) != int(frozen.get("n_pairs", n_rows)):
        hard_failures.append(
            f"total n_pairs: ours {n_rows} vs frozen {frozen.get('n_pairs')}")
    return {
        "per_bucket": comparison,
        "total_n_pairs": {"ours": int(n_rows),
                          "frozen": int(frozen.get("n_pairs", -1)),
                          "delta": int(n_rows) - int(frozen.get("n_pairs", -1))},
        "hard_assertions_passed": not hard_failures,
        "hard_failures": hard_failures,
        "assertion_policy": ("integer fields (n_pairs, n_sign_correct, "
                             "n_abstained, n_predicted, forming/breaking and "
                             "flip counts) must match frozen EXACTLY "
                             "(|delta|==0); continuous fields reported as "
                             "deltas, not asserted"),
    }


# ---------------------------------------------------------------------------
# b) per-bucket residual stats
# ---------------------------------------------------------------------------

def residual_stats(rows: Sequence[dict]) -> dict:
    out: dict[str, dict] = {}
    for b in range(4):
        sel = [r for r in rows if r["bucket"] == b]
        err = np.asarray([r["dd_pred"] - r["dd_true"] for r in sel],
                         dtype=np.float64)
        at = np.asarray([abs(r["dd_true"]) for r in sel], dtype=np.float64)
        ap = np.asarray([abs(r["dd_pred"]) for r in sel], dtype=np.float64)
        nz = at >= 1e-12
        ratio = ap[nz] / at[nz] if bool(nz.any()) else np.zeros(0)
        flips_forming = sum(1 for r in sel if r["pair_edit"] > 0
                            and not r["abstained"]
                            and np.sign(r["dd_pred"]) > 0)
        flips_breaking = sum(1 for r in sel if r["pair_edit"] < 0
                             and not r["abstained"]
                             and np.sign(r["dd_pred"]) < 0)
        out[str(b)] = {
            "label": BUCKET_LABELS[b],
            "n_pairs": len(sel),
            "residual_dd_pred_minus_dd_true": {
                "mean": float(err.mean()) if err.size else None,
                "median": quantile(err, 0.5),
                "q25": quantile(err, 0.25),
                "q75": quantile(err, 0.75),
                "rms": (float(np.sqrt(np.mean(err ** 2)))
                        if err.size else None),
            },
            "abs_ratio_pred_over_true": {
                "n_defined": int(nz.sum()),
                "n_zero_true": int((~nz).sum()),
                "median": quantile(ratio, 0.5),
                "q25": quantile(ratio, 0.25),
                "q75": quantile(ratio, 0.75),
                "note": ("per-pair |dd_pred|/|dd_true| defined only where "
                         "|dd_true| >= 1e-12 (dead-bucket zeros excluded)"),
            },
            "sign_flips": {
                "forming_predicted_longer": int(flips_forming),
                "breaking_predicted_shorter": int(flips_breaking),
            },
        }
    return out


# ---------------------------------------------------------------------------
# c) strong-bucket squared-error decomposition
# ---------------------------------------------------------------------------

#: Category precedence (mutually exclusive partition of the strong bucket):
#:   iv) abstention (|dd_pred| < 0.005) — checked FIRST, so an abstention
#:       with an accidental matching sign still lands in (iv);
#:   i)  sign-wrong (non-abstained, sign mismatch);
#:   ii) sign-correct under-shoot: |dd_pred| < 0.5 * |dd_true|;
#:   iii) sign-correct adequate-magnitude residual.
STRONG_DECOMP_PRECEDENCE: Final[tuple[str, ...]] = (
    "abstention", "sign_wrong", "undershoot", "adequate")


def strong_decomposition(rows: Sequence[dict]) -> dict:
    sel = [r for r in rows if r["bucket"] == 3]
    cats = {c: {"n_pairs": 0, "sum_sq_err": 0.0}
            for c in STRONG_DECOMP_PRECEDENCE}
    cluster_err: dict[str, float] = {}
    cluster_n: dict[str, int] = {}
    total_sq = 0.0
    for r in sel:
        err2 = (r["dd_pred"] - r["dd_true"]) ** 2
        total_sq += err2
        if r["abstained"]:
            c = "abstention"
        elif np.sign(r["dd_pred"]) != np.sign(r["dd_true"]):
            c = "sign_wrong"
        elif abs(r["dd_pred"]) < UNDERSHOOT_FACTOR * abs(r["dd_true"]):
            c = "undershoot"
        else:
            c = "adequate"
        cats[c]["n_pairs"] += 1
        cats[c]["sum_sq_err"] += err2
        cluster_err[r["group_id"]] = cluster_err.get(r["group_id"], 0.0) + err2
        cluster_n[r["group_id"]] = cluster_n.get(r["group_id"], 0) + 1
    out_cats = {}
    for c in STRONG_DECOMP_PRECEDENCE:
        e = cats[c]
        out_cats[c] = {
            "n_pairs": e["n_pairs"],
            "sum_sq_err_ang2": e["sum_sq_err"],
            "share_of_sum_sq_err": (
                e["sum_sq_err"] / total_sq if total_sq > 0 else None),
            "mean_sq_err_per_pair": (
                e["sum_sq_err"] / e["n_pairs"] if e["n_pairs"] else None),
        }
    # per-cluster error share vs pair share
    n_total = len(sel)
    per_cluster = []
    for gid in sorted(cluster_err):
        err_share = (cluster_err[gid] / total_sq) if total_sq > 0 else 0.0
        pair_share = cluster_n[gid] / n_total if n_total else 0.0
        per_cluster.append({
            "group_id": gid,
            "n_pairs": cluster_n[gid],
            "error_share": err_share,
            "pair_share": pair_share,
            "excess_error_share": err_share - pair_share,
        })
    top8 = sorted(per_cluster,
                  key=lambda e: (-e["excess_error_share"], e["group_id"]))[:8]
    return {
        "n_strong_pairs": n_total,
        "total_sum_sq_err_ang2": total_sq,
        "category_precedence": list(STRONG_DECOMP_PRECEDENCE),
        "categories": out_cats,
        "top8_clusters_by_excess_error_share": top8,
        "per_cluster": per_cluster,
    }


# ---------------------------------------------------------------------------
# d) dead-bucket spurious-change profile
# ---------------------------------------------------------------------------

def dead_profile(rows: Sequence[dict]) -> dict:
    sel = [r for r in rows if r["bucket"] == 0]
    ap = np.asarray([abs(r["dd_pred"]) for r in sel], dtype=np.float64)
    per_cluster: dict[str, list[float]] = {}
    for r in sel:
        per_cluster.setdefault(r["group_id"], []).append(abs(r["dd_pred"]))
    cluster_means = sorted(
        ((gid, float(np.mean(v)), len(v))
         for gid, v in per_cluster.items()),
        key=lambda t: (-t[1], t[0]))
    return {
        "n_dead_pairs": len(sel),
        "abs_dd_pred": {
            "median": quantile(ap, 0.5),
            "q75": quantile(ap, 0.75),
            "q95": quantile(ap, 0.95),
        },
        "tail_fractions": {
            "fraction_gt_0.005": (float((ap > 0.005).mean())
                                  if ap.size else None),
            "fraction_gt_0.01": (float((ap > 0.01).mean())
                                 if ap.size else None),
            "fraction_gt_0.02": (float((ap > 0.02).mean())
                                 if ap.size else None),
        },
        "worst8_clusters_by_mean_abs_dd_pred": [
            {"group_id": gid, "mean_abs_dd_pred": m, "n_pairs": n}
            for gid, m, n in cluster_means[:8]],
        "definition": ("dead bucket = |dd_true| < 0.005 A; spurious change "
                       "= predicted |dd_pred| above the dead-zone threshold"),
    }


# ---------------------------------------------------------------------------
# e) frame-level SRE attribution (exact evaluate_g53 mirror)
# ---------------------------------------------------------------------------

def sre_attribution(rows: Sequence[dict], frozen_sre: dict) -> dict:
    """Frame SRE EXACTLY like evaluate_g53; per-bucket term decomposition.

    Mirrors diagnose_v51_g53_buckets.sre_frame_contribution: frames with
    >= 2 pairs and total teacher |Δd| >= 0.01 A; SRE = 0.5·Σ|p̂−p| on
    |Δd| shares; a fully-zero prediction maps to SRE = 1.0 attributed to
    the bucket holding the largest teacher share.
    """
    frames: dict[int, list[dict]] = {}
    for r in rows:
        frames.setdefault(r["sre_frame_id"], []).append(r)
    sre_all: list[float] = []
    bucket_terms: dict[str, list[float]] = {str(b): [] for b in range(4)}
    n_zero_pred = 0
    for _fid in sorted(frames):
        frows = frames[_fid]
        if len(frows) < 2:
            continue
        ddt = np.asarray([r["dd_true"] for r in frows], dtype=np.float64)
        ddp = np.asarray([r["dd_pred"] for r in frows], dtype=np.float64)
        pb = np.asarray([r["bucket"] for r in frows])
        total = float(np.abs(ddt).sum())
        if total < SRE_MIN_TOTAL_CHANGE:
            continue
        p = np.abs(ddt) / total
        pred_change = float(np.abs(ddp).sum())
        if pred_change <= 0.0:
            n_zero_pred += 1
            contrib = {int(pb[int(np.argmax(p))]): 1.0}
        else:
            p_hat = np.abs(ddp) / pred_change
            terms = 0.5 * np.abs(p_hat - p)
            contrib = {int(b): float(terms[pb == b].sum())
                       for b in np.unique(pb)}
        sre_all.append(sum(contrib.values()))
        for b in range(4):
            bucket_terms[str(b)].append(contrib.get(b, 0.0))
    sre_arr = np.asarray(sre_all, dtype=np.float64)
    ours_contrib = {b: (float(np.mean(v)) if v else None)
                    for b, v in bucket_terms.items()}
    frozen_contrib = dict(frozen_sre.get("per_bucket_mean_contribution", {}))
    check = {}
    ok = True
    for b in range(4):
        key = str(b)
        mv, rv = ours_contrib.get(key), frozen_contrib.get(key)
        delta = (None if mv is None or rv is None else mv - rv)
        within = (delta is not None and abs(delta) <= SRE_RECONCILE_TOL)
        ok = ok and within
        check[key] = {"ours": mv, "frozen": rv, "delta": delta,
                      "within_1e-6": bool(within)}
    return {
        "n_frames": len(sre_all),
        "n_zero_prediction_frames": n_zero_pred,
        "sre_median": quantile(sre_arr, 0.5),
        "sre_q75": quantile(sre_arr, 0.75),
        "mean_frame_sre": (float(sre_arr.mean()) if sre_arr.size else None),
        "per_bucket_mean_contribution": ours_contrib,
        "reconciliation_vs_frozen": check,
        "reconciliation_passed": bool(ok),
        "definition": ("G5.3 SRE on frames with >=2 pairs and total teacher "
                       "|Δd| >= 0.01 A (evaluate_g53 semantics, mirrored "
                       "verbatim); bucket contribution = the 0.5*|p_hat-p| "
                       "terms of that bucket's pairs; zero-prediction frames "
                       "(SRE=1) attributed to the dominant teacher bucket"),
    }


# ---------------------------------------------------------------------------
# f) near-TS vs far-TS split
# ---------------------------------------------------------------------------

def near_ts_split(rows: Sequence[dict]) -> dict:
    out: dict[str, dict] = {}
    for name, pred in (("near_ts_le_0.20", True), ("far_ts_gt_0.20", False)):
        sub = [r for r in rows if r["is_near"] == pred]
        strong = [r for r in sub if r["bucket"] == 3]
        dead = [r for r in sub if r["bucket"] == 0]
        n_strong = len(strong)
        n_dead = len(dead)
        out[name] = {
            "n_pairs_all": len(sub),
            "strong_bucket": {
                "n_pairs": n_strong,
                "n_sign_correct": sum(1 for r in strong if r["sign_correct"]),
                "sign_accuracy": (sum(1 for r in strong if r["sign_correct"])
                                  / n_strong if n_strong else None),
                "abstention_rate": (
                    sum(1 for r in strong if r["abstained"]) / n_strong
                    if n_strong else None),
            },
            "dead_bucket": {
                "n_pairs": n_dead,
                "n_spurious_ge_0.005": sum(
                    1 for r in dead if abs(r["dd_pred"]) >= SIGN_THRESHOLD),
                "spurious_rate_ge_0.005": (
                    sum(1 for r in dead if abs(r["dd_pred"]) >= SIGN_THRESHOLD)
                    / n_dead if n_dead else None),
                "spurious_rate_gt_0.01": (
                    sum(1 for r in dead if abs(r["dd_pred"]) > 0.01) / n_dead
                    if n_dead else None),
                "spurious_rate_gt_0.02": (
                    sum(1 for r in dead if abs(r["dd_pred"]) > 0.02) / n_dead
                    if n_dead else None),
            },
        }
    return {
        "near_ts_cut": NEAR_TS_CUT,
        "definition": ("near-TS iff frame near_ts <= 0.20 (same cut as "
                       "compute_cluster_metrics cos_near); strong sign "
                       "accuracy uses the G5.3 main denominator; dead "
                       "spurious rate = fraction of dead pairs with "
                       "|dd_pred| >= 0.005 A"),
        "splits": out,
    }


# ---------------------------------------------------------------------------
# markdown draft
# ---------------------------------------------------------------------------

def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{nd}f}"


def write_markdown(path: Path, doc: dict) -> None:
    lines: list[str] = []
    add = lines.append
    add("# R1 pair-error attribution — data tables (DRAFT)")
    add("")
    add(f"- schema: `{doc['schema_version']}`  generated: {doc['generated_at']}")
    add(f"- input cache: `{doc['inputs']['oof_cache']}` "
        f"(sha256 `{doc['inputs']['oof_cache_sha256'][:16]}…`)")
    add(f"- frozen reference: `{doc['inputs']['frozen_buckets']}` "
        f"(sha256 `{doc['inputs']['frozen_buckets_sha256'][:16]}…`)")
    add(f"- n_records={doc['n_records']}  n_pairs={doc['n_pairs']}  "
        f"hard reconciliation assertions: "
        f"{'PASS' if doc['reconciliation']['hard_assertions_passed'] else 'FAIL'}")
    add("")

    add("## a) Reconciliation vs frozen g53_buckets.json")
    add("")
    add("| bucket | n ours | n frozen | Δn | sign acc ours | frozen | Δ | "
        "abstention ours | frozen | Δ | amp ratio ours | frozen | Δ |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in range(4):
        c = doc["reconciliation"]["per_bucket"][str(b)]
        i = c["int_fields"]
        f = c["float_fields"]
        add(f"| {BUCKET_LABELS[b]} | {i['n_pairs']['ours']} | "
            f"{i['n_pairs']['frozen']} | {i['n_pairs']['delta']} | "
            f"{_fmt(f['sign_accuracy']['ours'])} | "
            f"{_fmt(f['sign_accuracy']['frozen'])} | "
            f"{_fmt(f['sign_accuracy']['delta'], 6)} | "
            f"{_fmt(f['abstention_rate']['ours'])} | "
            f"{_fmt(f['abstention_rate']['frozen'])} | "
            f"{_fmt(f['abstention_rate']['delta'], 6)} | "
            f"{_fmt(f['amplitude_ratio_mean']['ours'])} | "
            f"{_fmt(f['amplitude_ratio_mean']['frozen'])} | "
            f"{_fmt(f['amplitude_ratio_mean']['delta'], 6)} |")
    add("")
    add("SRE block reconciliation (per-bucket mean contribution, tol 1e-6):")
    add("")
    add("| bucket | ours | frozen | Δ | within 1e-6 |")
    add("|---|---|---|---|---|")
    sre = doc["sre_attribution"]
    for b in range(4):
        chk = sre["reconciliation_vs_frozen"][str(b)]
        add(f"| {BUCKET_LABELS[b]} | {_fmt(chk['ours'], 6)} | "
            f"{_fmt(chk['frozen'], 6)} | {_fmt(chk['delta'], 8)} | "
            f"{chk['within_1e-6']} |")
    add("")
    add(f"SRE frames: ours {sre['n_frames']} (zero-prediction frames: "
        f"{sre['n_zero_prediction_frames']}), median "
        f"{_fmt(sre['sre_median'])}, q75 {_fmt(sre['sre_q75'])}.")
    add("")

    add("## b) Per-bucket residual statistics")
    add("")
    add("| bucket | resid mean | median | q25 | q75 | rms | "
        "|dd_pred|/|dd_true| median | q25 | q75 | flips forming | flips breaking |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for b in range(4):
        e = doc["residual_stats"][str(b)]
        r = e["residual_dd_pred_minus_dd_true"]
        q = e["abs_ratio_pred_over_true"]
        fl = e["sign_flips"]
        add(f"| {BUCKET_LABELS[b]} | {_fmt(r['mean'], 6)} | "
            f"{_fmt(r['median'], 6)} | {_fmt(r['q25'], 6)} | "
            f"{_fmt(r['q75'], 6)} | {_fmt(r['rms'], 6)} | "
            f"{_fmt(q['median'])} | {_fmt(q['q25'])} | {_fmt(q['q75'])} | "
            f"{fl['forming_predicted_longer']} | "
            f"{fl['breaking_predicted_shorter']} |")
    add("")

    add("## c) Strong-bucket squared-error decomposition")
    add("")
    sd = doc["strong_decomposition"]
    add(f"Strong pairs: {sd['n_strong_pairs']}, total sum(err²) = "
        f"{_fmt(sd['total_sum_sq_err_ang2'], 6)} Å².  Category precedence: "
        f"{', '.join(sd['category_precedence'])} (first match wins; the four "
        f"categories partition the bucket).")
    add("")
    add("| category | n_pairs | sum err² (Å²) | share | mean err²/pair |")
    add("|---|---|---|---|---|")
    for c in sd["category_precedence"]:
        e = sd["categories"][c]
        add(f"| {c} | {e['n_pairs']} | {_fmt(e['sum_sq_err_ang2'], 6)} | "
            f"{_fmt(e['share_of_sum_sq_err'])} | "
            f"{_fmt(e['mean_sq_err_per_pair'], 8)} |")
    add("")
    add("Top 8 clusters by excess strong-bucket error share "
        "(error_share − pair_share):")
    add("")
    add("| cluster | n strong pairs | error share | pair share | excess |")
    add("|---|---|---|---|---|")
    for e in sd["top8_clusters_by_excess_error_share"]:
        add(f"| {e['group_id']} | {e['n_pairs']} | {_fmt(e['error_share'])} | "
            f"{_fmt(e['pair_share'])} | {_fmt(e['excess_error_share'])} |")
    add("")

    add("## d) Dead-bucket spurious-change profile")
    add("")
    dp = doc["dead_profile"]
    add(f"Dead pairs: {dp['n_dead_pairs']}.  |dd_pred| median "
        f"{_fmt(dp['abs_dd_pred']['median'], 6)}, q75 "
        f"{_fmt(dp['abs_dd_pred']['q75'], 6)}, q95 "
        f"{_fmt(dp['abs_dd_pred']['q95'], 6)}.")
    add("")
    add(f"- fraction |dd_pred| > 0.005: {_fmt(dp['tail_fractions']['fraction_gt_0.005'])}")
    add(f"- fraction |dd_pred| > 0.01:  {_fmt(dp['tail_fractions']['fraction_gt_0.01'])}")
    add(f"- fraction |dd_pred| > 0.02:  {_fmt(dp['tail_fractions']['fraction_gt_0.02'])}")
    add("")
    add("Worst 8 clusters by dead-pair mean |dd_pred|:")
    add("")
    add("| cluster | mean |dd_pred| | n dead pairs |")
    add("|---|---|---|")
    for e in dp["worst8_clusters_by_mean_abs_dd_pred"]:
        add(f"| {e['group_id']} | {_fmt(e['mean_abs_dd_pred'], 6)} | "
            f"{e['n_pairs']} |")
    add("")

    add("## e) Frame-level SRE attribution")
    add("")
    add("| bucket | mean contribution (ours) | frozen | share of mean SRE |")
    add("|---|---|---|---|")
    mean_sre = float(sre["mean_frame_sre"])
    for b in range(4):
        contrib = sre["per_bucket_mean_contribution"][str(b)]
        frozen_v = sre["reconciliation_vs_frozen"][str(b)]["frozen"]
        share = (contrib / mean_sre if contrib is not None and mean_sre > 0
                 else None)
        add(f"| {BUCKET_LABELS[b]} | {_fmt(contrib, 6)} | "
            f"{_fmt(frozen_v, 6)} | {_fmt(share, 4)} |")
    add("")
    add(f"Mean frame SRE = {_fmt(mean_sre, 6)}; sum of bucket means = "
        f"{_fmt(sum(v for v in sre['per_bucket_mean_contribution'].values() if v is not None), 6)}; "
        f"consistency: {'PASS' if sre['reconciliation_passed'] else 'FAIL'} "
        f"(each bucket |Δ| ≤ 1e-6).")
    add("")

    add("## f) Near-TS vs far-TS split")
    add("")
    nt = doc["near_ts_split"]
    add(f"Cut: near_ts ≤ {nt['near_ts_cut']}.")
    add("")
    add("| split | strong n | strong sign acc | strong abstain | "
        "dead n | dead spurious ≥0.005 | >0.01 | >0.02 |")
    add("|---|---|---|---|---|---|---|---|")
    for name in ("near_ts_le_0.20", "far_ts_gt_0.20"):
        e = nt["splits"][name]
        s, d = e["strong_bucket"], e["dead_bucket"]
        add(f"| {name} | {s['n_pairs']} | {_fmt(s['sign_accuracy'])} | "
            f"{_fmt(s['abstention_rate'])} | {d['n_pairs']} | "
            f"{_fmt(d['spurious_rate_ge_0.005'])} | "
            f"{_fmt(d['spurious_rate_gt_0.01'])} | "
            f"{_fmt(d['spurious_rate_gt_0.02'])} |")
    add("")
    add("_Draft data tables only — narrative/interpretation belongs to the "
        "R1 lane report, not to this generated file._")
    add("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--frozen", default=str(DEFAULT_FROZEN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    cache_path = Path(args.cache)
    frozen_path = Path(args.frozen)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key_atom_protocols = import_with_retry(
        "pes2ts_core.pesn.offline.evaluation.key_atom_protocols")
    exact_delta_d = key_atom_protocols.exact_delta_d

    with cache_path.open("rb") as fh:
        cache = pickle.load(fh)
    if cache.get("schema_version") != "pesn_v51_oof_cache_v1":
        raise SystemExit(f"unexpected cache schema "
                         f"{cache.get('schema_version')!r}")
    records = cache["records"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != "pesn_v51_g53_buckets_diag_v1":
        raise SystemExit(f"unexpected frozen schema "
                         f"{frozen.get('schema_version')!r}")

    rows = pair_table(records, exact_delta_d)
    ours = bucketize(rows)
    recon = reconcile(ours, frozen, len(rows))
    if not recon["hard_assertions_passed"]:
        for msg in recon["hard_failures"]:
            print(f"[r1_pair] RECONCILIATION FAILURE: {msg}", file=sys.stderr)
        raise SystemExit("hard reconciliation assertions failed — refusing "
                         "to write outputs (frozen g53_buckets mismatch)")

    res_stats = residual_stats(rows)
    sdecomp = strong_decomposition(rows)
    dprof = dead_profile(rows)
    sre = sre_attribution(rows, frozen["sre"])
    if not sre["reconciliation_passed"]:
        for b, chk in sre["reconciliation_vs_frozen"].items():
            if not chk["within_1e-6"]:
                print(f"[r1_pair] SRE bucket {b} delta {chk['delta']} "
                      f"> 1e-6", file=sys.stderr)
        raise SystemExit("SRE per_bucket_mean_contribution does not "
                         "reconcile with frozen g53_buckets within 1e-6")
    ntsplit = near_ts_split(rows)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "n_records": len(records),
        "n_pairs": len(rows),
        "inputs": {
            "oof_cache": str(cache_path),
            "oof_cache_sha256": sha256_file(cache_path),
            "oof_fingerprint": cache.get("fingerprint"),
            "frozen_buckets": str(frozen_path),
            "frozen_buckets_sha256": sha256_file(frozen_path),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "thresholds": {
            "sign_pair_threshold_angstrom": SIGN_THRESHOLD,
            "bucket_edges_angstrom": list(BUCKET_EDGES),
            "bucket_edge_policy": ("half-open [lo, hi): a pair EXACTLY on an "
                                   "edge belongs to the stronger bucket"),
            "abstention_policy": ("predicted |Δd| < 0.005 counts incorrect in "
                                  "the main sign denominator (G5.3)"),
            "sre_min_total_change_angstrom": SRE_MIN_TOTAL_CHANGE,
            "near_ts_cut": NEAR_TS_CUT,
            "undershoot_factor": UNDERSHOOT_FACTOR,
        },
        "reconciliation": recon,
        "buckets_ours": ours,
        "residual_stats": res_stats,
        "strong_decomposition": sdecomp,
        "dead_profile": dprof,
        "sre_attribution": sre,
        "near_ts_split": ntsplit,
    }

    json_path = out_dir / "pair_attribution.json"
    json_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    md_path = out_dir / "pair_error_report_data.md"
    write_markdown(md_path, doc)

    print(f"[r1_pair] wrote {json_path}")
    print(f"[r1_pair] wrote {md_path}")
    print(f"  n_records={len(records)} n_pairs={len(rows)}")
    for b in range(4):
        e = ours[str(b)]
        print(f"  {BUCKET_LABELS[b]:20s} n={e['n_pairs']:6d} "
              f"sign={_fmt(e['sign_accuracy'])} "
              f"abstain={_fmt(e['abstention_rate'])} "
              f"amp={_fmt(e['amplitude_ratio_mean'])} "
              f"(frozen Δn={recon['per_bucket'][str(b)]['int_fields']['n_pairs']['delta']})")
    sd = sdecomp["categories"]
    print("  strong err² shares: " + ", ".join(
        f"{c}={_fmt(sd[c]['share_of_sum_sq_err'])}"
        for c in STRONG_DECOMP_PRECEDENCE))
    print(f"  SRE bucket means reconcile: {sre['reconciliation_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
