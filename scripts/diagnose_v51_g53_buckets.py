#!/usr/bin/env python3
"""V5.1a D2: where does the G5.3 sign/SRE failure live? (plan §3.1 D2).

Runs on the SAME five-fold OOF predictions as the baseline re-derivation
(reuses :func:`diagnose_v51_baseline_oof.collect_oof_records`, including its
prediction cache).  Every key pair enumerated exactly like
``evaluate_g53`` (node-axis segmentation + ``pair_edit != 0``) gets its
EXACT finite-step Δd under both teacher y and prediction u, then buckets by
``|Δd_true|``:

    [0, 0.005)   dead zone (noise / abstention region)
    [0.005, 0.01)   weak signal
    [0.01, 0.02)    medium signal
    [0.02, inf)     strong signal

Per bucket: sign accuracy (abstention = incorrect in the main denominator,
per G5.3), the |Δd_pred|<0.005 abstention rate, sign accuracy conditional
on predicting anyway, SRE contribution, amplitude ratio
mean|Δd_pred|/|Δd_true|, and bond-forming vs bond-breaking sign flips
(pair_edit sign vs predicted Δd sign).  Plus a per-cluster × bucket matrix
for all 49 clusters and a verdict block implementing the plan decision
rule:

    strong > 0.9 and weak < 0.8 -> dead-zone/amplitude problem,
                                    pair margin loss first
    strong < 0.8                 -> representation/feature/label problem,
                                    loss alone insufficient
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diagnose_v51_baseline_oof import (  # noqa: E402
    BUCKET_LABELS,
    DEAD_ZONE_MAG,
    D0,
    DEFAULT_CACHE,
    DEFAULT_OUT,
    DEFAULT_RUNS_ROOT,
    DEFAULT_SPLIT,
    bucket_of,
    bucket_verdict,
    collect_oof_records,
    manifest_digest,
)
from diagnose_v51_baseline_oof import (  # noqa: E402
    _oof_fingerprint as oof_fingerprint,
    _quantile as quantile,
    _utc_now as utc_now,
)

SCHEMA_VERSION: Final[str] = "pesn_v51_g53_buckets_diag_v1"
SIGN_THRESHOLD: Final[float] = 0.005
SRE_MIN_TOTAL_CHANGE: Final[float] = 0.01

DEFAULT_OUT = (PROJECT_ROOT / "outputs/pesn/v5_1/diagnostics"
               / "g53_buckets.json")


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def pair_rows(records: Sequence[dict]):
    """Flatten OOF records into per-pair rows.

    Each row: ``group_id``, ``fold``, ``branch``, ``dd_true``, ``dd_pred``,
    ``pair_edit`` (signed ΔB: >0 forming, <0 breaking), ``sre_frame_id``.
    Δd is EXACT finite-step (``exact_delta_d`` semantics: current-frame r,
    key-slot resolution).
    """
    from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
        exact_delta_d)
    rows: list[dict] = []
    for r_idx, r in enumerate(records):
        pairs = r["pairs"]
        if not pairs:
            continue
        dd_true = exact_delta_d(r["pos"], r["key_rows"], pairs, r["y"])
        dd_pred = exact_delta_d(r["pos"], r["key_rows"], pairs, r["u"])
        for l, (i, j) in enumerate(pairs):
            rows.append({
                "group_id": r["group_id"],
                "fold": r["fold"],
                "branch": r["branch"],
                "pair": (int(i), int(j)),
                "pair_edit": float(r["pair_edit"][l]),
                "dd_true": float(dd_true[l]),
                "dd_pred": float(dd_pred[l]),
                "sre_frame_id": r_idx,
            })
    return rows


def sre_frame_contribution(dd_true: np.ndarray, dd_pred: np.ndarray,
                           pair_bucket: np.ndarray) -> dict[int, float] | None:
    """Per-bucket SRE numerator contribution for one frame (G5.3 SRE).

    Frame qualifies when it has >= 2 pairs and total teacher |Δd| >= 0.01 A
    (G5.3).  SRE = 0.5 * sum_l |p_hat_l - p_l| with p from |Δd| shares; the
    returned dict maps bucket -> that pair's 0.5*|p_hat-p| term (the frame's
    SRE is the sum over buckets).  A fully-zero prediction maps to SRE=1
    (reported under bucket of the largest teacher share).
    """
    dd_true = np.asarray(dd_true, dtype=np.float64)
    dd_pred = np.asarray(dd_pred, dtype=np.float64)
    if len(dd_true) < 2:
        return None
    total = float(np.abs(dd_true).sum())
    if total < SRE_MIN_TOTAL_CHANGE:
        return None
    p = np.abs(dd_true) / total
    pred_change = float(np.abs(dd_pred).sum())
    if pred_change <= 0.0:
        out: dict[int, float] = {}
        dom = int(pair_bucket[int(np.argmax(p))])
        out[dom] = 1.0
        return out
    p_hat = np.abs(dd_pred) / pred_change
    terms = 0.5 * np.abs(p_hat - p)
    out = {}
    for b in np.unique(np.asarray(pair_bucket)):
        out[int(b)] = float(terms[np.asarray(pair_bucket) == b].sum())
    return out


def _sign_correct(dd_true: float, dd_pred: float) -> bool:
    """G5.3 main-denominator correctness: matching sign AND |pred| >= 0.005.

    Abstention (|Δd_pred| < 0.005) is NEVER free (evaluate_g53 semantics).
    """
    return (np.sign(dd_pred) == np.sign(dd_true)
            and abs(dd_pred) >= SIGN_THRESHOLD)


def bucketize(rows: Sequence[dict]) -> dict:
    """Aggregate the per-pair rows into the D2 bucket table + matrices."""
    per_bucket = {
        b: {"n_pairs": 0, "n_sign_correct": 0, "n_abstained": 0,
            "n_predicted": 0, "n_predicted_sign_correct": 0,
            "abs_dd_true_sum": 0.0, "abs_dd_pred_sum": 0.0,
            "n_forming": 0, "n_breaking": 0,
            "n_flips_forming": 0, "n_flips_breaking": 0}
        for b in range(4)}
    cluster_bucket = {}
    frame_buckets: dict[int, list] = {}
    for row in rows:
        b = bucket_of(abs(row["dd_true"]))
        acc = per_bucket[b]
        acc["n_pairs"] += 1
        abst = abs(row["dd_pred"]) < SIGN_THRESHOLD
        acc["n_abstained"] += int(abst)
        if not abst:
            acc["n_predicted"] += 1
            ok = np.sign(row["dd_pred"]) == np.sign(row["dd_true"])
            acc["n_predicted_sign_correct"] += int(ok)
        ok_main = _sign_correct(row["dd_true"], row["dd_pred"])
        acc["n_sign_correct"] += int(ok_main)
        acc["abs_dd_true_sum"] += abs(row["dd_true"])
        acc["abs_dd_pred_sum"] += abs(row["dd_pred"])
        # forming (ΔB>0) vs breaking (ΔB<0): predicted direction vs the
        # chemical direction implied by the edit sign
        if row["pair_edit"] > 0:
            acc["n_forming"] += 1
            if not abst and np.sign(row["dd_pred"]) > 0:
                acc["n_flips_forming"] += 1  # forming bond predicted LONGER
        elif row["pair_edit"] < 0:
            acc["n_breaking"] += 1
            if not abst and np.sign(row["dd_pred"]) < 0:
                acc["n_flips_breaking"] += 1  # breaking bond predicted SHORTER
        cb = cluster_bucket.setdefault(row["group_id"], {
            str(b): {"n_pairs": 0, "n_sign_correct": 0, "n_abstained": 0}
            for b in range(4)})[str(b)]
        cb["n_pairs"] += 1
        cb["n_sign_correct"] += int(ok_main)
        cb["n_abstained"] += int(abst)
        frame_buckets.setdefault(row["sre_frame_id"], []).append(row)

    buckets_out = {}
    for b in range(4):
        acc = per_bucket[b]
        n = acc["n_pairs"]
        sign_acc = (acc["n_sign_correct"] / n) if n else None
        acc_out = {
            "label": BUCKET_LABELS[b],
            "n_pairs": n,
            "n_sign_correct": acc["n_sign_correct"],
            "n_abstained": acc["n_abstained"],
            "n_predicted": acc["n_predicted"],
            "sign_accuracy": sign_acc,
            "abstention_rate": (acc["n_abstained"] / n) if n else None,
            "sign_accuracy_where_predicted": (
                acc["n_predicted_sign_correct"] / acc["n_predicted"]
                if acc["n_predicted"] else None),
            "amplitude_ratio_mean": (
                acc["abs_dd_pred_sum"] / acc["abs_dd_true_sum"]
                if acc["abs_dd_true_sum"] > 0 else None),
            "n_forming": acc["n_forming"],
            "n_breaking": acc["n_breaking"],
            "sign_flips_forming": acc["n_flips_forming"],
            "sign_flips_breaking": acc["n_flips_breaking"],
        }
        buckets_out[str(b)] = acc_out

    # SRE: full-frame G5.3 SRE + per-bucket contribution share
    sre_all: list[float] = []
    sre_bucket_terms: dict[str, list[float]] = {str(b): [] for b in range(4)}
    sre_bucket_share: dict[str, list[float]] = {str(b): [] for b in range(4)}
    for _fid, frame_rows in sorted(frame_buckets.items()):
        ddt = np.asarray([r["dd_true"] for r in frame_rows])
        ddp = np.asarray([r["dd_pred"] for r in frame_rows])
        pb = np.asarray([bucket_of(abs(r["dd_true"])) for r in frame_rows])
        contrib = sre_frame_contribution(ddt, ddp, pb)
        if contrib is None:
            continue
        sre_all.append(sum(contrib.values()))
        frame_sre = max(sum(contrib.values()), 1e-12)
        for b in range(4):
            term = contrib.get(b)
            if term is None:
                sre_bucket_terms[str(b)].append(0.0)
                sre_bucket_share[str(b)].append(0.0)
            else:
                sre_bucket_terms[str(b)].append(term)
                sre_bucket_share[str(b)].append(term / frame_sre)
    sre = {
        "n_frames": len(sre_all),
        "sre_median": (float(np.median(sre_all)) if sre_all else None),
        "sre_q75": (float(np.quantile(sre_all, 0.75)) if sre_all else None),
        "per_bucket_mean_contribution": {
            b: (float(np.mean(v)) if v else None)
            for b, v in sre_bucket_terms.items()},
        "per_bucket_mean_share_of_frame_sre": {
            b: (float(np.mean(v)) if v else None)
            for b, v in sre_bucket_share.items()},
        "definition": ("G5.3 SRE on frames with >=2 pairs and total "
                       "teacher |Δd| >= 0.01 A; bucket contribution = the "
                       "0.5*|p_hat-p| terms of that bucket's pairs"),
    }
    return {"buckets": buckets_out, "sre": sre,
            "per_cluster_bucket": cluster_bucket}


def matrix_summary(cluster_bucket: dict) -> dict:
    """Cross-cluster medians of per-bucket sign accuracy (cluster x bucket)."""
    out = {}
    for b in range(4):
        key = str(b)
        accs = [v[key]["n_sign_correct"] / v[key]["n_pairs"]
                for v in cluster_bucket.values() if v[key]["n_pairs"] > 0]
        out[key] = {
            "label": BUCKET_LABELS[b],
            "n_clusters_with_pairs": len(accs),
            "sign_accuracy_cluster_median": (
                float(np.median(accs)) if accs else None),
            "sign_accuracy_cluster_q25": (
                float(np.quantile(accs, 0.25)) if accs else None),
        }
    return out


def make_verdict(buckets: dict) -> dict:
    """Plan §3.1 D2 decision rule (strong vs weak bucket sign accuracy)."""
    strong = buckets["3"]["sign_accuracy"]
    weak = buckets["1"]["sign_accuracy"]
    medium = buckets["2"]["sign_accuracy"]
    dead = buckets["0"]
    verdict = bucket_verdict(strong, weak)
    return {
        "verdict": verdict,
        "strong_bucket_sign_accuracy": strong,
        "weak_bucket_sign_accuracy": weak,
        "medium_bucket_sign_accuracy": medium,
        "dead_zone_abstention_rate": dead["abstention_rate"],
        "dead_zone_sign_accuracy_where_predicted":
            dead["sign_accuracy_where_predicted"],
        "decision_rule": (
            "strong>0.9 & weak<0.8 -> dead-zone/amplitude problem, pair "
            "margin loss first; strong<0.8 -> representation/feature/label "
            "problem, loss alone insufficient (plan §3.1 D2)"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root)
    split_path = Path(args.split_manifest)
    fingerprint = oof_fingerprint(runs_root, split_path, D0)
    feature_root = PROJECT_ROOT / "outputs/pesn/v5/features" / "d0_0.05"
    fingerprint["feature_shards_digest"] = manifest_digest(
        sorted(feature_root.glob("*.npz")))

    records = collect_oof_records(
        runs_root=runs_root, split_path=split_path, d0=D0,
        device_name=args.device, batch_size=args.batch_size,
        cache_path=(None if args.no_cache else Path(args.cache)),
        use_cache=not args.no_cache)
    rows = pair_rows(records)
    buckets = bucketize(rows)
    verdict = make_verdict(buckets["buckets"])
    matrix = matrix_summary(buckets["per_cluster_bucket"])

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "d0": D0,
        "n_samples": len(records),
        "n_pairs": len(rows),
        "thresholds": {
            "sign_pair_threshold_angstrom": SIGN_THRESHOLD,
            "bucket_edges_angstrom": [0.005, 0.01, 0.02],
            "bucket_edge_policy": "half-open [lo, hi): a pair EXACTLY on an "
                                  "edge belongs to the stronger bucket "
                                  "(|dd|=0.005 is ACTIVE)",
            "abstention_policy": "predicted |Δd|<0.005 counts incorrect in "
                                 "the main sign denominator (G5.3)",
            "sre_min_total_change_angstrom": SRE_MIN_TOTAL_CHANGE,
        },
        "buckets": buckets["buckets"],
        "sre": buckets["sre"],
        "per_cluster_bucket": buckets["per_cluster_bucket"],
        "cluster_bucket_matrix_summary": matrix,
        "verdict": verdict,
        "fingerprint": fingerprint,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"[g53_buckets] wrote {out}  ({len(rows)} pairs / "
          f"{len(records)} samples)")
    for b in range(4):
        e = buckets["buckets"][str(b)]
        print(f"  {e['label']:20s} n={e['n_pairs']:6d} "
              f"sign={_fmt(e['sign_accuracy'])} "
              f"abstain={_fmt(e['abstention_rate'])} "
              f"amp={_fmt(e['amplitude_ratio_mean'])}")
    print(f"  verdict: {verdict['verdict']}")
    return 0


def _fmt(v) -> str:
    return "  n/a " if v is None else f"{v:6.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
