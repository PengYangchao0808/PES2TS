#!/usr/bin/env python3
"""R1 preregistered cross-fit calibration study (campaign V5_1A_R1_REPAIR).

Reads the SEVEN landed B0 V5.1a runs
(``B0_f0_s{17,29,43}_v51``, ``B0_f{1..4}_s17_v51``) under
``outputs/pesn/v5_1/runs``, rebuilds each run's FROZEN 8-cluster
inner-val pool (config ``config/campaigns/v5_1_sprint.yaml``, split
seed 20261019), re-derives the raw predictions from
``checkpoint_best.pt`` (CPU by default), and performs the protocol-
preregistered 2-half cross-fit post-hoc global-scalar calibration
(``outputs/pesn/v5_1_r1/protocol_r1.json`` §calibration /
§partitions.calibration_crossfit):

* half A = first 4 inner-val clusters sorted by cluster id, half B = last 4;
* s* = Σ⟨u,y⟩ / Σ‖u‖² (clipped ≥ 0) fitted on ONE half, scored on the
  OTHER half only; pooled cross-fit score aggregates both directions;
  a scale is never scored on its own fitting clusters;
* primary weights w=1 on all key atoms; effective-atom weighting
  (‖y‖ > 0.02 Å) reported as secondary diagnostic;
* s = 1/alpha_median is computed for COMPARISON ONLY (protocol
  §calibration.forbidden — alpha is a norm-ratio median, not the MSE
  minimizer);
* cluster-level paired bootstrap (2000 draws, seed 20261018) on the
  cross-fit scores for RMSE/zero and SRE deltas (calibrated − raw).

Development evidence only: inner-val pools only, NO outer-dev usage, NO
formal OOF claims, NO sealed data.  Outputs (overwritten atomically per
run of this script):

* ``outputs/pesn/v5_1_r1/calibration/calibration_results.json``
  (schema ``pesn_v5_1_r1_calibration_v1``);
* ``outputs/pesn/v5_1_r1/calibration/calibration_report.md`` (draft).

Usage::

    /root/miniforge3/bin/python3 scripts/campaigns/r1_calibration.py \
        --device cpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION: Final[str] = "pesn_v5_1_r1_calibration_v1"
PROTOCOL_SCHEMA: Final[str] = "pesn_v5_1_r1_protocol_v1"

DEFAULT_RUNS: Final[tuple[str, ...]] = (
    "B0_f0_s17_v51", "B0_f0_s29_v51", "B0_f0_s43_v51",
    "B0_f1_s17_v51", "B0_f2_s17_v51", "B0_f3_s17_v51", "B0_f4_s17_v51",
)
DEFAULT_CONFIG = PROJECT_ROOT / "config/campaigns/v5_1_sprint.yaml"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "outputs/pesn/v5_1/runs"
DEFAULT_PROTOCOL = PROJECT_ROOT / "outputs/pesn/v5_1_r1/protocol_r1.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/pesn/v5_1_r1/calibration"

#: Metrics reported per run/half (protocol §calibration.reported_metrics).
REPORTED_METRICS: Final[tuple[str, ...]] = (
    "rmse_over_zero", "alpha_median", "sre_cluster_median",
    "strong_sign_accuracy", "dead_zone_fraction")


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _metric_block(records) -> dict:
    """The five protocol-reported metrics for a record set."""
    from pes2ts_core.pesn.offline.evaluation import r1_eval

    agg = r1_eval.canonical_metrics(records)["aggregate"]
    return {
        "rmse_over_zero": agg.get("rmse_over_zero"),
        "alpha_median": agg.get("alpha_median"),
        "sre_cluster_median": agg.get("sre_cluster_median"),
        "strong_sign_accuracy": r1_eval.strong_sign_accuracy(records),
        "dead_zone_fraction": agg.get("dead_zone_fraction"),
    }


def _delta(cal: dict, raw: dict) -> dict:
    out = {}
    for k in REPORTED_METRICS:
        a, b = cal.get(k), raw.get(k)
        out[k] = (float(a - b)
                  if a is not None and b is not None else None)
    return out


def _direction(delta: float | None, *, higher_is_better: bool,
               neutral_atol: float = 1e-4) -> str:
    if delta is None:
        return "n/a"
    if abs(delta) <= neutral_atol:
        return "neutral"
    better = delta > 0 if higher_is_better else delta < 0
    return "helped" if better else "hurt"


def _run_fold_seed(run_id: str) -> tuple[int, int]:
    # B0_f{fold}_s{seed}_v51
    parts = run_id.split("_")
    fold = int(parts[1][1:])
    seed = int(parts[2][1:])
    return fold, seed


# ---------------------------------------------------------------------------
# core study
# ---------------------------------------------------------------------------

def analyze_run(run_id: str, *, config, config_path: Path, runs_root: Path,
                device: str, draws: int, seed: int) -> dict:
    import torch

    from pes2ts_core.pesn.models.key_atom_egnn import StarEGNN
    from pes2ts_core.pesn.offline.evaluation import r1_eval
    from pes2ts_core.pesn.offline.training.key_atom_bc import load_lane_b_pools

    run_dir = runs_root / run_id
    ckpt_path = run_dir / "checkpoints/checkpoint_best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"{run_id}: missing {ckpt_path}")
    run_record = json.loads(
        (run_dir / "run_record.json").read_text(encoding="utf-8"))
    fold, run_seed = _run_fold_seed(run_id)
    if int(run_record.get("fold", fold)) != fold:
        raise RuntimeError(f"{run_id}: fold mismatch vs run_record")
    d0 = float((run_record.get("arm") or {}).get("d0", 0.05))
    split_path = str(PROJECT_ROOT
                     / str((config.get("paths") or {}).get(
                         "split_manifest", "data/manifests/v5/split_manifest.json")))
    v51_inner = (config.get("v51") or {}).get("inner_val") or {}
    inner_val = {"n_val": int(v51_inner.get("n_val_clusters", 8)),
                 "seed": int(v51_inner.get("seed", 20261019))}

    pools = load_lane_b_pools(config, fold=fold, d0=d0,
                              split_path=split_path, inner_val=inner_val)
    pool_val_ids = list(pools["inner_split"]["val_ids"])
    record_val_ids = list((run_record.get("inner_val_split") or {}).get(
        "val_ids") or [])
    if record_val_ids and pool_val_ids != record_val_ids:
        raise RuntimeError(
            f"{run_id}: rebuilt inner-val split {pool_val_ids} != "
            f"run_record {record_val_ids}")

    doc = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if str(doc.get("schema_version")) != "pesn_key_atom_bc_checkpoint_v1":
        raise RuntimeError(f"{run_id}: unexpected checkpoint schema")
    model = StarEGNN().to(torch.float32).to(torch.device(device))
    model.load_state_dict(doc["model"])
    records = r1_eval.predict_pool(
        model, list(pools["inner_val_dense"]), torch.device(device),
        config=config, fold=fold, d0=d0, split_path=split_path)

    clusters = sorted({str(r["cluster"]) for r in records})
    if clusters != sorted(pool_val_ids):
        raise RuntimeError(
            f"{run_id}: predicted clusters {clusters} != val ids "
            f"{sorted(pool_val_ids)}")
    half_a_ids, half_b_ids = clusters[:4], clusters[4:]
    set_a, set_b = set(half_a_ids), set(half_b_ids)
    rec_a = [r for r in records if str(r["cluster"]) in set_a]
    rec_b = [r for r in records if str(r["cluster"]) in set_b]

    # pooled raw baseline (per-half raw is computed inside _fit_score)
    raw_pooled = _metric_block(records)

    def raw_of(score_records):
        return _metric_block(score_records)

    def _fit_score(fit_records, score_records) -> dict:
        s_all = r1_eval.scale_solution(fit_records, weight="all")
        s_eff = r1_eval.scale_solution(fit_records, weight="effective")
        diag = r1_eval.scale_diagnostics(fit_records)
        s_alpha = diag["one_over_alpha_median"]  # COMPARISON ONLY
        cal = r1_eval.apply_scale(score_records, s_all)
        cal_alpha = (r1_eval.apply_scale(score_records, s_alpha)
                     if s_alpha is not None else None)
        return {
            "fit_clusters": sorted({str(r["cluster"]) for r in fit_records}),
            "score_clusters": sorted({str(r["cluster"]) for r in score_records}),
            "s_star_all": s_all,                       # PRIMARY
            "s_star_effective": s_eff,                 # secondary diagnostic
            "one_over_alpha_median_diagnostic_only": s_alpha,
            "alpha_median_on_fit_half": diag["alpha_median"],
            "raw": raw_of(score_records),
            "calibrated": _metric_block(cal),
            "calibrated_by_one_over_alpha_diagnostic_only": (
                _metric_block(cal_alpha) if cal_alpha is not None else None),
        }

    dir_ab = _fit_score(rec_a, rec_b)   # fit A → score B
    dir_ba = _fit_score(rec_b, rec_a)   # fit B → score A

    # pooled cross-fit: each half scored with the scale fitted on the
    # OTHER half (no scale ever scored on its own fitting clusters)
    cal_pooled_records = (
        r1_eval.apply_scale(rec_b, dir_ab["s_star_all"])
        + r1_eval.apply_scale(rec_a, dir_ba["s_star_all"]))
    cal_pooled = _metric_block(cal_pooled_records)
    pooled_delta = _delta(cal_pooled, raw_pooled)

    # cluster-level paired bootstrap on cross-fit scores (cal − raw)
    def _rmse_metric(sub):
        return r1_eval.canonical_metrics(sub)["aggregate"]["rmse_over_zero"]

    def _sre_metric(sub):
        return r1_eval.canonical_metrics(sub)["aggregate"]["sre_cluster_median"]

    boot_rmse = r1_eval.cluster_paired_bootstrap(
        cal_pooled_records, records, _rmse_metric, draws=draws, seed=seed)
    boot_sre = r1_eval.cluster_paired_bootstrap(
        cal_pooled_records, records, _sre_metric, draws=draws, seed=seed)

    # MSE-optimal s* on the FULL inner-val pool (the fitted value itself;
    # never scored in-sample — cross-fit above is the scored estimate)
    s_full = r1_eval.scale_solution(records, weight="all")
    s_full_eff = r1_eval.scale_solution(records, weight="effective")

    return {
        "run_id": run_id,
        "fold": fold,
        "seed": run_seed,
        "best_update": run_record.get("best_update"),
        "status": run_record.get("status"),
        "fingerprints": {
            "checkpoint_sha256": sha256_file(ckpt_path),
            "config_sha256": sha256_file(config_path),
            "split_manifest_sha256": sha256_file(split_path),
            "inner_val_split_hash": (run_record.get("inner_val_split_hash")
                                     or pools["inner_split"].get("hash")),
        },
        "inner_val_cluster_ids": clusters,
        "half_a_cluster_ids": half_a_ids,
        "half_b_cluster_ids": half_b_ids,
        "n_records": len(records),
        "n_records_half_a": len(rec_a),
        "n_records_half_b": len(rec_b),
        "directions": {"fit_A__score_B": dir_ab, "fit_B__score_A": dir_ba},
        "per_fold_scales": {
            "s_fit_A": dir_ab["s_star_all"],
            "s_fit_B": dir_ba["s_star_all"],
            "s_fit_A_effective": dir_ab["s_star_effective"],
            "s_fit_B_effective": dir_ba["s_star_effective"],
            "one_over_alpha_fit_A_diagnostic_only":
                dir_ab["one_over_alpha_median_diagnostic_only"],
            "one_over_alpha_fit_B_diagnostic_only":
                dir_ba["one_over_alpha_median_diagnostic_only"],
        },
        "s_star_full_pool_mse_optimal_diagnostic_not_crossfit": {
            "s_star_all": s_full, "s_star_effective": s_full_eff},
        "pooled_crossfit": {
            "raw": raw_pooled,
            "calibrated": cal_pooled,
            "delta_cal_minus_raw": pooled_delta,
            "direction": {
                "rmse_over_zero": _direction(
                    pooled_delta["rmse_over_zero"], higher_is_better=False),
                "alpha_median": _direction(
                    (None if raw_pooled["alpha_median"] in (None, 0)
                     else abs(float(cal_pooled["alpha_median"]) - 1.0)
                     - abs(float(raw_pooled["alpha_median"]) - 1.0)),
                    higher_is_better=False, neutral_atol=0.01),
                "sre_cluster_median": _direction(
                    pooled_delta["sre_cluster_median"], higher_is_better=False),
                "strong_sign_accuracy": _direction(
                    pooled_delta["strong_sign_accuracy"],
                    higher_is_better=True),
                "dead_zone_fraction": _direction(
                    pooled_delta["dead_zone_fraction"],
                    higher_is_better=False),
            },
        },
        "bootstrap_crossfit_cal_minus_raw": {
            "rmse_over_zero": boot_rmse,
            "sre_cluster_median": boot_sre,
        },
    }


def _spread(values: Sequence[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {"n": int(arr.size), "min": float(arr.min()),
            "median": float(np.median(arr)), "max": float(arr.max()),
            "mean": float(arr.mean())}


def build_results(run_results: list[dict], protocol: dict, *,
                  device: str, draws: int, seed: int) -> dict:
    import platform

    import numpy
    import torch

    s_fitted = [r["per_fold_scales"]["s_fit_A"] for r in run_results] + \
        [r["per_fold_scales"]["s_fit_B"] for r in run_results]
    per_run_mean_s = [
        float(np.mean([r["per_fold_scales"]["s_fit_A"],
                       r["per_fold_scales"]["s_fit_B"]]))
        for r in run_results]
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "V5_1A_R1_REPAIR",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_evidence_only",
        "disclaimer": (
            "R1 cross-fit calibration over INNER-VAL pools only; no formal "
            "OOF claims, no outer-dev usage, no sealed data. A scale is "
            "fitted on one half of the 8 inner-val clusters and scored only "
            "on the other half."),
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "device": device,
        },
        "protocol_excerpts": {
            "calibration": protocol.get("calibration"),
            "partitions_calibration_crossfit":
                (protocol.get("partitions") or {}).get("calibration_crossfit"),
            "pair_buckets_angstrom": protocol.get("pair_buckets_angstrom"),
        },
        "constants": {
            "effective_atom_tau_angstrom": r1_eval_const("EFFECTIVE_ATOM_TAU"),
            "sign_pair_threshold_angstrom": r1_eval_const("SIGN_PAIR_THRESHOLD"),
            "sre_min_total_change_angstrom": r1_eval_const("SRE_MIN_TOTAL_CHANGE"),
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
        },
        "runs": run_results,
        "spread_across_folds_seeds": {
            "all_half_fitted_s_star": _spread(s_fitted),
            "per_run_mean_s_star": _spread(per_run_mean_s),
        },
    }


def r1_eval_const(name: str) -> float:
    from pes2ts_core.pesn.offline.evaluation import r1_eval
    return getattr(r1_eval, name)


# ---------------------------------------------------------------------------
# report (draft, neutral)
# ---------------------------------------------------------------------------

def _fmt(v, spec: str = ".4f") -> str:
    if v is None:
        return "n/a"
    return format(float(v), spec)


def write_report(results: dict, out_path: Path) -> None:
    runs = results["runs"]
    lines: list[str] = []
    add = lines.append
    add("# R1 post-hoc global-scalar calibration — DRAFT report")
    add("")
    add(f"Generated {results['created_at']} (schema "
        f"`{results['schema_version']}`, device={results['environment']['device']}, "
        f"numpy {results['environment']['numpy']}, "
        f"torch {results['environment']['torch']}, "
        f"python {results['environment']['python']}).")
    add("")
    add(results["disclaimer"])
    add("")
    add("Design (preregistered, protocol_r1.json §calibration / "
        "§partitions.calibration_crossfit): per run, the 8 frozen "
        "inner-val clusters are split by sorted cluster id into halves "
        "A (first 4) and B (last 4); s* = Σ⟨u,y⟩/Σ‖u‖² is fitted on one "
        "half and scored ONLY on the other; pooled cross-fit aggregates "
        "both directions. Primary weights: w=1 on all key atoms. "
        "`1/alpha_median` is shown for comparison only and is NOT the "
        "fitted scale (protocol §calibration.forbidden).")
    add("")

    # --- per-run fitted scales ---
    add("## Per-run fitted scales (primary: all-atom s*)")
    add("")
    add("| run | fold | seed | s* fit A | s* fit B | s* eff A | s* eff B "
        "| 1/α fit A (diag) | 1/α fit B (diag) | s* full pool (diag, not cross-fit) |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        ps = r["per_fold_scales"]
        add(f"| {r['run_id']} | {r['fold']} | {r['seed']} "
            f"| {_fmt(ps['s_fit_A'])} | {_fmt(ps['s_fit_B'])} "
            f"| {_fmt(ps['s_fit_A_effective'])} | {_fmt(ps['s_fit_B_effective'])} "
            f"| {_fmt(ps['one_over_alpha_fit_A_diagnostic_only'])} "
            f"| {_fmt(ps['one_over_alpha_fit_B_diagnostic_only'])} "
            f"| {_fmt(r['s_star_full_pool_mse_optimal_diagnostic_not_crossfit']['s_star_all'])} |")
    add("")
    spread = results["spread_across_folds_seeds"]
    afs = spread["all_half_fitted_s_star"]
    add(f"Scale stability across all {afs.get('n', 0)} half-fitted scales: "
        f"min {_fmt(afs.get('min'))}, median {_fmt(afs.get('median'))}, "
        f"max {_fmt(afs.get('max'))}. Per-run mean s*: "
        f"median {_fmt(spread['per_run_mean_s_star'].get('median'))} "
        f"(min {_fmt(spread['per_run_mean_s_star'].get('min'))}, "
        f"max {_fmt(spread['per_run_mean_s_star'].get('max'))}).")
    add("")

    # --- per run/half raw vs calibrated ---
    add("## Per run / half: raw vs calibrated (cross-fit: scale from the other half)")
    add("")
    for r in runs:
        add(f"### {r['run_id']} (best_update={r['best_update']})")
        add("")
        add("| half (scored) | metric | raw | calibrated (s from other half) | Δ (cal−raw) | 1/α diag cal |")
        add("|---|---|---|---|---|---|")
        for tag, key in (("B", "fit_A__score_B"), ("A", "fit_B__score_A")):
            d = r["directions"][key]
            for m in REPORTED_METRICS:
                raw_v = d["raw"].get(m)
                cal_v = d["calibrated"].get(m)
                dl = (cal_v - raw_v
                      if raw_v is not None and cal_v is not None else None)
                alp = (d["calibrated_by_one_over_alpha_diagnostic_only"] or {}
                       ).get(m)
                add(f"| {tag} | {m} | {_fmt(raw_v)} | {_fmt(cal_v)} "
                    f"| {_fmt(dl, '+.4f')} | {_fmt(alp)} |")
        add("")

    # --- pooled cross-fit + bootstrap ---
    add("## Pooled cross-fit (both directions) + cluster paired bootstrap")
    add("")
    add("| run | metric | raw | calibrated | Δ (cal−raw) | bootstrap median Δ "
        "| 95% CI [2.5, 97.5] | direction |")
    add("|---|---|---|---|---|---|---|---|")
    for r in runs:
        pc = r["pooled_crossfit"]
        boot = r["bootstrap_crossfit_cal_minus_raw"]
        for m, bkey in (("rmse_over_zero", "rmse_over_zero"),
                        ("sre_cluster_median", "sre_cluster_median")):
            b = boot[bkey]
            add(f"| {r['run_id']} | {m} | {_fmt(pc['raw'].get(m))} "
                f"| {_fmt(pc['calibrated'].get(m))} "
                f"| {_fmt(pc['delta_cal_minus_raw'].get(m), '+.4f')} "
                f"| {_fmt(b['median_delta'], '+.4f')} "
                f"| [{_fmt(b['ci_lo_2_5'], '+.4f')}, {_fmt(b['ci_hi_97_5'], '+.4f')}] "
                f"| {pc['direction'].get(m)} |")
        for m in ("alpha_median", "strong_sign_accuracy", "dead_zone_fraction"):
            add(f"| {r['run_id']} | {m} | {_fmt(pc['raw'].get(m))} "
                f"| {_fmt(pc['calibrated'].get(m))} "
                f"| {_fmt(pc['delta_cal_minus_raw'].get(m), '+.4f')} "
                f"| n/a | n/a | {pc['direction'].get(m)} |")
    add("")

    # --- neutral findings ---
    add("## Findings (neutral, development evidence only)")
    add("")
    counts = {m: {"helped": 0, "hurt": 0, "neutral": 0, "n/a": 0}
              for m in REPORTED_METRICS}
    for r in runs:
        for m, d in r["pooled_crossfit"]["direction"].items():
            if m in counts:
                counts[m][d] = counts[m].get(d, 0) + 1
    add("Observed direction of the cross-fit calibrated metric vs raw, "
        "counted over the 7 runs (helped/hurt/neutral; neutral = |Δ| "
        "≤ 1e-4 for RMSE/SRE/sign/dead-zone, |Δ| ≤ 0.01 for alpha; for "
        "alpha, helped means |α−1| decreased — a DIAGNOSTIC only per "
        "protocol §confirmation_admission_r1.alpha_role):")
    add("")
    add("| metric | helped | hurt | neutral | n/a |")
    add("|---|---|---|---|---|")
    for m in REPORTED_METRICS:
        c = counts[m]
        add(f"| {m} | {c['helped']} | {c['hurt']} | {c['neutral']} | {c['n/a']} |")
    add("")
    ci_covers_zero = sum(
        1 for r in runs
        for m in ("rmse_over_zero", "sre_cluster_median")
        if (r["bootstrap_crossfit_cal_minus_raw"][m]["ci_lo_2_5"] <= 0.0
            <= r["bootstrap_crossfit_cal_minus_raw"][m]["ci_hi_97_5"]))
    add(f"The cluster paired bootstrap 95% interval covers zero for "
        f"{ci_covers_zero} of {2 * len(runs)} run×metric comparisons "
        "(RMSE/zero and SRE).")
    add("")
    add("Interpretation constraints: (1) these are inner-val cross-fit "
        "numbers on 8 clusters per run — small-sample development "
        "evidence, NOT sealed or outer-dev performance; (2) a global "
        "positive scale leaves pair signs and the SRE ordering "
        "distribution nearly invariant by construction, so movement in "
        "those metrics comes only from the |Δd| sign-threshold and "
        "dead-zone boundaries; (3) alpha moving toward 1 is a "
        "diagnostic, never a selection criterion (protocol "
        "§confirmation_admission_r1.alpha_role); (4) no claim of "
        "generalization is made or implied — adopting or rejecting the "
        "A1 arm requires the separate R1 admission rules.")
    add("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r1_calibration.py",
        description="R1 preregistered cross-fit global-scalar calibration "
                    "over the seven landed B0 V5.1a runs (inner-val only).")
    p.add_argument("--device", default="cpu",
                   help="torch device (default cpu; CPU-only campaign)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    p.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--runs", nargs="*", default=None,
                   help="optional subset of run ids (default: the 7 frozen)")
    p.add_argument("--draws", type=int, default=2000,
                   help="bootstrap draws (protocol: 2000)")
    p.add_argument("--seed", type=int, default=20261018,
                   help="bootstrap seed (protocol: 20261018)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    from pes2ts_core.config_loader import load_config_mapping

    config_path = Path(args.config)
    runs_root = Path(args.runs_root)
    protocol_path = Path(args.protocol)
    out_dir = Path(args.out_dir)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if str(protocol.get("schema_version")) != PROTOCOL_SCHEMA:
        raise SystemExit(
            f"unexpected protocol schema {protocol.get('schema_version')!r} "
            f"(want {PROTOCOL_SCHEMA})")
    config = load_config_mapping(config_path)
    run_ids = tuple(args.runs) if args.runs else DEFAULT_RUNS
    missing = [r for r in run_ids
               if not (runs_root / r / "checkpoints/checkpoint_best.pt").is_file()]
    if missing:
        raise SystemExit(f"missing checkpoints for runs: {missing}")

    run_results = []
    for run_id in run_ids:
        print(f"[r1_calibration] {run_id}: predict inner-val + cross-fit ...",
              flush=True)
        res = analyze_run(
            run_id, config=config, config_path=config_path,
            runs_root=runs_root, device=args.device,
            draws=int(args.draws), seed=int(args.seed))
        pc = res["pooled_crossfit"]
        print(f"  s*A={res['per_fold_scales']['s_fit_A']:.4f} "
              f"s*B={res['per_fold_scales']['s_fit_B']:.4f} "
              f"Δrmse={pc['delta_cal_minus_raw']['rmse_over_zero']:+.4f} "
              f"Δsre={pc['delta_cal_minus_raw']['sre_cluster_median']:+.4f}",
              flush=True)
        run_results.append(res)

    results = build_results(run_results, protocol, device=args.device,
                            draws=int(args.draws), seed=int(args.seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "calibration_results.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    md_path = out_dir / "calibration_report.md"
    write_report(results, md_path)
    # validity re-check (self-gate)
    json.loads(json_path.read_text(encoding="utf-8"))
    print(f"[r1_calibration] wrote {json_path}")
    print(f"[r1_calibration] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
