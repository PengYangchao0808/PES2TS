#!/usr/bin/env python3
"""E7 statistics/reporting toolkit for the 30h campaign (lane W-DATA).

Analysis only: reads run artifacts (read-only), writes deterministic
JSON+Markdown reports under ``<campaign_root>/stats/`` (supervisor
sanctioned artifact dir — nothing else is ever written).

Implements plan §7 (E7 统计复核: cluster 置信区间、错误簇卡片、成本 ledger、门报告)
and §4.3 metric rules:

* Aggregate unit is the **independent curve/cluster**, never frames.
  Each run's ``key_rmse_over_zero`` is already a cluster-median over its
  validation clusters; its ``key_rmse_over_zero_bootstrap`` (unit=
  ``reaction_cluster``) is REUSED verbatim as the per-run cluster-level CI.
  Arm-level CIs are a seeded bootstrap over runs (fold×seed evaluations);
  this unit is labelled explicitly in every output.  ``n_clusters`` /
  unique val-curve counts are reported everywhere.
* Sealed-fold discipline: folds 3/4 are EXCLUDED from every dev
  aggregate; any encountered sealed run is listed and flagged.
* Error cluster cards: worst-N dev evaluations per arm with their val
  curve identifiers (the cluster units) + run diagnostics.  metrics.json
  carries no per-cluster error decomposition, so cards attribute the
  run's cluster-median to its whole val curve set and say so.
* Cost accounting from ``run_records.jsonl`` (latest record per run_id):
  attempts, retries, wall-clock (metrics wall_clock_s preferred,
  ledger span fallback), blocked list with reasons, GPU-hours estimate
  from cuda-device wall-clock.
* Gate snapshot from ``docs/agent_runs/state/gates/*.json``.

Outputs are deterministic for fixed inputs + seed: sorted keys, sorted
lists, no wall-clock timestamps (an input fingerprint over artifact
hashes is embedded instead, so incremental run arrival is detectable).

Usage::

    python scripts/agent_ops/data_gate/r3_stats.py \\
        --campaign-root outputs/pesn/30h_campaign \\
        --gates-dir docs/agent_runs/state/gates \\
        [--label v1] [--worst-n 5] [--bootstrap-draws 2000] \\
        [--bootstrap-seed 20261021] [--no-write]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA = "pes2ts_r3_stats_report_v1"
CAMPAIGN_ID = "PES2TS_30H_R3_V52_V53"
DEV_FOLDS = (0, 1, 2)
SEALED_FOLDS = (3, 4)
DEFAULT_BOOTSTRAP_SEED = 20261021
DEFAULT_BOOTSTRAP_DRAWS = 2000
DEFAULT_WORST_N = 5
FLOAT_ROUND = 9

MAIN_TREE_DEFAULT = Path("/mnt/e/Calculations/AI4S_ML_Studys/PES2TS")
_REL_CAMPAIGN = Path("outputs/pesn/30h_campaign")


def resolve_campaign_root(explicit: str | None) -> Path:
    """Resolve the campaign root: explicit > env-free worktree > main tree."""
    import os

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("PES2TS_CAMPAIGN_ROOT")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parents[3]  # repo root of this script
    candidates.append(here / _REL_CAMPAIGN)
    candidates.append(MAIN_TREE_DEFAULT / _REL_CAMPAIGN)
    for c in candidates:
        if (c / "run_records.jsonl").is_file():
            return c
    raise SystemExit(
        "could not resolve campaign root (run_records.jsonl not found); "
        f"tried: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------- loading


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run_metrics(runs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Scan ``r3_runs/*/metrics.json``; tolerate partial/incremental state.

    Returns (runs, skipped, pending_run_ids):
      * runs    — one record per readable metrics.json (with ``run_id``).
      * skipped — unreadable/invalid files with reasons (never fatal).
      * pending — ledger run_ids without a metrics.json yet (incremental).
    """
    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return runs, skipped, []
    for md in sorted(runs_dir.glob("*/metrics.json")):
        run_id = md.parent.name
        try:
            data = json.loads(md.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            skipped.append({"run_id": run_id, "reason": f"unreadable_metrics: {exc}"})
            continue
        if not isinstance(data, dict) or "arm" not in data or "fold" not in data:
            skipped.append({"run_id": run_id, "reason": "metrics_missing_arm_or_fold"})
            continue
        if "key_rmse_over_zero" not in data:
            skipped.append({"run_id": run_id, "reason": "metrics_missing_key_rmse_over_zero"})
            continue
        data = dict(data)
        data["run_id"] = run_id
        data["_metrics_sha256"] = sha256_file(md)
        runs.append(data)
    return runs, skipped, []


def load_ledger(ledger_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read run_records.jsonl (read-only).

    Returns (final_records, raw_records): ``final_records`` is the latest
    record per run_id ordered by (updated_at, line order); malformed lines
    are tolerated and reported in ``raw_records`` under ``_parse_error``.
    """
    raw: list[dict[str, Any]] = []
    if ledger_path.is_file():
        with open(ledger_path) as f:
            for lineno, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # malformed/partial line: tolerated, absent from final
                if not isinstance(rec, dict) or "run_id" not in rec:
                    continue
                rec["_lineno"] = lineno
                raw.append(rec)
    latest: dict[str, dict[str, Any]] = {}
    for rec in raw:
        rid = rec["run_id"]
        prev = latest.get(rid)
        key = (rec.get("updated_at") or "", rec["_lineno"])
        if prev is None or key > (prev.get("updated_at") or "", prev["_lineno"]):
            latest[rid] = rec
    final = [latest[k] for k in sorted(latest)]
    # malformed/partial lines are tolerated (incremental ledger writes) and
    # simply absent from `final` — callers can detect via line counts.
    return final, raw


def load_gate_snapshot(gates_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not gates_dir.is_dir():
        return rows
    for gp in sorted(gates_dir.glob("*.json")):
        try:
            data = json.loads(gp.read_text())
        except (json.JSONDecodeError, OSError):
            rows.append({"gate": gp.stem, "status": "unreadable", "at": None})
            continue
        gate = data.get("gate_id") or data.get("gate") or gp.stem
        rows.append(
            {
                "gate": str(gate),
                "status": data.get("status", "unknown"),
                "at": data.get("evaluated_at"),
            }
        )
    rows.sort(key=lambda r: r["gate"])
    return rows


def load_curve_quality(data_dir: Path) -> dict[str, dict[str, Any]]:
    """curve_id -> {quality_status, quality_score} from data/curve_quality.csv."""
    out: dict[str, dict[str, Any]] = {}
    cq = data_dir / "curve_quality.csv"
    if not cq.is_file():
        return out
    with open(cq, newline="") as f:
        for row in csv.DictReader(f):
            cid = row.get("curve_id")
            if not cid:
                continue
            try:
                score = float(row["quality_score"]) if row.get("quality_score") else None
            except ValueError:
                score = None
            out[cid] = {"quality_status": row.get("quality_status"), "quality_score": score}
    return out


# ------------------------------------------------------------ statistics


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
            out.append(float(v))
    return out


def bootstrap_ci(
    values: Sequence[float],
    rng: random.Random,
    draws: int,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
) -> list[float] | None:
    """Percentile CI (2.5/97.5) of ``statistic`` via seeded resampling."""
    vals = list(values)
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        v = statistic(vals)
        return [v, v]
    samples = []
    for _ in range(draws):
        resample = [vals[rng.randrange(n)] for _ in range(n)]
        samples.append(statistic(resample))
    samples.sort()
    lo = samples[int(round(0.025 * (draws - 1)))]
    hi = samples[int(round(0.975 * (draws - 1)))]
    return [lo, hi]


def summarize(values: Sequence[float], rng: random.Random, draws: int) -> dict[str, Any]:
    """Mean/spread/CI summary for a list of run-level values."""
    vals = sorted(_finite(values))
    if not vals:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "ci95_mean": None,
        }
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "median": statistics.median(vals),
        "min": vals[0],
        "max": vals[-1],
        "ci95_mean": bootstrap_ci(vals, rng, draws),
    }


def run_metric_value(run: dict[str, Any], path: Sequence[str]) -> float | None:
    node: Any = run
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, (int, float)) else None


# ------------------------------------------------------------ aggregation


ARM_METRICS: dict[str, tuple[str, ...]] = {
    "rmse_over_zero": ("key_rmse_over_zero", "median_rmse_over_zero"),
    "mae_over_zero": ("key_rmse_over_zero", "median_mae_over_zero"),
    "cos_eff_median": ("key_rmse_over_zero", "cos_eff_median"),
    "sign_accuracy_cluster_median": ("sre_and_sign", "sign_accuracy_cluster_median"),
    "sre_cluster_median": ("sre_and_sign", "sre_cluster_median"),
    "dead_zone_fraction": ("sre_and_sign", "dead_zone_fraction"),
}


def aggregate_arms(
    dev_runs: list[dict[str, Any]],
    rng: random.Random,
    draws: int,
) -> dict[str, Any]:
    """Per-arm aggregate over dev runs (folds 0-2 only)."""
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for run in dev_runs:
        by_arm.setdefault(str(run["arm"]), []).append(run)

    arms: dict[str, Any] = {}
    for arm in sorted(by_arm):
        runs = sorted(by_arm[arm], key=lambda r: (int(r.get("fold", -1)), int(r.get("seed", -1)), r["run_id"]))
        entry: dict[str, Any] = {
            "n_runs": len(runs),
            "folds": sorted({int(r["fold"]) for r in runs}),
            "seeds": sorted({int(r.get("seed", -1)) for r in runs}),
            "run_ids": [r["run_id"] for r in runs],
        }
        # cluster accounting: every run value is a cluster-median over its val clusters
        val_curves: set[str] = set()
        n_cluster_evals = 0
        per_run_clusters = []
        for r in runs:
            curves = [c for c in (r.get("inner_val_curves") or []) if isinstance(c, str)]
            val_curves.update(curves)
            n = run_metric_value(r, ("key_rmse_over_zero", "n_clusters"))
            n_int = int(n) if isinstance(n, (int, float)) else len(curves)
            n_cluster_evals += n_int
            per_run_clusters.append({"run_id": r["run_id"], "n_clusters": n_int, "val_curves": sorted(curves)})
        entry["n_unique_val_curves"] = len(val_curves)
        entry["val_curves"] = sorted(val_curves)
        entry["cluster_evaluations_total"] = n_cluster_evals
        entry["per_run_clusters"] = per_run_clusters

        # arm-level metric summaries (seeded bootstrap over runs)
        metrics_out: dict[str, Any] = {}
        for name, path in ARM_METRICS.items():
            values = [run_metric_value(r, path) for r in runs]
            metrics_out[name] = summarize([v for v in values if v is not None], rng, draws)
            metrics_out[name]["values_by_run"] = {
                r["run_id"]: run_metric_value(r, path) for r in runs
            }
        # fraction better than zero (pooled cluster accounting, not frame-level)
        n_better = 0
        n_defined = 0
        for r in runs:
            nb = run_metric_value(r, ("key_rmse_over_zero", "n_better_than_zero"))
            nd = run_metric_value(r, ("key_rmse_over_zero", "n_clusters_rmse_over_zero_defined"))
            if nb is not None:
                n_better += int(nb)
            if nd is not None:
                n_defined += int(nd)
        metrics_out["fraction_better_than_zero"] = {
            "n_better": n_better,
            "n_defined": n_defined,
            "fraction": (n_better / n_defined) if n_defined else None,
        }
        entry["metrics"] = metrics_out

        # per-run cluster-level bootstrap CIs REUSED from metrics.json
        per_run_ci: dict[str, Any] = {}
        for r in runs:
            boot = r.get("key_rmse_over_zero_bootstrap")
            if isinstance(boot, dict):
                per_run_ci[r["run_id"]] = {
                    "ci95": boot.get("median_rmse_over_zero_ci95"),
                    "unit": boot.get("unit"),
                    "draws": boot.get("draws"),
                    "seed": boot.get("seed"),
                }
        entry["per_run_cluster_bootstrap_ci95"] = per_run_ci

        # seed stability: per fold, per seed values + ≥2/3-seeds-below-1.0 count
        per_fold: dict[str, dict[str, float]] = {}
        for r in runs:
            v = run_metric_value(r, ("key_rmse_over_zero", "median_rmse_over_zero"))
            if v is None:
                continue
            per_fold.setdefault(str(r["fold"]), {})[str(r.get("seed"))] = v
        folds_2of3 = 0
        for fold, seedmap in per_fold.items():
            vals = sorted(seedmap.values())
            if len(vals) >= 2 and sum(1 for v in vals if v < 1.0) >= 2:
                folds_2of3 += 1
        entry["seed_stability"] = {
            "rmse_over_zero_by_fold_seed": {
                f: {s: seedmap[s] for s in sorted(seedmap, key=int)}
                for f, seedmap in sorted(per_fold.items(), key=lambda kv: int(kv[0]))
            },
            "n_folds_with_2of3_seeds_below_1": folds_2of3,
        }
        arms[arm] = entry
    return arms


def error_cluster_cards(
    dev_runs: list[dict[str, Any]],
    worst_n: int,
    curve_quality: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Worst-N dev evaluations per arm, with cluster identifiers + diagnostics.

    metrics.json has no per-cluster error decomposition, so each card is a
    whole-run evaluation card: it names the val curves (the cluster units
    aggregated by that run's median) and carries run-level diagnostics.
    """
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for run in dev_runs:
        by_arm.setdefault(str(run["arm"]), []).append(run)

    cards: dict[str, list[dict[str, Any]]] = {}
    for arm in sorted(by_arm):
        ranked = sorted(
            by_arm[arm],
            key=lambda r: (
                -(run_metric_value(r, ("key_rmse_over_zero", "median_rmse_over_zero")) or -1.0),
                r["run_id"],
            ),
        )
        arm_cards = []
        for r in ranked[:worst_n]:
            curves = sorted(c for c in (r.get("inner_val_curves") or []) if isinstance(c, str))
            card = {
                "run_id": r["run_id"],
                "fold": r.get("fold"),
                "seed": r.get("seed"),
                "level": r.get("level"),
                "val_curves": curves,
                "n_clusters": run_metric_value(r, ("key_rmse_over_zero", "n_clusters")),
                "median_rmse_over_zero": run_metric_value(
                    r, ("key_rmse_over_zero", "median_rmse_over_zero")
                ),
                "q75_rmse_over_zero": run_metric_value(r, ("key_rmse_over_zero", "q75_rmse_over_zero")),
                "cluster_ci95": (r.get("key_rmse_over_zero_bootstrap") or {}).get(
                    "median_rmse_over_zero_ci95"
                )
                if isinstance(r.get("key_rmse_over_zero_bootstrap"), dict)
                else None,
                "sign_accuracy_cluster_median": run_metric_value(
                    r, ("sre_and_sign", "sign_accuracy_cluster_median")
                ),
                "sre_cluster_median": run_metric_value(r, ("sre_and_sign", "sre_cluster_median")),
                "dead_zone_fraction": run_metric_value(r, ("sre_and_sign", "dead_zone_fraction")),
                "cos_eff_median": run_metric_value(r, ("key_rmse_over_zero", "cos_eff_median")),
                "rmse_degradation_vs_zero_step_pct": (r.get("guards") or {}).get(
                    "rmse_degradation_vs_zero_step_pct"
                )
                if isinstance(r.get("guards"), dict)
                else None,
                "curve_quality": {c: curve_quality.get(c) for c in curves},
            }
            arm_cards.append(card)
        cards[arm] = arm_cards
    return {
        "worst_n": worst_n,
        "unit_note": (
            "Card = one dev run evaluation (fold×seed). metrics.json stores the "
            "cluster-median over the run's val clusters but no per-cluster "
            "decomposition; val_curves are the cluster units of that median. "
            "Cards never mix sealed folds (3/4)."
        ),
        "cards_by_arm": cards,
    }


# ------------------------------------------------------------- cost


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def cost_accounting(
    final_records: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attempts/retries/wall-clock/blocked/GPU-hours per (phase, arm)."""
    wall_by_run: dict[str, dict[str, Any]] = {}
    for r in runs:
        wall = run_metric_value(r, ("wall_clock_s",))
        wall_by_run[r["run_id"]] = {
            "wall_clock_s": wall,
            "device": r.get("device"),
        }

    # ledger spans: created_at(first) -> updated_at(latest) per run
    first_ts: dict[str, datetime] = {}
    last_ts: dict[str, datetime] = {}
    for rec in raw_records:
        rid = rec["run_id"]
        c = _parse_ts(rec.get("created_at")) or _parse_ts(rec.get("updated_at"))
        u = _parse_ts(rec.get("updated_at"))
        if c and (rid not in first_ts or c < first_ts[rid]):
            first_ts[rid] = c
        if u and (rid not in last_ts or u > last_ts[rid]):
            last_ts[rid] = u

    blocked: list[dict[str, Any]] = []
    retried: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    status_totals: dict[str, int] = {}

    for rec in final_records:
        rid = rec["run_id"]
        phase = str(rec.get("phase"))
        arm = str(rec.get("arm"))
        status = str(rec.get("status"))
        status_totals[status] = status_totals.get(status, 0) + 1
        key = (phase, arm)
        bucket = by_key.setdefault(
            key,
            {
                "phase": phase,
                "arm": arm,
                "n_runs": 0,
                "status_counts": {},
                "total_attempts": 0,
                "n_retried": 0,
                "wall_clock_s_metrics": 0.0,
                "wall_clock_s_metrics_n": 0,
                "wall_clock_s_ledger_span": 0.0,
                "wall_clock_s_ledger_span_n": 0,
                "gpu_wall_clock_s": 0.0,
                "cpu_wall_clock_s": 0.0,
                "n_gpu_runs": 0,
                "n_cpu_runs": 0,
            },
        )
        bucket["n_runs"] += 1
        bucket["status_counts"][status] = bucket["status_counts"].get(status, 0) + 1
        attempt = rec.get("attempt")
        attempt = int(attempt) if isinstance(attempt, (int, float)) else 1
        retry = rec.get("retry_count")
        retry = int(retry) if isinstance(retry, (int, float)) else 0
        bucket["total_attempts"] += attempt
        if attempt > 1 or retry > 0:
            bucket["n_retried"] += 1
            retried.append(
                {"run_id": rid, "phase": phase, "arm": arm, "attempt": attempt, "retry_count": retry}
            )

        w = wall_by_run.get(rid)
        if w and w.get("wall_clock_s") is not None:
            bucket["wall_clock_s_metrics"] += float(w["wall_clock_s"])
            bucket["wall_clock_s_metrics_n"] += 1
            if str(w.get("device", "")).startswith("cuda"):
                bucket["gpu_wall_clock_s"] += float(w["wall_clock_s"])
                bucket["n_gpu_runs"] += 1
            else:
                bucket["cpu_wall_clock_s"] += float(w["wall_clock_s"])
                bucket["n_cpu_runs"] += 1
        else:
            span = None
            if rid in first_ts and rid in last_ts:
                span = (last_ts[rid] - first_ts[rid]).total_seconds()
            if span is not None and span >= 0:
                bucket["wall_clock_s_ledger_span"] += span
                bucket["wall_clock_s_ledger_span_n"] += 1

        if rec.get("blocked_with_reason"):
            blocked.append(
                {
                    "run_id": rid,
                    "phase": phase,
                    "arm": arm,
                    "fold": rec.get("fold"),
                    "seed": rec.get("seed"),
                    "reason": rec.get("blocked_with_reason"),
                    "at": rec.get("blocked_at"),
                    "source": rec.get("blocked_source"),
                }
            )

    rows = []
    for key in sorted(by_key):
        b = by_key[key]
        b["gpu_hours_estimate"] = b["gpu_wall_clock_s"] / 3600.0
        b["total_wall_clock_s"] = b["wall_clock_s_metrics"] + b["wall_clock_s_ledger_span"]
        rows.append(b)

    blocked.sort(key=lambda x: (x["phase"], x["arm"], x["run_id"]))
    retried.sort(key=lambda x: (x["phase"], x["arm"], x["run_id"]))
    return {
        "ledger_note": (
            "latest record per run_id; wall-clock prefers metrics.wall_clock_s "
            "(train wall time, cuda counted as GPU-hours estimate) and falls "
            "back to ledger created_at->updated_at span for runs without "
            "metrics yet; orchestration/queue time is not included"
        ),
        "totals": {
            "n_ledger_lines": len(raw_records),
            "n_unique_runs": len(final_records),
            "status_counts": dict(sorted(status_totals.items())),
            "gpu_hours_estimate": sum(b["gpu_hours_estimate"] for b in rows),
            "wall_clock_s_total": sum(b["total_wall_clock_s"] for b in rows),
        },
        "by_phase_arm": rows,
        "blocked": blocked,
        "retried": retried,
    }


# ------------------------------------------------------------- report


def input_fingerprint(
    runs: list[dict[str, Any]],
    ledger_path: Path,
    gates_dir: Path,
) -> dict[str, Any]:
    lines = [f"{r['run_id']}:{r['_metrics_sha256']}" for r in sorted(runs, key=lambda r: r["run_id"])]
    h = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    gate_hashes = {}
    if gates_dir.is_dir():
        for gp in sorted(gates_dir.glob("*.json")):
            gate_hashes[gp.name] = sha256_file(gp)
    return {
        "n_metrics_files": len(runs),
        "metrics_set_sha256": h,
        "run_records_sha256": sha256_file(ledger_path) if ledger_path.is_file() else None,
        "gate_files_sha256": gate_hashes,
    }


def build_report(
    campaign_root: Path,
    gates_dir: Path,
    bootstrap_seed: int,
    bootstrap_draws: int,
    worst_n: int,
) -> dict[str, Any]:
    runs_dir = campaign_root / "r3_runs"
    ledger_path = campaign_root / "run_records.jsonl"

    runs, skipped_metrics, _ = load_run_metrics(runs_dir)
    final_records, raw_records = load_ledger(ledger_path)
    curve_quality = load_curve_quality(campaign_root / "data")

    ledger_run_ids = {r["run_id"] for r in final_records}
    metrics_run_ids = {r["run_id"] for r in runs}
    pending = sorted(ledger_run_ids - metrics_run_ids)

    sealed_runs = sorted(
        (r for r in runs if int(r.get("fold", -1)) in SEALED_FOLDS),
        key=lambda r: (int(r["fold"]), str(r.get("arm")), r["run_id"]),
    )
    dev_runs = [r for r in runs if int(r.get("fold", -1)) in DEV_FOLDS]
    other_fold_runs = [
        r for r in runs if int(r.get("fold", -1)) not in DEV_FOLDS and int(r.get("fold", -1)) not in SEALED_FOLDS
    ]

    rng = random.Random(bootstrap_seed)
    arms = aggregate_arms(dev_runs, rng, bootstrap_draws)
    cards = error_cluster_cards(dev_runs, worst_n, curve_quality)
    cost = cost_accounting(final_records, raw_records, runs)
    gates = load_gate_snapshot(gates_dir)

    report = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "aggregate_unit": "independent curve/cluster",
        "dev_folds": list(DEV_FOLDS),
        "sealed_folds": list(SEALED_FOLDS),
        "bootstrap": {
            "seed": bootstrap_seed,
            "draws": bootstrap_draws,
            "arm_ci_method": (
                "percentile bootstrap of the mean over run-level cluster-median "
                "values (resampling unit: run = fold×seed evaluation; each run "
                "value is itself a cluster-median over its val clusters, whose "
                "per-run cluster-level CI is reused under "
                "arms.<arm>.per_run_cluster_bootstrap_ci95)"
            ),
        },
        "input_fingerprint": input_fingerprint(runs, ledger_path, gates_dir),
        "scan": {
            "n_metrics_files": len(runs),
            "n_dev_runs": len(dev_runs),
            "n_sealed_runs": len(sealed_runs),
            "n_other_fold_runs": len(other_fold_runs),
            "skipped_metrics": skipped_metrics,
            "pending_ledger_runs_without_metrics": pending,
        },
        "sealed_runs_flagged": [
            {
                "run_id": r["run_id"],
                "arm": r.get("arm"),
                "fold": r.get("fold"),
                "seed": r.get("seed"),
                "action": "excluded_from_dev_aggregates",
            }
            for r in sealed_runs
        ],
        "arms": arms,
        "error_cluster_cards": cards,
        "cost": cost,
        "gates": {"summary_table": gates, "source_dir_name": gates_dir.name},
        "discipline_notes": [
            "Aggregate unit is the reaction cluster/curve, never frames; "
            "n_clusters and unique val-curve counts are reported everywhere.",
            "Sealed folds 3/4 are excluded from every aggregate in this "
            "report and listed under sealed_runs_flagged.",
            "Per-run cluster bootstrap CIs are REUSED from metrics.json "
            "(unit=reaction_cluster); no frame-level resampling is performed.",
            "Arm-level CI resampling unit is the run (fold×seed); runs within "
            "an arm share fold-level val curves, so arm CIs are descriptive, "
            "not independent-cluster discovery claims.",
            "No cherry-picking: every readable dev run with the required "
            "metric fields is included; skipped/pending runs are listed.",
        ],
    }
    return report


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(report: dict[str, Any]) -> str:
    out: list[str] = []
    fp = report["input_fingerprint"]
    out.append(f"# E7 cross-arm statistics report — {report['campaign_id']}")
    out.append("")
    out.append(f"- schema: `{report['schema']}`")
    out.append(f"- aggregate unit: **{report['aggregate_unit']}** (dev folds {report['dev_folds']}; sealed folds {report['sealed_folds']} excluded)")
    boot = report["bootstrap"]
    out.append(f"- arm-level CI: percentile bootstrap of the mean over run-level cluster-medians, seed={boot['seed']}, draws={boot['draws']}")
    out.append(f"- input fingerprint: {fp['n_metrics_files']} metrics files, `metrics_set_sha256={fp['metrics_set_sha256'][:16]}…`, `run_records_sha256={(fp['run_records_sha256'] or 'none')[:16]}…`")
    out.append("")

    scan = report["scan"]
    out.append("## Scan")
    out.append("")
    out.append(f"- metrics files: {scan['n_metrics_files']} (dev {scan['n_dev_runs']}, sealed {scan['n_sealed_runs']}, other-fold {scan['n_other_fold_runs']})")
    out.append(f"- skipped metrics: {len(scan['skipped_metrics'])}  |  ledger runs pending metrics: {len(scan['pending_ledger_runs_without_metrics'])}")
    sealed = report["sealed_runs_flagged"]
    out.append(f"- sealed runs encountered & excluded: {len(sealed)}" + (f" → {', '.join(s['run_id'][:8] + '(' + str(s['arm']) + ' f' + str(s['fold']) + ')' for s in sealed)}" if sealed else ""))
    out.append("")

    out.append("## Cross-arm comparison (dev folds 0–2)")
    out.append("")
    out.append("All values are run-level cluster-medians; `n_runs` = fold×seed evaluations; clusters = unique val curves.")
    out.append("")
    out.append("| arm | n_runs | n_clusters (evals) | n_val_curves | RMSE/zero mean ± std [CI95] | sign med | SRE med | cos med | dead-zone med | frac better zero |")
    out.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---|")
    for arm, a in report["arms"].items():
        m = a["metrics"]
        rmse = m["rmse_over_zero"]
        ci = rmse.get("ci95_mean")
        ci_s = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "—"
        frac = m["fraction_better_than_zero"]
        frac_s = f"{_fmt(frac['fraction'], 3)} ({frac['n_better']}/{frac['n_defined']})"
        out.append(
            f"| {arm} | {a['n_runs']} | {a['cluster_evaluations_total']} | {a['n_unique_val_curves']} "
            f"| {_fmt(rmse['mean'])} ± {_fmt(rmse['std'])} {ci_s} "
            f"| {_fmt(m['sign_accuracy_cluster_median']['median'])} "
            f"| {_fmt(m['sre_cluster_median']['median'])} "
            f"| {_fmt(m['cos_eff_median']['median'])} "
            f"| {_fmt(m['dead_zone_fraction']['median'])} "
            f"| {frac_s} |"
        )
    out.append("")
    out.append("RMSE/zero < 1.0 beats the zero-step baseline. Per-run cluster-level CI95s (reused from metrics.json) are in the JSON report under `arms.<arm>.per_run_cluster_bootstrap_ci95`.")
    out.append("")

    out.append("### Seed stability (RMSE/zero by fold×seed; ≥2/3 seeds < 1.0)")
    out.append("")
    out.append("| arm | fold: {seed: value, …} | folds with ≥2/3 seeds < 1.0 |")
    out.append("|---|---|---:|")
    for arm, a in report["arms"].items():
        st = a["seed_stability"]
        parts = []
        for f, sm in st["rmse_over_zero_by_fold_seed"].items():
            inner = ", ".join(f"{s}: {_fmt(v)}" for s, v in sm.items())
            parts.append(f"f{f} {{{inner}}}")
        out.append(f"| {arm} | {'; '.join(parts)} | {st['n_folds_with_2of3_seeds_below_1']} |")
    out.append("")

    cards = report["error_cluster_cards"]
    out.append(f"## Error cluster cards (worst {cards['worst_n']} dev evaluations per arm)")
    out.append("")
    out.append(f"_{cards['unit_note']}_")
    out.append("")
    for arm, arm_cards in cards["cards_by_arm"].items():
        out.append(f"### {arm}")
        out.append("")
        out.append("| run | fold×seed | val curves (cluster units) | n_cl | RMSE/zero [CI95] | sign | SRE | curve quality |")
        out.append("|---|---|---|---:|---|---:|---:|---|")
        for c in arm_cards:
            ci = c["cluster_ci95"]
            ci_s = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "—"
            q = ", ".join(
                f"{cv.split('_')[-1]}:{(c['curve_quality'].get(cv) or {}).get('quality_status', '?')}"
                for cv in c["val_curves"]
            )
            out.append(
                f"| {c['run_id'][:8]} | f{c['fold']}×s{c['seed']} | {', '.join(cv.split('_')[-1] for cv in c['val_curves'])} "
                f"| {_fmt(c['n_clusters'], 0)} | {_fmt(c['median_rmse_over_zero'])} {ci_s} "
                f"| {_fmt(c['sign_accuracy_cluster_median'])} | {_fmt(c['sre_cluster_median'])} | {q} |"
            )
        out.append("")

    cost = report["cost"]
    t = cost["totals"]
    out.append("## Cost ledger")
    out.append("")
    out.append(f"_{cost['ledger_note']}_")
    out.append("")
    out.append(f"- ledger lines: {t['n_ledger_lines']}, unique runs: {t['n_unique_runs']}, statuses: {t['status_counts']}")
    out.append(f"- wall-clock total: {_fmt(t['wall_clock_s_total'], 1)} s  |  GPU-hours estimate: {_fmt(t['gpu_hours_estimate'], 3)} h")
    out.append("")
    out.append("| phase | arm | n_runs | status counts | attempts | retried | wall (metrics) s | wall (ledger span) s | GPU-h est |")
    out.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for b in cost["by_phase_arm"]:
        out.append(
            f"| {b['phase']} | {b['arm']} | {b['n_runs']} | {b['status_counts']} | {b['total_attempts']} "
            f"| {b['n_retried']} | {_fmt(b['wall_clock_s_metrics'], 1)} ({b['wall_clock_s_metrics_n']} runs) "
            f"| {_fmt(b['wall_clock_s_ledger_span'], 1)} ({b['wall_clock_s_ledger_span_n']} runs) "
            f"| {_fmt(b['gpu_hours_estimate'], 3)} |"
        )
    out.append("")
    if cost["blocked"]:
        out.append("### Blocked runs")
        out.append("")
        out.append("| run | phase | arm | fold | reason | at |")
        out.append("|---|---|---|---:|---|---|")
        for b in cost["blocked"]:
            out.append(
                f"| {b['run_id'][:8]} | {b['phase']} | {b['arm']} | {b['fold']} | {b['reason']} | {b['at'] or '—'} |"
            )
        out.append("")

    out.append("## Gate snapshot")
    out.append("")
    out.append("| gate | status | at |")
    out.append("|---|---|---|")
    for g in report["gates"]["summary_table"]:
        out.append(f"| {g['gate']} | {g['status']} | {g['at'] or '—'} |")
    out.append("")

    out.append("## Discipline notes")
    out.append("")
    for note in report["discipline_notes"]:
        out.append(f"- {note}")
    out.append("")
    return "\n".join(out)


def _round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        r = round(obj, FLOAT_ROUND)
        return 0.0 if r == 0 else r
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    return obj


def write_outputs(out_dir: Path, label: str, report: dict[str, Any], md: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"r3_stats_report_{label}.json"
    md_path = out_dir / f"r3_stats_report_{label}.md"
    payload = _round_floats(report)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    md_path.write_text(md)
    return [json_path, md_path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-root", default=None, help="campaign outputs dir (contains r3_runs/, run_records.jsonl)")
    parser.add_argument("--gates-dir", default=None, help="docs/agent_runs/state/gates directory")
    parser.add_argument("--out-dir", default=None, help="output dir (default <campaign-root>/stats)")
    parser.add_argument("--label", default="v1")
    parser.add_argument("--worst-n", type=int, default=DEFAULT_WORST_N)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--no-write", action="store_true", help="print report JSON to stdout only")
    args = parser.parse_args(argv)

    campaign_root = resolve_campaign_root(args.campaign_root)
    gates_dir = Path(args.gates_dir) if args.gates_dir else Path(__file__).resolve().parents[3] / "docs/agent_runs/state/gates"
    out_dir = Path(args.out_dir) if args.out_dir else campaign_root / "stats"

    report = build_report(
        campaign_root=campaign_root,
        gates_dir=gates_dir,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_draws=args.bootstrap_draws,
        worst_n=args.worst_n,
    )
    md = render_markdown(report)

    if args.no_write:
        print(json.dumps(_round_floats(report), indent=2, sort_keys=True))
        return 0

    paths = write_outputs(out_dir, args.label, report, md)
    for p in paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
