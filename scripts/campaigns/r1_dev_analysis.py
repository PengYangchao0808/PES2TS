#!/usr/bin/env python3
"""R1 dev-arm analysis (campaign V5_1A_R1_REPAIR, CPU only).

Schema/version: ``pesn_v5_1_r1_dev_analysis_v1``.

Re-derives development evidence for the preregistered R1 dev matrix
(``outputs/pesn/v5_1_r1/protocol_r1.json``): for every requested
(arm, fold, seed) cell FOUND ON DISK it

* reads ``run_record.json`` + ``metrics.json`` (reported inner-val
  metrics at ``best_update``),
* loads ``checkpoints/checkpoint_best.pt``, rebuilds the run's FROZEN
  fold inner-val pool (config of THAT run; ``run_record.inner_val_split
  .val_ids`` cross-checked against the rebuilt split) and predicts via
  the canonical :func:`r1_eval.predict_pool`,
* computes :func:`r1_eval.canonical_metrics` +
  :func:`r1_eval.strong_pair_eval` buckets plus the dead/strong extra
  bucket diagnostics (dead-bucket spurious share ``|Δd_pred| ≥ 0.005``
  and mean ``|Δd_pred|``; strong-bucket undershoot share
  ``|Δd_pred| < |Δd_true|``), and per-cluster canonical metrics,

then aggregates per arm per fold (3-seed medians of every scalar),
computes A2-vs-A0 / A3-vs-A0 per-fold deltas, a cluster-level paired
bootstrap (2000 draws, seed 20261018; candidate seeds pooled: per-
cluster medians across the common seeds, then clusters resampled), the
preregistered futility check (protocol §futility_rules) and the A4 gate
inputs (§new_losses.arms.A4 + spec complementarity tolerances).

Missing cells are recorded as ABSENT (never fabricated): a cell is
analyzable only when ``run_record.json`` AND ``metrics.json`` AND
``checkpoints/checkpoint_best.pt`` exist (an in-flight run is absent
with its status as the reason).

Development evidence only — inner-val pools, no outer-dev, no formal
OOF, no sealed data.

Usage::

    # smoke: A0 fold 0, 3 legacy seeds, reported-vs-canonical assertions
    /root/miniforge3/bin/python3 scripts/campaigns/r1_dev_analysis.py \
        --smoke --out outputs/pesn/v5_1_r1/dev/dev_analysis_smoke.json

    # dev analysis (A0/A2/A3 × folds 0-2)
    /root/miniforge3/bin/python3 scripts/campaigns/r1_dev_analysis.py \
        --arms A0,A2,A3 --folds 0,1,2 \
        --out outputs/pesn/v5_1_r1/dev/dev_analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION: Final[str] = "pesn_v5_1_r1_dev_analysis_v1"
PROTOCOL_SCHEMA: Final[str] = "pesn_v5_1_r1_protocol_v1"

LEGACY_RUNS_ROOT: Final[Path] = PROJECT_ROOT / "outputs/pesn/v5_1/runs"
R1_RUNS_ROOT: Final[Path] = PROJECT_ROOT / "outputs/pesn/v5_1_r1/runs"
PROTOCOL_PATH: Final[Path] = (
    PROJECT_ROOT / "outputs/pesn/v5_1_r1/protocol_r1.json")
DEFAULT_OUT: Final[Path] = (
    PROJECT_ROOT / "outputs/pesn/v5_1_r1/dev/dev_analysis.json")

#: The 7 preregistered V5.1a A0 reuse cells (run_v5_1_r1.A0_REUSE_CELLS).
LEGACY_A0_CELLS: Final[frozenset[tuple[int, int]]] = frozenset({
    (0, 17), (0, 29), (0, 43), (1, 17), (2, 17), (3, 17), (4, 17)})

SEEDS: Final[tuple[int, ...]] = (17, 29, 43)
DEV_FOLDS: Final[tuple[int, ...]] = (0, 1, 2)
CANDIDATE_ARMS: Final[tuple[str, ...]] = ("A2", "A3")

#: Smoke: reported vs canonical absolute tolerance (spec: 2e-3).
SMOKE_TOL: Final[float] = 2e-3

#: Protocol §calibration.uncertainty / §futility_rules (frozen).
BOOTSTRAP_DRAWS: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 20261018
FUTILITY_SRE_REGRESSION: Final[float] = 0.03     # absolute, protocol text
FUTILITY_RMSE_REL_REGRESSION: Final[float] = 0.05  # relative, protocol text
FUTILITY_MIN_FOLDS: Final[int] = 2               # ">= 2 of 3 dev folds"

#: A4 gate inputs (spec): SRE improves >= 0.01 median on >= 2/3 dev
#: folds with RMSE within 1%; complementarity tolerances.
A4_SRE_IMPROVEMENT: Final[float] = 0.01
A4_RMSE_REL_TOL: Final[float] = 0.01
A4_MIN_FOLDS: Final[int] = 2
COMPAT_DEAD_SPURIOUS_TOL: Final[float] = 0.02    # A2, absolute share
COMPAT_STRONG_AMP_TOL: Final[float] = 0.1        # A3, absolute ratio drop

#: Preregistered regressions are strict (">0.03", ">5%"): a tiny epsilon
#: keeps float noise (0.70*1.05 → 0.05000000000000014) from firing a
#: boundary that the protocol text treats as NOT a regression.
BOUNDARY_EPS: Final[float] = 1e-9

DELIVERED_STATUSES: Final[tuple[str, ...]] = (
    "complete", "censored_patience", "censored_screen")

#: Flat per-cell scalars aggregated to per-(arm, fold) 3-seed medians.
SCALAR_FIELDS: Final[tuple[str, ...]] = (
    "reported_rmse_over_zero", "reported_sre_median", "reported_near_ts_cos",
    "reported_sign_accuracy", "reported_alpha_median",
    "reported_dead_zone_fraction",
    "canonical_rmse_over_zero", "canonical_sre_cluster_median",
    "canonical_near_ts_cos", "canonical_sign_accuracy",
    "canonical_alpha_median", "canonical_dead_zone_fraction",
    "dead_n", "dead_spurious_ge_0.005", "dead_mean_abs_dd_pred",
    "strong_n", "strong_sign_accuracy", "strong_amplitude_ratio_median",
    "strong_undershoot_share",
)

#: Aggregates used by the futility check and the A4 gate.
FUTILITY_SRE_KEY: Final[str] = "canonical_sre_cluster_median"
FUTILITY_RMSE_KEY: Final[str] = "canonical_rmse_over_zero"


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def import_with_retry(module_name: str, *, retries: int = 3,
                      delay_s: float = 30.0):
    """Import a core module, retrying transient failures (concurrent lane
    edits to ``key_atom_bc.py`` can briefly break imports)."""
    import time

    last: Exception | None = None
    for attempt in range(retries):
        try:
            return __import__(module_name, fromlist=["*"])
        except (ImportError, SyntaxError) as exc:  # pragma: no cover
            last = exc
            if attempt + 1 < retries:
                time.sleep(delay_s)
    raise last  # type: ignore[misc]


def _median(values: Sequence[float | None]) -> float | None:
    """Median over finite non-None values (None when nothing survives)."""
    arr = np.asarray(
        [float(v) for v in values
         if v is not None and np.isfinite(float(v))],
        dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _read_json(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# cell discovery (pure — unit-testable)
# ---------------------------------------------------------------------------

def resolve_cell(arm: str, fold: int, seed: int, *,
                 legacy_root: Path | None = None,
                 r1_root: Path | None = None) -> tuple[Path, str] | None:
    """Locate the run dir for ``(arm, fold, seed)``; None when absent.

    A0 resolves via the preregistered mapping: the 7 legacy V5.1a cells
    live under ``outputs/pesn/v5_1/runs`` (falling back to the R1 root if
    a legacy cell vanished, mirroring the driver), the 8 baseline-fill
    cells and all A2/A3 cells live under ``outputs/pesn/v5_1_r1/runs``.
    Returns ``(run_dir, source)`` with source ``legacy`` / ``r1``.
    """
    legacy = Path(legacy_root) if legacy_root else LEGACY_RUNS_ROOT
    r1 = Path(r1_root) if r1_root else R1_RUNS_ROOT
    if str(arm) == "A0":
        run_id = f"B0_f{int(fold)}_s{int(seed)}_v51"
        order: list[tuple[Path, str]] = []
        if (int(fold), int(seed)) in LEGACY_A0_CELLS:
            order.append((legacy / run_id, "legacy"))
        order.append((r1 / run_id, "r1"))
    else:
        run_id = f"{arm}_f{int(fold)}_s{int(seed)}_v51"
        order = [(r1 / run_id, "r1")]
    for run_dir, source in order:
        if (run_dir / "run_record.json").is_file():
            return run_dir, source
    return None


def cell_availability(run_dir: Path) -> tuple[bool, str]:
    """Analyzable ⇔ run_record + metrics + checkpoint_best all present.

    Returns ``(ok, reason)``; ``reason`` carries the status of an
    in-flight/failed run so absent cells stay explainable.
    """
    if not (run_dir / "run_record.json").is_file():
        return False, "no run_record.json"
    if not (run_dir / "checkpoints/checkpoint_best.pt").is_file():
        record = _read_json(run_dir / "run_record.json") or {}
        status = str(record.get("status", "unknown"))
        return False, f"no checkpoint_best.pt (status={status})"
    if not (run_dir / "metrics.json").is_file():
        record = _read_json(run_dir / "run_record.json") or {}
        status = str(record.get("status", "unknown"))
        return False, f"no metrics.json (status={status})"
    return True, ""


# ---------------------------------------------------------------------------
# per-cell canonical analysis (torch path)
# ---------------------------------------------------------------------------

def _reported_from_metrics(run_record: Mapping, metrics: Mapping) -> dict:
    """Reported inner-val metrics at ``best_update`` (metrics.json
    top-level mirrors the trainer's selection row; run_record is the
    persisted copy — both are kept when they agree)."""
    best = metrics.get("best_update", run_record.get("best_update"))
    inner = dict(metrics.get("inner_val_score")
                 or run_record.get("inner_val_score") or {})
    return {
        "best_update": best,
        "inner_val_score": {
            "rmse_over_zero": inner.get("rmse_over_zero"),
            "sre_median": inner.get("sre_median"),
            "near_ts_cos": inner.get("near_ts_cos"),
        },
        "alpha_median": metrics.get(
            "alpha_median", run_record.get("alpha_median")),
        "dead_zone_fraction": metrics.get(
            "dead_zone_fraction", run_record.get("dead_zone_fraction")),
        "sign_accuracy": metrics.get(
            "sign_accuracy", run_record.get("sign_accuracy")),
    }


def _bucket_extras(records: Sequence[Mapping], spe: Mapping) -> dict:
    """Dead/strong bucket extras beyond strong_pair_eval.

    Uses the frozen ``r1_eval`` helpers only: ``strong_pair_eval`` for
    ``n`` / sign accuracy / amplitude-ratio median (canonical buckets),
    ``_collect_pair_deltas`` + ``pair_bucket`` for the dead-bucket
    spurious share / mean |Δd_pred| and the strong-bucket undershoot
    share that ``strong_pair_eval`` does not expose.
    """
    r1_eval = import_with_retry(
        "pes2ts_core.pesn.offline.evaluation.r1_eval")
    per_bucket = dict(spe.get("per_bucket") or {})
    dd_true, dd_pred = r1_eval._collect_pair_deltas(list(records))
    abs_t = np.abs(dd_true)
    abs_p = np.abs(dd_pred)
    bucket_of = np.asarray(
        [r1_eval.pair_bucket(float(a)) for a in abs_t], dtype=object)
    dead_m = bucket_of == "dead"
    strong_m = bucket_of == "strong"
    sign_thr = float(r1_eval.SIGN_PAIR_THRESHOLD)

    dead_entry = dict(per_bucket.get("dead") or {})
    dead_n = int(dead_entry.get("n_pairs", int(dead_m.sum())))
    dead = {
        "n": dead_n,
        "spurious_ge_0.005": None,
        "n_spurious_ge_0.005": 0,
        "mean_abs_dd_pred": None,
    }
    if dead_n > 0 and int(dead_m.sum()) == dead_n:
        abs_p_dead = abs_p[dead_m]
        n_spur = int((abs_p_dead >= sign_thr).sum())
        dead["n_spurious_ge_0.005"] = n_spur
        dead["spurious_ge_0.005"] = float(n_spur / dead_n)
        dead["mean_abs_dd_pred"] = float(abs_p_dead.mean())
    elif dead_n > 0:
        # defensive: delta enumeration disagrees with the canonical
        # bucket count — keep canonical n, leave extras unresolved
        dead["spurious_ge_0.005"] = None
        dead["mean_abs_dd_pred"] = None

    strong_entry = dict(per_bucket.get("strong") or {})
    strong_n = int(strong_entry.get("n_pairs", int(strong_m.sum())))
    strong = {
        "n": strong_n,
        "sign_accuracy": strong_entry.get("sign_accuracy"),
        "amplitude_ratio_median": strong_entry.get("amplitude_ratio_median"),
        "undershoot_share": None,
        "n_undershoot": 0,
    }
    if strong_n > 0 and int(strong_m.sum()) == strong_n:
        # |dd_true| >= 0.02 A for every strong pair → ratio well-defined
        ratios = abs_p[strong_m] / abs_t[strong_m]
        n_under = int((ratios < 1.0).sum())
        strong["n_undershoot"] = n_under
        strong["undershoot_share"] = float(n_under / strong_n)
    return {"dead": dead, "strong": strong}


def analyze_cell(run_dir: Path, source: str, *, arm: str, fold: int,
                 seed: int, device: str,
                 pool_cache: dict) -> dict:
    """Canonical re-derivation for one delivered cell."""
    import torch

    from pes2ts_core.pesn.models.key_atom_egnn import StarEGNN
    from pes2ts_core.pesn.offline.evaluation import r1_eval
    from pes2ts_core.config_loader import load_config_mapping

    key_atom_bc = import_with_retry(
        "pes2ts_core.pesn.offline.training.key_atom_bc")
    load_lane_b_pools = key_atom_bc.load_lane_b_pools

    run_record = _read_json(run_dir / "run_record.json")
    metrics = _read_json(run_dir / "metrics.json")
    if run_record is None:
        raise RuntimeError(f"{_rel(run_dir)}: unreadable run_record.json")
    if metrics is None:
        raise RuntimeError(f"{_rel(run_dir)}: unreadable metrics.json")
    if int(run_record.get("fold", fold)) != int(fold) or \
            int(run_record.get("seed", seed)) != int(seed):
        raise RuntimeError(
            f"{_rel(run_dir)}: fold/seed mismatch vs requested cell")

    arm_block = dict(run_record.get("arm") or {})
    d0 = float(arm_block.get("d0", 0.05))
    anchor = str(arm_block.get("anchor", "on"))
    config_path = Path(str(run_record.get("config_path", "")))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{_rel(run_dir)}: config {config_path} not found")
    config = load_config_mapping(config_path)

    split_path = str(PROJECT_ROOT / str(
        (config.get("paths") or {}).get(
            "split_manifest", "data/manifests/v5/split_manifest.json")))
    v51_inner = (config.get("v51") or {}).get("inner_val") or {}
    inner_val = {"n_val": int(v51_inner.get("n_val_clusters", 8)),
                 "seed": int(v51_inner.get("seed", 20261019))}

    # fold inner-val pool (cached per config+fold+d0+split — identical
    # across the 3 seeds of a cell)
    cache_key = (str(config_path), int(fold), d0, split_path,
                 inner_val["n_val"], inner_val["seed"])
    pools = pool_cache.get(cache_key)
    if pools is None:
        pools = load_lane_b_pools(
            config, fold=int(fold), d0=d0, split_path=split_path,
            inner_val=inner_val)
        pool_cache[cache_key] = pools
    pool_val_ids = list(pools["inner_split"]["val_ids"])
    record_val_ids = list(
        (run_record.get("inner_val_split") or {}).get("val_ids") or [])
    if record_val_ids and pool_val_ids != record_val_ids:
        raise RuntimeError(
            f"{_rel(run_dir)}: rebuilt inner-val split {pool_val_ids} != "
            f"run_record {record_val_ids}")

    ckpt_path = run_dir / "checkpoints/checkpoint_best.pt"
    doc = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if str(doc.get("schema_version")) != "pesn_key_atom_bc_checkpoint_v1":
        raise RuntimeError(
            f"{_rel(run_dir)}: unexpected checkpoint schema "
            f"{doc.get('schema_version')!r}")
    model = StarEGNN(
        magnitude_head=bool(arm_block.get("magnitude_head", False))
    ).to(torch.float32).to(torch.device(device))
    model.load_state_dict(doc["model"])
    records = r1_eval.predict_pool(
        model, list(pools["inner_val_dense"]), torch.device(device),
        config=config, fold=int(fold), d0=d0, split_path=split_path,
        anchor=anchor)
    del model

    clusters = sorted({str(r["cluster"]) for r in records})
    if clusters != sorted(pool_val_ids):
        raise RuntimeError(
            f"{_rel(run_dir)}: predicted clusters {clusters} != "
            f"val ids {sorted(pool_val_ids)}")

    canon = r1_eval.canonical_metrics(records)
    agg = dict(canon["aggregate"])
    spe = r1_eval.strong_pair_eval(records)
    buckets = _bucket_extras(records, spe)
    reported = _reported_from_metrics(run_record, metrics)

    sre_pc = dict(canon["g53"]["per_cluster_sre"])
    sign_pc = dict(canon["g53"]["per_cluster_sign_accuracy"])
    per_cluster: dict[str, dict] = {}
    for cid, entry in dict(canon["per_cluster"]).items():
        per_cluster[cid] = {
            "n_samples": entry.get("n_samples"),
            "rmse_over_zero": entry.get("rmse_over_zero"),
            "alpha_median": entry.get("alpha_median"),
            "cos_eff_near_ts_median": entry.get("cos_eff_near_ts_median"),
            "sre_cluster_median": sre_pc.get(cid),
            "sign_accuracy": sign_pc.get(cid),
        }

    canonical = {
        "rmse_over_zero": agg.get("rmse_over_zero"),
        "sre_cluster_median": agg.get("sre_cluster_median"),
        "near_ts_cos": agg.get("cos_eff_near_ts_median"),
        "alpha_median": agg.get("alpha_median"),
        "dead_zone_fraction": agg.get("dead_zone_fraction"),
        "sign_accuracy": agg.get("sign_accuracy_cluster_median"),
        "n_clusters": agg.get("n_clusters"),
    }

    def _diff(a, b):
        if a is None or b is None:
            return None
        return float(abs(float(a) - float(b)))

    cell = {
        "arm": arm,
        "fold": int(fold),
        "seed": int(seed),
        "run_id": run_record.get("run_id", run_dir.name),
        "run_dir": _rel(run_dir),
        "source": source,
        "status": run_record.get("status"),
        "config_path": _rel(config_path),
        "loss_schema_version": run_record.get("loss_schema_version"),
        "inner_val_split_hash": run_record.get("inner_val_split_hash"),
        "inner_val_cluster_ids": pool_val_ids,
        "n_records": len(records),
        "reported": reported,
        "canonical": canonical,
        "reported_vs_canonical_abs_diff": {
            "rmse_over_zero": _diff(
                reported["inner_val_score"]["rmse_over_zero"],
                canonical["rmse_over_zero"]),
            "sre_cluster_median": _diff(
                reported["inner_val_score"]["sre_median"],
                canonical["sre_cluster_median"]),
            "near_ts_cos": _diff(
                reported["inner_val_score"]["near_ts_cos"],
                canonical["near_ts_cos"]),
            "alpha_median": _diff(
                reported["alpha_median"], canonical["alpha_median"]),
            "dead_zone_fraction": _diff(
                reported["dead_zone_fraction"],
                canonical["dead_zone_fraction"]),
            "sign_accuracy": _diff(
                reported["sign_accuracy"], canonical["sign_accuracy"]),
        },
        "buckets": buckets,
        "per_cluster": per_cluster,
    }
    cell["scalars"] = cell_scalars(cell)
    return cell


def cell_scalars(cell: Mapping) -> dict:
    """Flatten a cell record to the aggregation scalar set."""
    rep = dict(cell.get("reported") or {})
    inner = dict(rep.get("inner_val_score") or {})
    can = dict(cell.get("canonical") or {})
    bk = dict(cell.get("buckets") or {})
    dead = dict(bk.get("dead") or {})
    strong = dict(bk.get("strong") or {})
    return {
        "reported_rmse_over_zero": inner.get("rmse_over_zero"),
        "reported_sre_median": inner.get("sre_median"),
        "reported_near_ts_cos": inner.get("near_ts_cos"),
        "reported_sign_accuracy": rep.get("sign_accuracy"),
        "reported_alpha_median": rep.get("alpha_median"),
        "reported_dead_zone_fraction": rep.get("dead_zone_fraction"),
        "canonical_rmse_over_zero": can.get("rmse_over_zero"),
        "canonical_sre_cluster_median": can.get("sre_cluster_median"),
        "canonical_near_ts_cos": can.get("near_ts_cos"),
        "canonical_sign_accuracy": can.get("sign_accuracy"),
        "canonical_alpha_median": can.get("alpha_median"),
        "canonical_dead_zone_fraction": can.get("dead_zone_fraction"),
        "dead_n": dead.get("n"),
        "dead_spurious_ge_0.005": dead.get("spurious_ge_0.005"),
        "dead_mean_abs_dd_pred": dead.get("mean_abs_dd_pred"),
        "strong_n": strong.get("n"),
        "strong_sign_accuracy": strong.get("sign_accuracy"),
        "strong_amplitude_ratio_median": strong.get(
            "amplitude_ratio_median"),
        "strong_undershoot_share": strong.get("undershoot_share"),
    }


# ---------------------------------------------------------------------------
# aggregation / deltas / bootstrap (pure — unit-testable)
# ---------------------------------------------------------------------------

def aggregate_cells(cells: Sequence[Mapping], *, folds: Sequence[int],
                    seeds: Sequence[int]) -> dict:
    """Per (arm, fold): 3-seed medians of every scalar (missing seeds
    simply drop out — ``n_seeds`` reports coverage, nothing is padded)."""
    out: dict[str, dict] = {}
    for cell in cells:
        arm = str(cell["arm"])
        fold = int(cell["fold"])
        out.setdefault(arm, {}).setdefault(str(fold), {
            "fold": fold,
            "seeds": [],
            "n_seeds": 0,
            "missing_seeds": [int(s) for s in seeds],
            "run_ids": [],
            "scalars": {},
        })
        entry = out[arm][str(fold)]
        entry["seeds"].append(int(cell["seed"]))
        entry["run_ids"].append(str(cell.get("run_id", "")))
        scalars = dict(cell.get("scalars") or cell_scalars(cell))
        for key in SCALAR_FIELDS:
            entry["scalars"].setdefault(key, []).append(scalars.get(key))
    for arm, folds_map in out.items():
        for fold_key, entry in folds_map.items():
            present = sorted(set(entry["seeds"]))
            entry["seeds"] = present
            entry["n_seeds"] = len(present)
            entry["missing_seeds"] = sorted(
                set(int(s) for s in seeds) - set(present))
            entry["run_ids"] = sorted(set(entry["run_ids"]))
            entry["scalars"] = {
                key: _median(vals)
                for key, vals in entry["scalars"].items()}
    # keep only requested folds (a stray cell outside --folds is kept in
    # `cells` but never enters the fold aggregates)
    requested = set(int(f) for f in folds)
    return {
        arm: {fk: v for fk, v in folds_map.items()
              if int(fk) in requested}
        for arm, folds_map in out.items()}


def fold_deltas(aggregates: Mapping, *, baseline: str = "A0",
                candidates: Sequence[str] = CANDIDATE_ARMS) -> dict:
    """Per fold: ``candidate_median − baseline_median`` for every scalar."""
    out: dict[str, dict] = {}
    base_folds = dict(aggregates.get(baseline) or {})
    for arm in candidates:
        arm_folds = dict(aggregates.get(arm) or {})
        per_fold: dict[str, dict] = {}
        for fold_key in sorted(arm_folds, key=int):
            if fold_key not in base_folds:
                continue
            cand = dict(arm_folds[fold_key].get("scalars") or {})
            base = dict(base_folds[fold_key].get("scalars") or {})
            deltas: dict[str, float | None] = {}
            for key in SCALAR_FIELDS:
                a, b = cand.get(key), base.get(key)
                deltas[key] = (
                    float(a - b)
                    if a is not None and b is not None else None)
            per_fold[fold_key] = {
                "fold": int(fold_key),
                "n_seeds_candidate": arm_folds[fold_key].get("n_seeds"),
                "n_seeds_baseline": base_folds[fold_key].get("n_seeds"),
                "delta_candidate_minus_baseline": deltas,
            }
        if per_fold:
            out[arm] = per_fold
    return out


def pool_cluster_values(candidate_cells: Sequence[Mapping],
                        baseline_cells: Sequence[Mapping], *,
                        metric_key: str) -> tuple[dict, dict, list[int]]:
    """Pool per-cluster metric values across the COMMON seeds.

    Returns ``(cand_by_cluster, base_by_cluster, common_seeds)`` where
    each value is the median over the seeds present in BOTH arms (a seed
    missing on either side contributes nothing — paired pooling).
    ``metric_key`` addresses a field of ``cell['per_cluster'][cid]``.
    """
    cand_by_seed = {int(c["seed"]): c for c in candidate_cells}
    base_by_seed = {int(c["seed"]): c for c in baseline_cells}
    common = sorted(set(cand_by_seed) & set(base_by_seed))
    cand_vals: dict[str, list[float]] = {}
    base_vals: dict[str, list[float]] = {}
    for seed in common:
        for cid, entry in dict(
                cand_by_seed[seed].get("per_cluster") or {}).items():
            v = entry.get(metric_key)
            if v is not None and np.isfinite(float(v)):
                cand_vals.setdefault(str(cid), []).append(float(v))
        for cid, entry in dict(
                base_by_seed[seed].get("per_cluster") or {}).items():
            v = entry.get(metric_key)
            if v is not None and np.isfinite(float(v)):
                base_vals.setdefault(str(cid), []).append(float(v))
    shared = sorted(set(cand_vals) & set(base_vals))
    cand_pooled = {c: float(np.median(cand_vals[c])) for c in shared}
    base_pooled = {c: float(np.median(base_vals[c])) for c in shared}
    return cand_pooled, base_pooled, common


def bootstrap_pooled(cand_values: Mapping[str, float],
                     base_values: Mapping[str, float], *,
                     draws: int = BOOTSTRAP_DRAWS,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """Cluster-level paired bootstrap of ``median(cand) − median(base)``
    over pooled per-cluster values — delegates to the FROZEN
    :func:`r1_eval.cluster_paired_bootstrap` with one synthetic record
    per cluster (metric_fn = median of the record values, i.e. the
    pooled value itself)."""
    from pes2ts_core.pesn.offline.evaluation import r1_eval

    records_a = [{"cluster": c, "value": v}
                 for c, v in sorted(cand_values.items())]
    records_b = [{"cluster": c, "value": v}
                 for c, v in sorted(base_values.items())]
    if not records_a or not records_b:
        return {
            "schema_version": r1_eval.SCHEMA_VERSION,
            "draws": int(draws), "seed": int(seed),
            "n_clusters": 0, "status": "no_shared_clusters",
        }

    def _metric(sub: Sequence[Mapping]) -> float:
        return float(np.median([float(r["value"]) for r in sub]))

    result = r1_eval.cluster_paired_bootstrap(
        records_a, records_b, _metric, draws=int(draws), seed=int(seed))
    result["status"] = "ok"
    return result


# ---------------------------------------------------------------------------
# futility + A4 gate (pure — unit-testable)
# ---------------------------------------------------------------------------

def _fold_scalar(aggregates: Mapping, arm: str, fold: int,
                 key: str) -> float | None:
    entry = ((aggregates.get(arm) or {}).get(str(fold)) or {})
    return (dict(entry.get("scalars") or {})).get(key)


def futility_check(aggregates: Mapping, *,
                   folds: Sequence[int] = DEV_FOLDS,
                   candidates: Sequence[str] = CANDIDATE_ARMS,
                   sre_regression: float = FUTILITY_SRE_REGRESSION,
                   rmse_regression: float = FUTILITY_RMSE_REL_REGRESSION,
                   min_folds: int = FUTILITY_MIN_FOLDS) -> dict:
    """Protocol §futility_rules: flag ``arm_stop`` when the 3-seed
    median SRE regresses > 0.03 vs A0 on >= 2/3 dev folds OR the
    RMSE/zero median regresses > 5% relative vs A0 on >= 2/3 dev folds.

    Only folds where BOTH the candidate and A0 aggregates exist are
    evaluated; fewer than ``min_folds`` evaluable folds cannot fire the
    rule (reported as ``insufficient_data`` — never silently passed).
    """
    out: dict[str, dict] = {}
    for arm in candidates:
        evaluated: list[int] = []
        sre_regs: list[int] = []
        rmse_regs: list[int] = []
        per_fold: dict[str, dict] = {}
        for fold in folds:
            a0_sre = _fold_scalar(aggregates, "A0", int(fold),
                                  FUTILITY_SRE_KEY)
            ar_sre = _fold_scalar(aggregates, arm, int(fold),
                                  FUTILITY_SRE_KEY)
            a0_rmse = _fold_scalar(aggregates, "A0", int(fold),
                                   FUTILITY_RMSE_KEY)
            ar_rmse = _fold_scalar(aggregates, arm, int(fold),
                                   FUTILITY_RMSE_KEY)
            if (a0_sre is None or ar_sre is None
                    or a0_rmse is None or ar_rmse is None):
                continue
            evaluated.append(int(fold))
            sre_delta = float(ar_sre - a0_sre)
            rmse_rel = (float(ar_rmse - a0_rmse) / float(a0_rmse)
                        if float(a0_rmse) != 0.0 else None)
            sre_hit = sre_delta > sre_regression + BOUNDARY_EPS
            rmse_hit = (rmse_rel is not None
                        and rmse_rel > rmse_regression + BOUNDARY_EPS)
            if sre_hit:
                sre_regs.append(int(fold))
            if rmse_hit:
                rmse_regs.append(int(fold))
            per_fold[str(int(fold))] = {
                "sre_delta": sre_delta,
                "rmse_rel_delta": rmse_rel,
                "sre_regression": bool(sre_hit),
                "rmse_regression": bool(rmse_hit),
            }
        enough = len(evaluated) >= int(min_folds)
        arm_stop = bool(enough and (len(sre_regs) >= int(min_folds)
                                    or len(rmse_regs) >= int(min_folds)))
        out[arm] = {
            "arm": arm,
            "folds_evaluated": evaluated,
            "n_folds_evaluated": len(evaluated),
            "insufficient_data": not enough,
            "sre_regression_folds": sre_regs,
            "rmse_regression_folds": rmse_regs,
            "n_sre_regressions": len(sre_regs),
            "n_rmse_regressions": len(rmse_regs),
            "per_fold": per_fold,
            "arm_stop": arm_stop,
            "rule": ("SRE median regression > 0.03 vs A0 on >= 2/3 dev "
                     "folds OR RMSE/zero median regression > 5% relative "
                     "on >= 2/3 dev folds (protocol_r1.json "
                     "futility_rules.preregistered_stop)"),
        }
    return out


def a4_gate(aggregates: Mapping, *, folds: Sequence[int] = DEV_FOLDS,
            candidates: Sequence[str] = CANDIDATE_ARMS) -> dict:
    """A4 gate inputs (spec).

    ``a2_positive`` / ``a3_positive``: SRE improves >= 0.01 (absolute,
    A0 − arm) on >= 2/3 dev folds with RMSE/zero within +1% on the same
    fold.  Complementarity: A2 must not degrade the dead-bucket spurious
    share vs A0 by > 0.02 absolute, A3 must not degrade the strong-bucket
    amplitude ratio by > 0.1 — worst evaluated fold decides; otherwise
    the arm is flagged ``not complementary``.  Booleans are strict
    (insufficient data ⇒ False with an explicit note, never fabricated
    True).
    """
    per_arm: dict[str, dict] = {}
    positives: dict[str, bool] = {}
    for arm in candidates:
        evaluated: list[int] = []
        qualifying: list[int] = []
        sre_imp: dict[str, float | None] = {}
        rmse_rel: dict[str, float | None] = {}
        for fold in folds:
            a0_sre = _fold_scalar(aggregates, "A0", int(fold),
                                  FUTILITY_SRE_KEY)
            ar_sre = _fold_scalar(aggregates, arm, int(fold),
                                  FUTILITY_SRE_KEY)
            a0_rmse = _fold_scalar(aggregates, "A0", int(fold),
                                   FUTILITY_RMSE_KEY)
            ar_rmse = _fold_scalar(aggregates, arm, int(fold),
                                   FUTILITY_RMSE_KEY)
            if (a0_sre is None or ar_sre is None
                    or a0_rmse is None or ar_rmse is None):
                continue
            evaluated.append(int(fold))
            imp = float(a0_sre - ar_sre)
            rel = (float(ar_rmse - a0_rmse) / float(a0_rmse)
                   if float(a0_rmse) != 0.0 else None)
            sre_imp[str(int(fold))] = imp
            rmse_rel[str(int(fold))] = rel
            if imp >= A4_SRE_IMPROVEMENT and rel is not None \
                    and rel <= A4_RMSE_REL_TOL:
                qualifying.append(int(fold))
        enough = len(evaluated) >= A4_MIN_FOLDS
        positive = bool(enough and len(qualifying) >= A4_MIN_FOLDS)
        positives[arm] = positive
        note = ""
        if not evaluated:
            note = f"{arm}: no folds with both {arm} and A0 data"
        elif not enough:
            note = (f"{arm}: only {len(evaluated)}/{len(list(folds))} "
                    f"dev folds evaluable — gate cannot pass")
        per_arm[arm] = {
            "arm": arm,
            "folds_evaluated": evaluated,
            "folds_qualifying": qualifying,
            "n_qualifying": len(qualifying),
            "sre_improvement_A0_minus_arm": sre_imp,
            "rmse_rel_delta_arm_vs_A0": rmse_rel,
            "positive": positive,
            "note": note,
        }

    # -- complementarity (worst evaluated fold decides) ---------------------
    def _worst_fold_delta(arm: str, key: str, *, higher_is_better: bool
                          ) -> tuple[list[int], dict[str, float], float | None]:
        """Worst (largest) DEGRADATION of the candidate vs A0: positive
        always means the candidate is worse, whichever direction the
        raw metric runs."""
        evaluated: list[int] = []
        deltas: dict[str, float] = {}
        for fold in folds:
            a0 = _fold_scalar(aggregates, "A0", int(fold), key)
            ar = _fold_scalar(aggregates, arm, int(fold), key)
            if a0 is None or ar is None:
                continue
            evaluated.append(int(fold))
            deltas[str(int(fold))] = (
                float(a0 - ar) if higher_is_better
                else float(ar - a0))
        worst = max(deltas.values()) if deltas else None
        return evaluated, deltas, worst

    a2_eval, a2_deltas, a2_worst = _worst_fold_delta(
        "A2", "dead_spurious_ge_0.005", higher_is_better=False)
    a3_eval, a3_deltas, a3_worst = _worst_fold_delta(
        "A3", "strong_amplitude_ratio_median", higher_is_better=True)
    a2_ok = a2_worst is not None and a2_worst <= COMPAT_DEAD_SPURIOUS_TOL
    a3_ok = a3_worst is not None and a3_worst <= COMPAT_STRONG_AMP_TOL

    notes: list[str] = []
    if "A2" in candidates:
        if a2_worst is None:
            notes.append("A2 not complementary: no A2-vs-A0 dead-spurious "
                         "data (insufficient)")
        elif not a2_ok:
            notes.append(
                f"A2 not complementary: dead-bucket spurious degrades by "
                f"{a2_worst:+.4f} > {COMPAT_DEAD_SPURIOUS_TOL} absolute")
        else:
            notes.append(
                f"A2 complementary: worst-fold dead-bucket spurious delta "
                f"{a2_worst:+.4f} <= {COMPAT_DEAD_SPURIOUS_TOL}")
    if "A3" in candidates:
        if a3_worst is None:
            notes.append("A3 not complementary: no A3-vs-A0 strong "
                         "amplitude data (insufficient)")
        elif not a3_ok:
            notes.append(
                f"A3 not complementary: strong amplitude ratio drops by "
                f"{a3_worst:+.4f} > {COMPAT_STRONG_AMP_TOL}")
        else:
            notes.append(
                f"A3 complementary: worst-fold strong amplitude delta "
                f"{a3_worst:+.4f} >= -{COMPAT_STRONG_AMP_TOL}")

    # both arms must be positive (missing arm ⇒ False, never a vacuous
    # pass) and complementarity must hold wherever it was evaluated
    a4_open = bool(
        positives.get("A2", False)
        and positives.get("A3", False)
        and ("A2" not in candidates or a2_ok)
        and ("A3" not in candidates or a3_ok))

    return {
        "a2_positive": bool(positives.get("A2", False)),
        "a3_positive": bool(positives.get("A3", False)),
        "criteria": {
            "positive": ("SRE median improves >= 0.01 vs A0 on >= 2/3 dev "
                         "folds with RMSE/zero relative delta <= +1% on "
                         "the same fold"),
            "a2_complementarity": ("dead-bucket spurious share delta "
                                   f"<= {COMPAT_DEAD_SPURIOUS_TOL} absolute "
                                   "(all evaluated folds)"),
            "a3_complementarity": ("strong-bucket amplitude ratio drop "
                                   f"<= {COMPAT_STRONG_AMP_TOL} absolute "
                                   "(all evaluated folds)"),
        },
        "per_arm": per_arm,
        "complementarity": {
            "a2_dead_spurious": {
                "folds_evaluated": a2_eval,
                "delta_arm_minus_A0": a2_deltas,
                "worst_fold_delta": a2_worst,
                "tolerance": COMPAT_DEAD_SPURIOUS_TOL,
                "ok": bool(a2_ok) if "A2" in candidates else None,
            },
            "a3_strong_amplitude_ratio": {
                "folds_evaluated": a3_eval,
                "delta_A0_minus_arm": a3_deltas,
                "worst_fold_delta": a3_worst,
                "tolerance": COMPAT_STRONG_AMP_TOL,
                "ok": bool(a3_ok) if "A3" in candidates else None,
            },
            "notes": notes,
        },
        "a4_gate_open": a4_open,
    }


# ---------------------------------------------------------------------------
# smoke assertions (reported vs canonical)
# ---------------------------------------------------------------------------

def smoke_check(cells: Sequence[Mapping], *,
                tol: float = SMOKE_TOL) -> dict:
    """Assert reported-vs-canonical RMSE/zero and SRE within ``tol``."""
    rows = []
    all_ok = True
    for cell in cells:
        diff = dict(cell.get("reported_vs_canonical_abs_diff") or {})
        rmse_d = diff.get("rmse_over_zero")
        sre_d = diff.get("sre_cluster_median")
        ok = (rmse_d is not None and sre_d is not None
              and rmse_d <= tol and sre_d <= tol)
        all_ok = all_ok and ok
        rows.append({
            "run_id": cell.get("run_id"),
            "fold": cell.get("fold"),
            "seed": cell.get("seed"),
            "reported_rmse_over_zero":
                (dict(cell.get("reported") or {}).get("inner_val_score")
                 or {}).get("rmse_over_zero"),
            "canonical_rmse_over_zero":
                (dict(cell.get("canonical") or {})).get("rmse_over_zero"),
            "abs_diff_rmse": rmse_d,
            "reported_sre_median":
                (dict(cell.get("reported") or {}).get("inner_val_score")
                 or {}).get("sre_median"),
            "canonical_sre_cluster_median":
                (dict(cell.get("canonical") or {})).get(
                    "sre_cluster_median"),
            "abs_diff_sre": sre_d,
            "ok": bool(ok),
        })
    return {"tolerance_abs": tol, "all_ok": bool(all_ok), "cells": rows}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_analysis(*, arms: Sequence[str], folds: Sequence[int],
                 seeds: Sequence[int], device: str, draws: int,
                 bootstrap_seed: int, smoke: bool,
                 legacy_root: Path, r1_root: Path) -> dict:
    pool_cache: dict = {}
    cells: list[dict] = []
    absent: list[dict] = []
    for arm in arms:
        for fold in folds:
            for seed in seeds:
                resolved = resolve_cell(
                    arm, int(fold), int(seed),
                    legacy_root=legacy_root, r1_root=r1_root)
                if resolved is None:
                    absent.append({
                        "arm": arm, "fold": int(fold), "seed": int(seed),
                        "reason": "not on disk",
                    })
                    continue
                run_dir, source = resolved
                ok, reason = cell_availability(run_dir)
                if not ok:
                    absent.append({
                        "arm": arm, "fold": int(fold), "seed": int(seed),
                        "run_dir": _rel(run_dir), "source": source,
                        "reason": reason,
                    })
                    continue
                print(f"[r1_dev_analysis] {arm} fold {fold} seed {seed}: "
                      f"{_rel(run_dir)} ({source}) ...", flush=True)
                cell = analyze_cell(
                    run_dir, source, arm=arm, fold=int(fold),
                    seed=int(seed), device=device,
                    pool_cache=pool_cache)
                rep = cell["reported"]["inner_val_score"]
                can = cell["canonical"]
                d = cell["reported_vs_canonical_abs_diff"]
                print(f"  rmse rep={rep['rmse_over_zero']:.6f} "
                      f"can={can['rmse_over_zero']:.6f} "
                      f"(abs diff {d['rmse_over_zero']:.2e}) | "
                      f"sre rep={rep['sre_median']:.6f} "
                      f"can={can['sre_cluster_median']:.6f} "
                      f"(abs diff {d['sre_cluster_median']:.2e})",
                      flush=True)
                cells.append(cell)

    aggregates = aggregate_cells(cells, folds=folds, seeds=seeds)
    deltas = fold_deltas(aggregates)

    # cluster-level paired bootstrap (candidate seeds pooled) per fold
    bootstrap: dict[str, dict] = {}
    for arm in arms:
        if arm == "A0":
            continue
        cand = [c for c in cells if c["arm"] == arm]
        base = [c for c in cells if c["arm"] == "A0"]
        if not cand or not base:
            continue
        per_fold: dict[str, dict] = {}
        for fold in folds:
            c_f = [c for c in cand if int(c["fold"]) == int(fold)]
            b_f = [c for c in base if int(c["fold"]) == int(fold)]
            if not c_f or not b_f:
                continue
            fold_boot: dict[str, dict] = {}
            for label, metric_key in (
                    ("sre_cluster_median", "sre_cluster_median"),
                    ("rmse_over_zero", "rmse_over_zero")):
                c_vals, b_vals, common = pool_cluster_values(
                    c_f, b_f, metric_key=metric_key)
                boot = bootstrap_pooled(
                    c_vals, b_vals, draws=draws, seed=bootstrap_seed)
                boot["common_seeds_pooled"] = common
                boot["status"] = (
                    "no_common_seeds" if not common else
                    boot.get("status", "ok"))
                fold_boot[label] = boot
            per_fold[str(int(fold))] = fold_boot
        if per_fold:
            bootstrap[arm] = per_fold

    futility = futility_check(aggregates, folds=folds)
    gate = a4_gate(aggregates, folds=folds,
                   candidates=tuple(
                       a for a in CANDIDATE_ARMS if a in arms))

    smoke_result = None
    if smoke:
        smoke_result = smoke_check(cells)

    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "V5_1A_R1_REPAIR",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_evidence_only",
        "disclaimer": (
            "R1 dev analysis over INNER-VAL pools only; no formal OOF "
            "claims, no outer-dev usage, no sealed data. Missing cells "
            "are recorded as absent, never fabricated."),
        "parameters": {
            "arms": list(arms), "folds": [int(f) for f in folds],
            "seeds": [int(s) for s in seeds], "device": device,
            "smoke": bool(smoke),
            "bootstrap_draws": int(draws),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "environment": _environment(device),
        "cells": cells,
        "absent_cells": absent,
        "aggregates": aggregates,
        "deltas_vs_a0": deltas,
        "bootstrap_vs_a0": bootstrap,
        "futility": futility,
        "a4_gate": gate,
        "smoke": smoke_result,
    }


def _environment(device: str) -> dict:
    import platform

    env = {"python": platform.python_version(), "device": device}
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:  # pragma: no cover
        env["numpy"] = None
    try:
        import torch
        env["torch"] = torch.__version__
    except Exception:  # pragma: no cover
        env["torch"] = None
    return env


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r1_dev_analysis.py",
        description="R1 dev-arm analysis (canonical inner-val "
                    "re-derivation, aggregation, futility + A4 gate "
                    "inputs; CPU only)")
    p.add_argument("--arms", default="A0,A2,A3",
                   help="comma-separated arms (default A0,A2,A3)")
    p.add_argument("--folds", default="0,1,2",
                   help="comma-separated dev folds (default 0,1,2)")
    p.add_argument("--seeds", default="17,29,43",
                   help="comma-separated seeds (default 17,29,43)")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"output JSON (default {DEFAULT_OUT})")
    p.add_argument("--device", default="cpu",
                   help="torch device (default cpu; CPU-only campaign)")
    p.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS,
                   help="bootstrap draws (protocol: 2000)")
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED,
                   help="bootstrap seed (protocol: 20261018)")
    p.add_argument("--legacy-runs-root", default=str(LEGACY_RUNS_ROOT))
    p.add_argument("--r1-runs-root", default=str(R1_RUNS_ROOT))
    p.add_argument("--smoke", action="store_true",
                   help="run on A0 fold 0 x 3 seeds only (legacy dirs) "
                        "and assert reported vs canonical RMSE/SRE "
                        f"within {SMOKE_TOL} absolute")
    return p


def _csv_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).replace(" ", "").split(",")
            if part]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    arms = [a for a in str(args.arms).replace(" ", "").split(",") if a]
    folds = _csv_ints(args.folds)
    seeds = _csv_ints(args.seeds)
    if args.smoke:
        arms, folds = ["A0"], [0]
        seeds = list(SEEDS)
        if not all((int(f), int(s)) in LEGACY_A0_CELLS
                   for f in folds for s in seeds):
            raise SystemExit("smoke cells are not the legacy A0 set")
    if not arms or not folds or not seeds:
        raise SystemExit("--arms/--folds/--seeds must be non-empty")

    protocol = _read_json(PROTOCOL_PATH)
    if protocol is None or str(
            protocol.get("schema_version")) != PROTOCOL_SCHEMA:
        raise SystemExit(
            f"unexpected protocol schema in {PROTOCOL_PATH} "
            f"(want {PROTOCOL_SCHEMA})")

    doc = run_analysis(
        arms=arms, folds=folds, seeds=seeds, device=str(args.device),
        draws=int(args.draws), bootstrap_seed=int(args.bootstrap_seed),
        smoke=bool(args.smoke),
        legacy_root=Path(args.legacy_runs_root),
        r1_root=Path(args.r1_runs_root))
    doc["protocol_excerpt"] = {
        "futility_rules": protocol.get("futility_rules"),
        "partitions_dev_folds_for_tuning":
            (protocol.get("partitions") or {}).get("dev_folds_for_tuning"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    json.loads(tmp.read_text(encoding="utf-8"))  # validity re-check
    tmp.replace(out_path)

    n_cells = len(doc["cells"])
    n_absent = len(doc["absent_cells"])
    print(f"[r1_dev_analysis] {n_cells} cells analyzed, {n_absent} absent")
    print(f"[r1_dev_analysis] wrote {out_path}")

    if args.smoke:
        smoke = doc["smoke"] or {}
        print(f"[r1_dev_analysis] SMOKE (tolerance abs "
              f"{smoke.get('tolerance_abs')}):")
        for row in smoke.get("cells", []):
            print(f"  {row['run_id']}: rmse rep="
                  f"{row['reported_rmse_over_zero']:.9f} can="
                  f"{row['canonical_rmse_over_zero']:.9f} "
                  f"diff={row['abs_diff_rmse']:.3e} | sre rep="
                  f"{row['reported_sre_median']:.9f} can="
                  f"{row['canonical_sre_cluster_median']:.9f} "
                  f"diff={row['abs_diff_sre']:.3e} "
                  f"-> {'OK' if row['ok'] else 'FAIL'}")
        if not smoke.get("all_ok", False):
            print("[r1_dev_analysis] SMOKE FAILED: reported vs canonical "
                  "RMSE/SRE outside tolerance", file=sys.stderr)
            return 1
        print("[r1_dev_analysis] SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
