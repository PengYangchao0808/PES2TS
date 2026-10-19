#!/usr/bin/env python3
"""V5.1a D-baseline: re-derive the V5 49-cluster OOF baseline (plan §3.1).

Read-only over existing V5 outputs (NO new training): loads the five formal
A05 seed-17 checkpoints (``outputs/pesn/v5/sprint/runs/A05_f*_s17_formal``),
predicts each fold's dev-dense pool, and re-derives the cluster-level OOF
aggregates that the V5.1 plan §5.1 cites as the V5 baseline:

* RMSE/zero cluster median/q25/q75, RMSE/Classic median,
* fraction of clusters better than zero (n/49),
* cos_eff / cos_eff-nearTS cluster medians,
* amplitude alpha = ||u||/||y|| per effective atom (||y|| > 0.02 A),
  cluster median -> aggregate median/q25/q75,
* dead-zone fraction (effective atoms with ||u|| < 0.005 A),
* 2000-replay cluster bootstrap (seed 20261018) for the Classic-improvement
  lower 95% bound.

The module ALSO exposes :func:`collect_oof_records` — the single OOF
prediction pass reused by ``diagnose_v51_g53_buckets.py`` (D2) and
``diagnose_v51_hard_clusters.py`` (D3) — plus the pure metric helpers that
``tests/pesn/test_diagnose_v51.py`` pins.  Everything here is deterministic
(CPU or a single cuda:0 device), offline, and writes only to ``--out``.

Plan-cited expectations are recorded with deltas but NEVER forced
(plan §0.2: diagnostics report actuals).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION: Final[str] = "pesn_v51_baseline_oof_diag_v1"
CACHE_SCHEMA: Final[str] = "pesn_v51_oof_cache_v1"

#: Plan §5.1 V5 baseline citations (verification targets, not constraints).
PLAN_CITED: Final[dict] = {
    "rmse_over_zero_median": 0.824,
    "better_than_zero": "43/49",
    "cos_eff_median": 0.780,
    "cos_eff_near_ts_median": 0.785,
    "alpha_median": 0.52,
}

#: Metric definitions frozen here (reported verbatim in every output).
EFFECTIVE_ATOM_TAU: Final[float] = 0.02    # ||y|| gate, A (plan §5.1 M1)
DEAD_ZONE_MAG: Final[float] = 0.005        # ||u|| dead zone, A (G5.3 threshold)
NEAR_TS_CUT: Final[float] = 0.20           # near-TS fraction cut (G5.2)
BOOTSTRAP_DRAWS: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 20261018
ARM: Final[str] = "A05"
SEED: Final[int] = 17
D0: Final[float] = 0.05

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "outputs/pesn/v5/sprint/runs"
DEFAULT_SPLIT = PROJECT_ROOT / "data/manifests/v5/split_manifest.json"
DEFAULT_CONFIG = PROJECT_ROOT / "config/campaigns/v5_sprint.yaml"
DEFAULT_CACHE = (PROJECT_ROOT / "outputs/pesn/v5_1/diagnostics"
                 / "oof_cache_v5.pkl")
DEFAULT_OUT = (PROJECT_ROOT / "outputs/pesn/v5_1/diagnostics"
               / "baseline_oof_v5.json")


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_digest(paths: Sequence[Path]) -> str:
    """Deterministic digest over sorted (relative name, size) pairs.

    Used for the LARGE feature-shard trees: hashing every byte of ~168 MB
    of shards adds minutes without provenance value a reviewer acts on; the
    name+size digest pins the exact shard set instead.  Checkpoints and
    small JSON inputs ARE byte-hashed via :func:`sha256_file`.
    """
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda q: str(q)):
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}\n".encode())
    return h.hexdigest()


def _quantile(values, q: float):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.quantile(np.asarray(vals, dtype=np.float64), q))


def _median(values):
    return _quantile(values, 0.5)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_device(name: str):
    import torch
    if name in ("auto", ""):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# pure metric helpers (unit-tested)
# ---------------------------------------------------------------------------

def atom_alphas(u: np.ndarray, y: np.ndarray, tau: float = EFFECTIVE_ATOM_TAU):
    """Per-effective-atom alpha = ||u||/||y|| for ||y|| > tau (plan M1).

    Returns ``(alphas, effective_mask)``; non-effective atoms (||y|| <= tau)
    are EXCLUDED from ``alphas`` entirely (never coerced to 0 or 1).
    """
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y_norm = np.linalg.norm(y, axis=-1)
    eff = y_norm > float(tau)
    if not bool(eff.any()):
        return np.zeros(0, dtype=np.float64), eff
    a = np.linalg.norm(u[eff], axis=-1) / y_norm[eff]
    return a, eff


def dead_zone_fraction(u: np.ndarray, y: np.ndarray,
                       tau: float = EFFECTIVE_ATOM_TAU,
                       dead: float = DEAD_ZONE_MAG):
    """Fraction of EFFECTIVE atoms whose prediction magnitude is < ``dead``.

    Dead zone is defined on the predicted per-atom displacement ||u|| (A)
    against the G5.3 pair threshold 0.005 A; denominator = effective atoms
    (||y|| > tau).  Returns ``None`` when there are no effective atoms.
    """
    a, eff = atom_alphas(u, y, tau=tau)
    n_eff = int(eff.sum())
    if n_eff == 0:
        return None
    u_norm = np.linalg.norm(np.asarray(u, dtype=np.float64)[eff], axis=-1)
    return float((u_norm < float(dead)).sum()) / n_eff


def bucket_of(abs_dd: float, edges: Sequence[float] = (0.005, 0.01, 0.02)) -> int:
    """D2 bucket index for a |dd_true| value.

    Half-open intervals ``[0, e0) [e0, e1) [e1, e2) [e2, inf)``: a pair
    EXACTLY on an edge belongs to the UPPER (stronger-signal) bucket, so
    |dd| == 0.005 is ACTIVE (weak bucket), 0.01 -> medium, 0.02 -> strong.
    Returns 0 (dead zone), 1 (weak), 2 (medium) or 3 (strong).
    """
    x = abs(float(abs_dd))
    if x < float(edges[0]):
        return 0
    if x < float(edges[1]):
        return 1
    if x < float(edges[2]):
        return 2
    return 3


BUCKET_LABELS: Final[tuple[str, ...]] = (
    "dead_lt_0.005", "weak_0.005_0.01", "medium_0.01_0.02", "strong_ge_0.02")


def bucket_verdict(strong_sign_acc, weak_sign_acc) -> str:
    """Plan §3.1 D2 decision rule on bucketed sign accuracy.

    * strong (>0.9) while weak < 0.8 -> dead-zone/amplitude problem,
      pair margin loss first;
    * strong < 0.8 -> representation/feature/label problem, loss alone
      insufficient;
    * anything else -> mixed / inconclusive.
    """
    if strong_sign_acc is None or weak_sign_acc is None:
        return "insufficient_data"
    s = float(strong_sign_acc)
    w = float(weak_sign_acc)
    if s < 0.8:
        return ("representation/feature/label problem — loss alone "
                "insufficient")
    if s > 0.9 and w < 0.8:
        return "dead-zone/amplitude problem, pair margin loss first"
    return "mixed/inconclusive"


def resolve_cluster_suffix(short_id: str, known_ids: Sequence[str]) -> dict:
    """Match a plan-cited short cluster id (e.g. ``RXN_1675``) to a real
    12-char cluster id by numeric-suffix equality.

    Exact numeric equality wins; otherwise ids whose numeric part ENDS WITH
    the requested digits are collected.  Returns a dict with ``status`` in
    {``resolved``, ``ambiguous``, ``absent``} and ``matches``.
    """
    digits = "".join(ch for ch in str(short_id) if ch.isdigit())
    if not digits:
        return {"status": "absent", "matches": [], "requested": short_id}
    exact, suffix = [], []
    for gid in known_ids:
        g_digits = "".join(ch for ch in str(gid) if ch.isdigit())
        if g_digits == digits:
            exact.append(str(gid))
        elif g_digits.endswith(digits):
            suffix.append(str(gid))
    if len(exact) == 1:
        return {"status": "resolved", "matches": exact, "requested": short_id,
                "rule": "exact_numeric"}
    if len(exact) > 1:
        return {"status": "ambiguous", "matches": sorted(exact),
                "requested": short_id, "rule": "exact_numeric"}
    if len(suffix) == 1:
        return {"status": "resolved", "matches": suffix, "requested": short_id,
                "rule": "numeric_suffix"}
    if len(suffix) > 1:
        return {"status": "ambiguous", "matches": sorted(suffix),
                "requested": short_id, "rule": "numeric_suffix"}
    return {"status": "absent", "matches": [], "requested": short_id,
            "rule": "numeric_suffix"}


# ---------------------------------------------------------------------------
# OOF prediction pass (shared by D-baseline / D2 / D3)
# ---------------------------------------------------------------------------

def _load_split_groups(split_path: Path, fold: int) -> dict[str, list[str]]:
    doc = json.loads(split_path.read_text(encoding="utf-8"))
    group_fold = doc["group_fold"]
    sel_path = PROJECT_ROOT / (
        "outputs/pesn/direct_egnn_v1_20261015/splits/"
        "selection_direct_fit50.json")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for gid, rid in zip(sel.get("group_ids", []), sel.get("reaction_ids", [])):
        if int(group_fold[str(gid)]) == int(fold):
            out.setdefault(str(gid), []).append(str(rid))
    return out


def _fold_checkpoint(runs_root: Path, fold: int) -> Path:
    return (runs_root / f"{ARM}_f{fold}_s{SEED}_formal" / "checkpoints"
            / "checkpoint_final.pt")


def _oof_fingerprint(runs_root: Path, split_path: Path, d0: float) -> dict:
    fps = {}
    for fold in range(5):
        ckpt = _fold_checkpoint(runs_root, fold)
        fps[f"fold{fold}_checkpoint_sha256"] = (
            sha256_file(ckpt) if ckpt.is_file() else None)
    fps["split_manifest_sha256"] = sha256_file(split_path)
    fps["arm"] = ARM
    fps["seed"] = SEED
    fps["d0"] = float(d0)
    return fps


def collect_oof_records(*, runs_root: Path = DEFAULT_RUNS_ROOT,
                        split_path: Path = DEFAULT_SPLIT,
                        config_path: Path = DEFAULT_CONFIG,
                        d0: float = D0, device_name: str = "cpu",
                        batch_size: int = 32,
                        cache_path: Path | None = DEFAULT_CACHE,
                        use_cache: bool = True) -> list[dict]:
    """One deterministic OOF prediction pass over the five formal runs.

    Returns a list of per-sample records with RAW model displacement ``u``
    [m,3], classic M0 baseline ``u_classic`` [m,3], teacher ``y`` [m,3],
    the CURRENT-frame geometry ``pos`` [n,3], local ``key_rows``, the
    ``pair_edit``-active key pairs (node-axis segmentation, identical to
    ``run_v5_sprint._static_diagnostics`` / ``evaluate_g53``) and their
    signed ΔB values, plus fold/group/near_ts metadata.
    """
    fingerprint = _oof_fingerprint(runs_root, split_path, d0)
    if any(v is None for v in
           [fingerprint[f"fold{f}_checkpoint_sha256"] for f in range(5)]):
        missing = [f for f in range(5)
                   if fingerprint[f"fold{f}_checkpoint_sha256"] is None]
        raise FileNotFoundError(
            f"missing formal checkpoints for folds {missing} under "
            f"{runs_root} (need {ARM}_f*_s{SEED}_formal/"
            "checkpoints/checkpoint_final.pt)")

    if use_cache and cache_path is not None and Path(cache_path).is_file():
        with Path(cache_path).open("rb") as fh:
            cached = pickle.load(fh)
        if (isinstance(cached, dict)
                and cached.get("schema_version") == CACHE_SCHEMA
                and cached.get("fingerprint") == fingerprint):
            return cached["records"]

    import torch
    from pes2ts_core.config_loader import load_config_mapping
    from pes2ts_core.pesn.models.key_atom_classic import classic_pair_driver
    from pes2ts_core.pesn.models.key_atom_egnn import StarEGNN
    from pes2ts_core.pesn.offline.training.key_atom_bc import (
        load_lane_b_pools, load_lane_b_view)

    config = load_config_mapping(config_path)
    device = _resolve_device(device_name)
    records: list[dict] = []
    for fold in range(5):
        pools = load_lane_b_pools(config, fold=fold, d0=d0,
                                  split_path=str(split_path))
        dev_rows = list(pools["dev_dense"])
        view, collate = load_lane_b_view(
            config=config, samples=dev_rows, fold=fold, d0=d0, anchor="on",
            split_path=str(split_path), device=device)
        model = StarEGNN().to(torch.float32).to(device)
        doc = torch.load(_fold_checkpoint(runs_root, fold),
                         map_location="cpu", weights_only=False)
        if str(doc.get("schema_version")) != "pesn_key_atom_bc_checkpoint_v1":
            raise RuntimeError(
                f"fold {fold}: unexpected checkpoint schema "
                f"{doc.get('schema_version')!r}")
        arm = dict(doc.get("arm") or {})
        if float(arm.get("d0", -1)) != float(d0):
            raise RuntimeError(
                f"fold {fold}: checkpoint d0={arm.get('d0')} != {d0}")
        model.load_state_dict(doc["model"])
        model.train(False)
        n_done = 0
        with torch.no_grad():
            for i in range(0, len(view), batch_size):
                chunk = [view[j] for j in
                         range(i, min(i + batch_size, len(view)))]
                if not chunk:
                    continue
                batch = collate(chunk)
                u, _z = model(batch)
                u_cls, _info = classic_pair_driver(batch)
                ptr = batch.ptr.cpu().numpy()
                key_ptr = batch.key_ptr.cpu().numpy()
                pos_np = batch.pos.cpu().numpy()
                key_index = batch.key_index.cpu().numpy()
                pair_index = batch.pair_index.cpu().numpy()
                pair_edit = batch.pair_edit.cpu().numpy()
                u_np = u.float().cpu().numpy()
                uc_np = u_cls.cpu().numpy()
                y_np = batch.y_teacher.cpu().numpy()
                near = (batch.near_ts.cpu().numpy()
                        if batch.near_ts is not None
                        else np.ones(len(key_ptr) - 1))
                groups = (batch.reaction_ids if batch.reaction_ids
                          else tuple("unknown" for _ in range(len(key_ptr) - 1)))
                branches = (batch.branch.cpu().numpy()
                            if batch.branch is not None
                            else np.zeros(len(key_ptr) - 1, dtype=np.int64))
                for b in range(len(key_ptr) - 1):
                    n_lo, n_hi = int(ptr[b]), int(ptr[b + 1])
                    k_lo, k_hi = int(key_ptr[b]), int(key_ptr[b + 1])
                    if k_hi <= k_lo:
                        continue
                    key_rows = [int(v) - n_lo for v in key_index[k_lo:k_hi]]
                    k_pairs: list[tuple[int, int]] = []
                    k_edit: list[float] = []
                    for col in range(pair_index.shape[1]):
                        a = int(pair_index[0, col])
                        c = int(pair_index[1, col])
                        # node-axis segmentation (run_v5_sprint
                        # ._static_diagnostics; identical to evaluate_g53)
                        if n_lo <= a < n_hi and n_lo <= c < n_hi and \
                                float(pair_edit[col]) != 0.0:
                            k_pairs.append((a - n_lo, c - n_lo))
                            k_edit.append(float(pair_edit[col]))
                    records.append({
                        "fold": int(fold),
                        "sample_index": n_done + b,
                        "group_id": str(groups[b]),
                        "reaction_id": str(groups[b]),
                        "branch": int(branches[b]),
                        "near_ts": float(near[b]),
                        "pos": pos_np[n_lo:n_hi].astype(np.float64),
                        "key_rows": key_rows,
                        "pairs": k_pairs,
                        "pair_edit": k_edit,
                        "u": u_np[k_lo:k_hi].astype(np.float64),
                        "u_classic": uc_np[k_lo:k_hi].astype(np.float64),
                        "y": y_np[k_lo:k_hi].astype(np.float64),
                    })
                n_done += len(chunk)
    if use_cache and cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(cache_path).open("wb") as fh:
            pickle.dump({"schema_version": CACHE_SCHEMA,
                         "fingerprint": fingerprint,
                         "records": records}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
    return records


# ---------------------------------------------------------------------------
# aggregates
# ---------------------------------------------------------------------------

def cluster_metrics(records: Sequence[dict]) -> tuple[dict, dict]:
    """Per-cluster + aggregate RMSE/cos metrics via the frozen V5 stack."""
    from pes2ts_core.pesn.offline.training.key_atom_bc import (
        compute_cluster_metrics)
    per_sample = [{
        "group_id": r["group_id"],
        "u": r["u"], "y": r["y"], "u_classic": r["u_classic"],
        "near_ts": r["near_ts"],
    } for r in records]
    return compute_cluster_metrics(per_sample, tau=EFFECTIVE_ATOM_TAU,
                                   near_ts_cut=NEAR_TS_CUT)


def alpha_and_deadzone(records: Sequence[dict],
                       tau: float = EFFECTIVE_ATOM_TAU) -> dict:
    """Per-cluster alpha (||u||/||y||) medians + dead-zone fractions.

    FOUR alpha definitions are reported so the plan-cited 0.52 can be
    reconciled without forcing (the citation's exact convention is not
    recorded in the plan):

    * ``effective_atom_median`` — per-atom ||u||/||y|| over effective atoms
      (||y|| > tau), cluster median → cross-cluster median (PRIMARY);
    * ``all_atom_median`` — same over ALL atoms with ||y|| > 1e-12;
    * ``rms_ratio`` — per-cluster rms(||u||)/rms(||y||) over effective atoms;
    * ``frame_mean_norm_ratio`` — per-frame mean||u||/mean||y||, cluster
      median.
    """
    by_group: dict[str, list[dict]] = {}
    for r in records:
        by_group.setdefault(r["group_id"], []).append(r)
    per_cluster: dict[str, dict] = {}
    for gid, rows in sorted(by_group.items()):
        alphas = np.zeros(0)
        alphas_all = np.zeros(0)
        dead_num = 0
        dead_den = 0
        u_eff = np.zeros(0)
        y_eff = np.zeros(0)
        frame_ratios: list[float] = []
        for r in rows:
            u = np.asarray(r["u"], dtype=np.float64)
            y = np.asarray(r["y"], dtype=np.float64)
            a, eff = atom_alphas(u, y, tau=tau)
            alphas = np.concatenate([alphas, a])
            y_norm = np.linalg.norm(y, axis=-1)
            nz = y_norm > 1e-12
            if bool(nz.any()):
                a_all = np.linalg.norm(u[nz], axis=-1) / y_norm[nz]
                alphas_all = np.concatenate([alphas_all, a_all])
            n_eff = int(eff.sum())
            if n_eff:
                un = np.linalg.norm(u[eff], axis=-1)
                u_eff = np.concatenate([u_eff, un])
                y_eff = np.concatenate([y_eff, y_norm[eff]])
                dead_num += int((un < DEAD_ZONE_MAG).sum())
                dead_den += n_eff
                m_u = float(un.mean())
                m_y = float(y_norm[eff].mean())
                if m_y > 1e-12:
                    frame_ratios.append(m_u / m_y)
        rms_ratio = (float(np.sqrt(np.mean(u_eff ** 2)))
                     / float(np.sqrt(np.mean(y_eff ** 2)))
                     if u_eff.size and float(np.mean(y_eff ** 2)) > 0
                     else None)
        per_cluster[gid] = {
            "alpha_cluster_median": _median(alphas),
            "alpha_cluster_q25": _quantile(alphas, 0.25),
            "alpha_cluster_q75": _quantile(alphas, 0.75),
            "alpha_all_atom_median": _median(alphas_all),
            "alpha_rms_ratio": rms_ratio,
            "alpha_frame_mean_norm_ratio_median": _median(frame_ratios),
            "n_effective_atoms": int(alphas.size),
            "dead_zone_fraction": (
                float(dead_num) / dead_den if dead_den else None),
        }
    medians = [v["alpha_cluster_median"] for v in per_cluster.values()]
    return {
        "per_cluster": per_cluster,
        "aggregate": {
            "alpha_cluster_median": _median(medians),
            "alpha_cluster_q25": _quantile(medians, 0.25),
            "alpha_cluster_q75": _quantile(medians, 0.75),
            "alpha_all_atom_median": _median(
                [v["alpha_all_atom_median"] for v in per_cluster.values()]),
            "alpha_rms_ratio_median": _median(
                [v["alpha_rms_ratio"] for v in per_cluster.values()]),
            "alpha_frame_mean_norm_ratio_median": _median(
                [v["alpha_frame_mean_norm_ratio_median"]
                 for v in per_cluster.values()]),
            "dead_zone_fraction_cluster_median": _median(
                [v["dead_zone_fraction"] for v in per_cluster.values()]),
        },
        "definitions": {
            "alpha_primary": "||u_raw||/||y|| per effective atom (||y|| > "
                             "0.02 A); cluster median, then cross-cluster "
                             "median/q25/q75",
            "alpha_all_atom": "same ratio over ALL atoms with ||y|| > 1e-12",
            "alpha_rms_ratio": "per-cluster rms(||u_eff||)/rms(||y_eff||)",
            "alpha_frame_mean_norm_ratio": "per-frame mean||u||/mean||y|| "
                                           "over effective atoms, cluster "
                                           "median",
            "dead_zone_fraction": "share of effective atoms with "
                                  "||u_raw|| < 0.005 A (G5.3 pair threshold "
                                  "applied to per-atom magnitude)",
        },
    }


def classic_bootstrap(per_cluster: dict,
                      draws: int = BOOTSTRAP_DRAWS,
                      seed: int = BOOTSTRAP_SEED) -> dict:
    """Cluster-level bootstrap of the Classic improvement (plan §5.2).

    Resamples the 49 clusters with replacement ``draws`` times (seed fixed);
    per draw the statistic is the MEDIAN of ``1 - rmse_over_classic``; the
    lower 95% bound is the 2.5% quantile of that replay distribution.
    """
    ids = sorted(per_cluster)
    rng = np.random.default_rng(int(seed))
    improvements = np.asarray(
        [1.0 - per_cluster[g]["rmse_over_classic"]
         if per_cluster[g].get("rmse_over_classic") is not None else np.nan
         for g in ids], dtype=np.float64)
    med_zero = np.asarray(
        [per_cluster[g]["rmse_over_zero"]
         if per_cluster[g].get("rmse_over_zero") is not None else np.nan
         for g in ids], dtype=np.float64)
    n = len(ids)
    imp_samples: list[float] = []
    zero_samples: list[float] = []
    for _ in range(int(draws)):
        pick = rng.integers(0, n, size=n)
        si = improvements[pick]
        si = si[np.isfinite(si)]
        if si.size:
            imp_samples.append(float(np.median(si)))
        sz = med_zero[pick]
        sz = sz[np.isfinite(sz)]
        if sz.size:
            zero_samples.append(float(np.median(sz)))
    return {
        "draws": int(draws),
        "seed": int(seed),
        "unit": "reaction_cluster",
        "statistic": "median(1 - rmse_over_classic)",
        "classic_improvement_point": _median(
            [float(v) for v in improvements if np.isfinite(v)]),
        "classic_improvement_lower95": _quantile(imp_samples, 0.025),
        "classic_improvement_upper95": _quantile(imp_samples, 0.975),
        "rmse_over_zero_median_ci95": [
            _quantile(zero_samples, 0.025),
            _quantile(zero_samples, 0.975)],
    }


def build_baseline_doc(records: Sequence[dict], fingerprint: dict) -> dict:
    per_cluster, aggregate = cluster_metrics(records)
    alpha = alpha_and_deadzone(records)
    for gid, entry in alpha["per_cluster"].items():
        per_cluster.setdefault(gid, {}).update({
            "alpha_cluster_median": entry["alpha_cluster_median"],
            "alpha_cluster_q25": entry["alpha_cluster_q25"],
            "alpha_cluster_q75": entry["alpha_cluster_q75"],
            "dead_zone_fraction": entry["dead_zone_fraction"],
        })
    better = sorted(g for g, v in per_cluster.items()
                    if v.get("rmse_over_zero") is not None
                    and v["rmse_over_zero"] < 1.0)
    n_defined = sum(1 for v in per_cluster.values()
                    if v.get("rmse_over_zero") is not None)
    bootstrap = classic_bootstrap(per_cluster)
    actual = {
        "rmse_over_zero_median": aggregate.get("rmse_over_zero"),
        "rmse_over_zero_q25": _quantile(
            [v.get("rmse_over_zero") for v in per_cluster.values()], 0.25),
        "rmse_over_zero_q75": _quantile(
            [v.get("rmse_over_zero") for v in per_cluster.values()], 0.75),
        "rmse_over_classic_median": aggregate.get("rmse_over_classic"),
        "n_better_than_zero": len(better),
        "n_clusters_defined": n_defined,
        "fraction_better_than_zero": (
            len(better) / n_defined if n_defined else None),
        "cos_eff_median": aggregate.get("cos_eff_median"),
        "cos_eff_near_ts_median": aggregate.get("cos_eff_near_ts_median"),
        "alpha_median": alpha["aggregate"]["alpha_cluster_median"],
        "alpha_q25": alpha["aggregate"]["alpha_cluster_q25"],
        "alpha_q75": alpha["aggregate"]["alpha_cluster_q75"],
        "dead_zone_fraction_cluster_median":
            alpha["aggregate"]["dead_zone_fraction_cluster_median"],
        "classic_improvement_lower95":
            bootstrap["classic_improvement_lower95"],
    }

    def _delta(key, cited):
        got = actual.get(key)
        if got is None or cited is None:
            return None
        return float(got) - float(cited)

    comparison = {
        "plan_cited": PLAN_CITED,
        "actual": {k: actual[k] for k in
                   ("rmse_over_zero_median", "cos_eff_median",
                    "cos_eff_near_ts_median", "alpha_median")},
        "deltas_actual_minus_cited": {
            "rmse_over_zero_median": _delta("rmse_over_zero_median", 0.824),
            "cos_eff_median": _delta("cos_eff_median", 0.780),
            "cos_eff_near_ts_median": _delta("cos_eff_near_ts_median", 0.785),
            "alpha_median": _delta("alpha_median", 0.52),
            "n_better_than_zero": (
                actual["n_better_than_zero"] - 43
                if actual["n_better_than_zero"] is not None else None),
        },
        "alpha_definition_variants": {
            "effective_atom_median": alpha["aggregate"][
                "alpha_cluster_median"],
            "all_atom_median": alpha["aggregate"]["alpha_all_atom_median"],
            "rms_ratio_median": alpha["aggregate"][
                "alpha_rms_ratio_median"],
            "frame_mean_norm_ratio_median": alpha["aggregate"][
                "alpha_frame_mean_norm_ratio_median"],
            "note": ("the plan cites alpha~=0.52 without recording its "
                     "aggregation convention; all four definitions are "
                     "reported so the citation can be reconciled without "
                     "forcing any of them"),
        },
        "policy": "actuals recorded with deltas; plan-cited values are "
                  "verification targets and are never forced (plan §0.2)",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "arm": ARM, "seed": SEED, "d0": D0,
        "checkpoint_policy": "checkpoint_final.pt (3000-update endpoint, "
                             "the ONLY saved V5 checkpoint; D1 owns the "
                             "best-checkpoint question)",
        "metric_definitions": {
            "effective_atom_tau_angstrom": EFFECTIVE_ATOM_TAU,
            "near_ts_cut": NEAR_TS_CUT,
            "dead_zone_mag_angstrom": DEAD_ZONE_MAG,
            "aggregation": "frame key-atom errors pooled per reaction "
                           "cluster; cross-cluster median/q25/q75",
        },
        "n_samples": len(records),
        "n_clusters": len(per_cluster),
        "per_cluster": per_cluster,
        "aggregate": actual,
        "bootstrap": bootstrap,
        "plan_comparison": comparison,
        "fingerprint": fingerprint,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="cpu",
                        help="cpu (default), cuda:0, or auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root)
    split_path = Path(args.split_manifest)
    fingerprint = _oof_fingerprint(runs_root, split_path, D0)
    feature_root = PROJECT_ROOT / "outputs/pesn/v5/features" / "d0_0.05"
    fingerprint["feature_shards_digest"] = manifest_digest(
        sorted(feature_root.glob("*.npz")))

    records = collect_oof_records(
        runs_root=runs_root, split_path=split_path,
        config_path=Path(args.config), d0=D0, device_name=args.device,
        batch_size=args.batch_size,
        cache_path=(None if args.no_cache else Path(args.cache)),
        use_cache=not args.no_cache)
    doc = build_baseline_doc(records, fingerprint)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    agg = doc["aggregate"]
    print(f"[baseline_oof] wrote {out}")
    print(f"  clusters={doc['n_clusters']} samples={doc['n_samples']}")
    print(f"  rmse/zero median={agg['rmse_over_zero_median']:.4f} "
          f"(plan 0.824)  better-than-zero={agg['n_better_than_zero']}/"
          f"{agg['n_clusters_defined']} (plan 43/49)")
    print(f"  cos={agg['cos_eff_median']:.4f} (plan 0.780)  "
          f"nearTS={agg['cos_eff_near_ts_median']:.4f} (plan 0.785)  "
          f"alpha={agg['alpha_median']:.4f} (plan 0.52)")
    variants = doc["plan_comparison"]["alpha_definition_variants"]
    print("  alpha variants: all-atom="
          f"{variants['all_atom_median']:.4f} rms-ratio="
          f"{variants['rms_ratio_median']:.4f} frame-mean-norm="
          f"{variants['frame_mean_norm_ratio_median']:.4f}")
    print(f"  classic-improvement lower95="
          f"{agg['classic_improvement_lower95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
