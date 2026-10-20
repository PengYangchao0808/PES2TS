"""R1 evaluation/consistency/calibration lane (campaign V5_1A_R1_REPAIR).

Schema/version: ``pesn_v5_1_r1_eval_v1``.

Canonical evaluator for the preregistered R1 post-hoc global-scalar
calibration study (``outputs/pesn/v5_1_r1/protocol_r1.json`` §calibration):

* :func:`predict_pool` — one deterministic CPU (or single-device) forward
  pass over Lane B dense pools; returns per-sample records mirroring the
  V5.1a OOF cache shape with the RAW (unbounded, unscaled) model
  displacement ``u``;
* :func:`canonical_metrics` — the SAME per-cluster + aggregate fields as
  the trainer's inner-val rows in ``metrics.json`` (RMSE/zero, alpha
  median/Q25/Q75, effective-atom cos medians, SRE/sign/dead-zone via the
  FROZEN :func:`evaluate_g53` — this module never reimplements SRE, the
  sign rule, or the dead zone);
* :func:`scale_solution` / :func:`apply_scale` — the analytic
  displacement-MSE-optimal scalar ``s* = Σ⟨u,y⟩ / Σ‖u‖²`` (clipped ≥ 0,
  protocol §calibration.analytic_solution) and its application
  ``u → s·u``;
* :func:`strong_pair_eval` — exact-Δd pair buckets
  (dead/weak/medium/strong, protocol §pair_buckets_angstrom edge policy:
  half-open [lo, hi), a pair exactly on an edge belongs to the STRONGER
  bucket) with sign accuracy, abstention, amplitude-ratio and residual
  statistics;
* :func:`cluster_paired_bootstrap` — cluster-level paired bootstrap
  (frozen seed 20261018) of a scalar metric fn over two record sets.

Frozen conventions reused from the trainer/protocol modules (READ-ONLY
imports, never retyped): :func:`compute_cluster_metrics` (per effective
atom ‖u‖/‖y‖ with ‖y‖ > 0.02 Å, cluster median then median over
clusters), :func:`evaluate_g53` (SRE, sign accuracy with abstention =
incorrect, dead-zone fraction) and :func:`exact_delta_d`.

Development evidence only — no formal OOF claims; inner-val pools only.
"""
from __future__ import annotations

import math
from typing import Callable, Final, Mapping, Sequence

import numpy as np
import torch

from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
    exact_delta_d,
    evaluate_g53,
)
from pes2ts_core.pesn.offline.training.key_atom_bc import (
    compute_cluster_metrics,
    load_lane_b_view,
)

SCHEMA_VERSION: Final[str] = "pesn_v5_1_r1_eval_v1"

#: Frozen protocol constants (protocol_r1.json §calibration / §gates).
EFFECTIVE_ATOM_TAU: Final[float] = 0.02      # ‖y‖ gate, Å
SIGN_PAIR_THRESHOLD: Final[float] = 0.005    # Å (G5.3)
SRE_MIN_TOTAL_CHANGE: Final[float] = 0.01    # Å (G5.3)
NEAR_TS_CUT: Final[float] = 0.20
SOFT_COS_NORM: Final[float] = 1e-4
BOOTSTRAP_DRAWS: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 20261018
#: Protocol §pair_buckets_angstrom.
BUCKET_DEAD_MAX: Final[float] = 0.005
BUCKET_WEAK_MAX: Final[float] = 0.01
BUCKET_MEDIUM_MAX: Final[float] = 0.02
BUCKETS: Final[tuple[str, ...]] = ("dead", "weak", "medium", "strong")


# ---------------------------------------------------------------------------
# Prediction pass
# ---------------------------------------------------------------------------

def predict_pool(
    model: torch.nn.Module,
    pool_samples: Sequence[Mapping],
    device,
    *,
    config: Mapping,
    fold: int,
    d0: float,
    split_path: str,
    anchor: str = "on",
    batch_size: int = 32,
) -> list[dict]:
    """One deterministic forward pass over a Lane B dense pool.

    ``pool_samples`` are Lane B dense rows (``{"sample_id", "q_weight"}``,
    e.g. ``load_lane_b_pools(...)["inner_val_dense"]``).  Returns one
    record per frame with key atoms, mirroring the V5.1a OOF cache shape:
    ``cluster`` (reaction cluster id — same id the trainer aggregates
    on), ``reaction_id``, ``branch``, ``near_ts``, ``pos`` [n,3] CURRENT
    frame, frame-local ``key_rows``, frame-local key pairs (only columns
    with ``pair_edit != 0``, node-axis segmentation identical to
    ``g53_sample_enumeration``), their signed ``pair_edit`` values, the
    RAW model displacement ``u`` [m,3] (unbounded — NO ``d0`` clipping,
    NO calibration scale applied here) and teacher ``y`` [m,3].

    The caller owns the model; this function flips it to eval mode and
    restores the previous mode on exit.
    """
    device = torch.device(device) if not isinstance(device, torch.device) \
        else device
    view, collate = load_lane_b_view(
        config=config, samples=list(pool_samples), fold=int(fold), d0=float(d0),
        anchor=str(anchor), split_path=str(split_path), device=device)
    was_training = bool(model.training)
    model.train(False)
    records: list[dict] = []
    n_done = 0
    with torch.no_grad():
        for i in range(0, len(view), int(batch_size)):
            chunk = [view[j] for j in
                     range(i, min(i + int(batch_size), len(view)))]
            if not chunk:
                continue
            batch = collate(chunk)
            u, _z = model(batch)
            ptr = batch.ptr.cpu().numpy()
            key_ptr = batch.key_ptr.cpu().numpy()
            pos_np = batch.pos.cpu().numpy()
            key_index = batch.key_index.cpu().numpy()
            pair_index = batch.pair_index.cpu().numpy()
            pair_edit = batch.pair_edit.cpu().numpy()
            u_np = u.float().cpu().numpy()
            y_np = batch.y_teacher.cpu().numpy()
            near = (batch.near_ts.cpu().numpy()
                    if batch.near_ts is not None
                    else np.ones(len(key_ptr) - 1, dtype=np.float64))
            # trainer evaluate() convention: cluster == batch.reaction_ids
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
                    if n_lo <= a < n_hi and n_lo <= c < n_hi and \
                            float(pair_edit[col]) != 0.0:
                        k_pairs.append((a - n_lo, c - n_lo))
                        k_edit.append(float(pair_edit[col]))
                records.append({
                    "cluster": str(groups[b]),
                    "reaction_id": str(groups[b]),
                    "branch": int(branches[b]),
                    "near_ts": float(near[b]),
                    "pos": pos_np[n_lo:n_hi].astype(np.float64),
                    "key_rows": key_rows,
                    "pairs": k_pairs,
                    "pair_edit": k_edit,
                    "u": u_np[k_lo:k_hi].astype(np.float64),
                    "y": y_np[k_lo:k_hi].astype(np.float64),
                })
            n_done += len(chunk)
    if was_training:
        model.train(True)
    return records


# ---------------------------------------------------------------------------
# Canonical metrics (trainer-identical path)
# ---------------------------------------------------------------------------

def canonical_metrics(
    records: Sequence[Mapping],
    *,
    tau_effective: float = EFFECTIVE_ATOM_TAU,
    sign_threshold: float = SIGN_PAIR_THRESHOLD,
    sre_min_total_change: float = SRE_MIN_TOTAL_CHANGE,
) -> dict:
    """Per-cluster + aggregate metrics identical to the trainer inner-val rows.

    One code path: displacement metrics via the trainer's
    :func:`compute_cluster_metrics` (raw ``u``, classic baseline absent →
    ``rmse_over_classic`` is None) and derived-chemistry metrics via the
    FROZEN :func:`evaluate_g53`.  The g53 scalars
    (``sre_cluster_median``, ``sign_accuracy_cluster_median``,
    ``dead_zone_fraction``, ``n_*`` counts) are merged into ``aggregate``
    exactly like ``KeyAtomBCTrainer._record_eval`` / ``_write_metrics``
    do for ``metrics.json`` ``raw`` rows.
    """
    per_sample: list[dict] = []
    g53_samples: list[dict] = []
    for rec in records:
        cluster = str(rec.get("cluster", rec.get("group_id", "unknown")))
        per_sample.append({
            "group_id": cluster,
            "u": np.asarray(rec["u"], dtype=np.float64),
            "y": np.asarray(rec["y"], dtype=np.float64),
            "u_classic": None,
            "near_ts": float(rec.get("near_ts", 1.0)),
        })
        g53_samples.append({
            "group_id": cluster,
            "reaction_id": str(rec.get("reaction_id", cluster)),
            "branch": int(rec.get("branch", 0)),
            "pos": np.asarray(rec["pos"], dtype=np.float64),
            "key_rows": [int(r) for r in rec["key_rows"]],
            "pairs": [(int(p[0]), int(p[1])) for p in rec["pairs"]],
            "u": np.asarray(rec["u"], dtype=np.float64),
            "y": np.asarray(rec["y"], dtype=np.float64),
        })
    per_cluster, aggregate = compute_cluster_metrics(
        per_sample, soft=SOFT_COS_NORM, tau=float(tau_effective),
        near_ts_cut=NEAR_TS_CUT)
    g53 = evaluate_g53(
        g53_samples, sign_threshold=float(sign_threshold),
        sre_min_total_change=float(sre_min_total_change))
    g53_agg = g53["aggregate"]
    merged = dict(aggregate)
    for key in ("sre_cluster_median", "sign_accuracy_cluster_median",
                "dead_zone_fraction", "n_sign_pairs", "n_sign_pairs_correct",
                "n_active_pairs", "n_dead_pairs", "n_sre_frames"):
        if key in g53_agg:
            merged[key] = g53_agg[key]
    return {
        "schema_version": SCHEMA_VERSION,
        "per_cluster": per_cluster,
        "aggregate": merged,
        "g53": {"aggregate": g53_agg,
                "per_cluster_sre": g53["per_cluster_sre"],
                "per_cluster_sign_accuracy":
                    g53["per_cluster_sign_accuracy"]},
    }


# ---------------------------------------------------------------------------
# Post-hoc global scalar calibration (protocol §calibration)
# ---------------------------------------------------------------------------

def scale_solution(
    records: Sequence[Mapping],
    weight: str = "all",
) -> float:
    """Analytic displacement-MSE-optimal scalar ``s*`` (clipped ≥ 0).

    ``s* = Σᵢ wᵢ⟨uᵢ,yᵢ⟩ / Σᵢ wᵢ‖uᵢ‖²`` over all key atoms
    (``weight="all"``, primary — matches the RMSE/zero denominator) or
    effective atoms only (``weight="effective"``, ‖y‖ > 0.02 Å,
    secondary diagnostic per protocol §calibration.secondary_weights).
    """
    if weight not in ("all", "effective"):
        raise ValueError(
            f"weight must be 'all' or 'effective', got {weight!r}")
    num = 0.0
    den = 0.0
    for rec in records:
        u = np.asarray(rec["u"], dtype=np.float64)
        y = np.asarray(rec["y"], dtype=np.float64)
        if weight == "effective":
            m = np.linalg.norm(y, axis=1) > EFFECTIVE_ATOM_TAU
            u, y = u[m], y[m]
            if u.size == 0:
                continue
        num += float(np.sum(u * y))
        den += float(np.sum(u * u))
    if den <= 0.0:
        return 0.0
    return float(max(num / den, 0.0))


def scale_diagnostics(records: Sequence[Mapping]) -> dict:
    """Fitted ``s*`` (both weightings) + the FORBIDDEN-as-fit diagnostic
    ``1/alpha_median`` (protocol §calibration.forbidden: report-for-
    comparison only, never the fitted scale)."""
    alpha = canonical_metrics(records)["aggregate"].get("alpha_median")
    return {
        "s_star_all": scale_solution(records, weight="all"),
        "s_star_effective": scale_solution(records, weight="effective"),
        "alpha_median": alpha,
        "one_over_alpha_median": (1.0 / alpha
                                  if alpha is not None and alpha > 0 else None),
    }


def apply_scale(records: Sequence[Mapping], s: float) -> list[dict]:
    """Return copies of ``records`` with ``u → s·u`` (records unchanged)."""
    s = float(s)
    out: list[dict] = []
    for rec in records:
        new = dict(rec)
        new["u"] = np.asarray(rec["u"], dtype=np.float64) * s
        out.append(new)
    return out


# ---------------------------------------------------------------------------
# Exact-Δd pair buckets (protocol §pair_buckets_angstrom)
# ---------------------------------------------------------------------------

def pair_bucket(abs_dd: float) -> str:
    """Bucket for |Δd_true| with the frozen edge policy: half-open
    [lo, hi); a pair exactly on an edge belongs to the STRONGER bucket
    (|dd| = 0.005 → weak i.e. ACTIVE; |dd| = 0.02 → strong)."""
    ad = float(abs_dd)
    if ad < BUCKET_DEAD_MAX:
        return "dead"
    if ad < BUCKET_WEAK_MAX:
        return "weak"
    if ad < BUCKET_MEDIUM_MAX:
        return "medium"
    return "strong"


def _collect_pair_deltas(records: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
    """Exact Δd for teacher (u→y) and prediction over all recorded pairs.

    ``Δd = ‖(pos_i+u_i)−(pos_j+u_j)‖ − ‖pos_i−pos_j‖`` computed via the
    frozen :func:`exact_delta_d` (identical frame and key-row resolution
    as G5.3).  Records already carry only ``pair_edit != 0`` columns;
    a defensive re-filter is applied when ``pair_edit`` is present.
    """
    dd_true_parts: list[np.ndarray] = []
    dd_pred_parts: list[np.ndarray] = []
    for rec in records:
        pairs = [(int(p[0]), int(p[1])) for p in rec["pairs"]]
        edits = rec.get("pair_edit")
        if edits is not None and len(edits) == len(pairs):
            keep = [(p, e) for p, e in zip(pairs, edits) if float(e) != 0.0]
            if not keep:
                continue
            pairs = [p for p, _e in keep]
        if not pairs:
            continue
        pos = np.asarray(rec["pos"], dtype=np.float64)
        key_rows = [int(r) for r in rec["key_rows"]]
        u = np.asarray(rec["u"], dtype=np.float64)
        y = np.asarray(rec["y"], dtype=np.float64)
        dd_pred_parts.append(exact_delta_d(pos, key_rows, pairs, u))
        dd_true_parts.append(exact_delta_d(pos, key_rows, pairs, y))
    if not dd_true_parts:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.concatenate(dd_true_parts), np.concatenate(dd_pred_parts)


def strong_pair_eval(
    records: Sequence[Mapping],
    threshold: float = BUCKET_MEDIUM_MAX,
    sign_threshold: float = SIGN_PAIR_THRESHOLD,
) -> dict:
    """Per-bucket statistics over exact-Δd key pairs (protocol §pair_buckets).

    Buckets by |Δd_true| (``threshold`` is the strong lower edge, kept for
    signature compatibility; the other edges are the frozen protocol
    constants).  Per bucket:

    * ``n_pairs``;
    * ``sign_accuracy`` — sign(Δd_pred) == sign(Δd_true) AND
      |Δd_pred| ≥ sign_threshold; ABSTENTION (|Δd_pred| < sign_threshold)
      counts as INCORRECT (never free);
    * ``sign_accuracy_where_predicted`` — accuracy restricted to pairs
      the model actually moved (|Δd_pred| ≥ sign_threshold);
    * ``abstention_rate`` — fraction with |Δd_pred| < sign_threshold;
    * ``amplitude_ratio_mean`` / ``amplitude_ratio_median`` —
      |Δd_pred|/|Δd_true| per pair (pairs with |Δd_true| ≤ 1e-12
      excluded, ``n_amplitude_ratio`` reports the kept count);
    * ``residual_median`` / ``residual_q75`` — |Δd_pred − Δd_true|.
    """
    dd_true, dd_pred = _collect_pair_deltas(records)
    abs_t = np.abs(dd_true)
    abs_p = np.abs(dd_pred)
    bucket_of = np.array([pair_bucket(a) for a in abs_t], dtype=object)
    predicted = abs_p >= float(sign_threshold)
    correct = (np.sign(dd_pred) == np.sign(dd_true)) & predicted
    residual = np.abs(dd_pred - dd_true)
    per_bucket: dict[str, dict] = {}
    for name in BUCKETS:
        m = bucket_of == name
        n = int(m.sum())
        entry: dict = {
            "n_pairs": n,
            "sign_accuracy": None,
            "sign_accuracy_where_predicted": None,
            "abstention_rate": None,
            "amplitude_ratio_mean": None,
            "amplitude_ratio_median": None,
            "n_amplitude_ratio": 0,
            "residual_median": None,
            "residual_q75": None,
        }
        if n > 0:
            entry["sign_accuracy"] = float(correct[m].mean())
            pred_m = m & predicted
            n_pred = int(pred_m.sum())
            entry["sign_accuracy_where_predicted"] = (
                float(correct[pred_m].mean()) if n_pred > 0 else None)
            entry["abstention_rate"] = float((~predicted[m]).mean())
            amp_mask = m & (abs_t > 1e-12)
            n_amp = int(amp_mask.sum())
            entry["n_amplitude_ratio"] = n_amp
            if n_amp > 0:
                amp = abs_p[amp_mask] / abs_t[amp_mask]
                entry["amplitude_ratio_mean"] = float(amp.mean())
                entry["amplitude_ratio_median"] = float(np.median(amp))
            entry["residual_median"] = float(np.median(residual[m]))
            entry["residual_q75"] = float(np.quantile(residual[m], 0.75))
        per_bucket[name] = entry
    return {
        "schema_version": SCHEMA_VERSION,
        "thresholds": {
            "strong_lower_angstrom": float(threshold),
            "sign_threshold_angstrom": float(sign_threshold),
            "bucket_edges_angstrom": {
                "dead": f"|dd| < {BUCKET_DEAD_MAX}",
                "weak": f"{BUCKET_DEAD_MAX} <= |dd| < {BUCKET_WEAK_MAX}",
                "medium": f"{BUCKET_WEAK_MAX} <= |dd| < {BUCKET_MEDIUM_MAX}",
                "strong": f"|dd| >= {BUCKET_MEDIUM_MAX}",
            },
            "edge_policy": ("half-open [lo, hi); a pair exactly on an edge "
                            "belongs to the stronger bucket"),
            "abstention_policy": ("|Δd_pred| < sign_threshold counts "
                                  "incorrect in sign_accuracy"),
        },
        "n_pairs_total": int(len(dd_true)),
        "per_bucket": per_bucket,
    }


def strong_sign_accuracy(records: Sequence[Mapping],
                         sign_threshold: float = SIGN_PAIR_THRESHOLD) -> float | None:
    """Sign accuracy on strong-bucket pairs (|Δd_true| ≥ 0.02 Å) with
    abstention = incorrect — the protocol §calibration.reported_metrics
    ``strong_sign_accuracy``."""
    return strong_pair_eval(records, sign_threshold=sign_threshold)[
        "per_bucket"]["strong"]["sign_accuracy"]


# ---------------------------------------------------------------------------
# Cluster-level paired bootstrap (protocol §calibration.uncertainty)
# ---------------------------------------------------------------------------

def cluster_paired_bootstrap(
    records_a: Sequence[Mapping],
    records_b: Sequence[Mapping],
    metric_fn: Callable[[Sequence[Mapping]], float | None],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Cluster-level paired bootstrap of ``metric_fn(a) − metric_fn(b)``.

    Per-cluster scalar values are computed once by applying ``metric_fn``
    to each cluster's records; each draw resamples the shared cluster ids
    WITH replacement (same resample for both sets — paired) and takes the
    median over the resampled clusters, matching the canonical
    median-over-clusters aggregation (exact for ``rmse_over_zero`` and
    ``sre_cluster_median`` from :func:`canonical_metrics`).  Returns the
    median delta plus the 2.5/97.5 percentiles of the replay
    distribution and the per-cluster paired deltas.
    """
    def _cluster_of(rec: Mapping) -> str:
        return str(rec.get("cluster", rec.get("group_id", "unknown")))

    clusters = sorted({_cluster_of(r) for r in records_a}
                      | {_cluster_of(r) for r in records_b})
    if not clusters:
        raise ValueError("cluster_paired_bootstrap: empty record sets")

    def _values(records: Sequence[Mapping]) -> dict[str, float]:
        by_cluster: dict[str, list[Mapping]] = {}
        for rec in records:
            by_cluster.setdefault(_cluster_of(rec), []).append(rec)
        out: dict[str, float] = {}
        for cid in clusters:
            sub = by_cluster.get(cid)
            if not sub:
                continue
            v = metric_fn(sub)
            if v is not None and math.isfinite(float(v)):
                out[cid] = float(v)
        return out

    val_a = _values(records_a)
    val_b = _values(records_b)
    shared = [c for c in clusters if c in val_a and c in val_b]
    if not shared:
        raise ValueError(
            "cluster_paired_bootstrap: no clusters with finite metric in both sets")
    arr_a = np.array([val_a[c] for c in shared], dtype=np.float64)
    arr_b = np.array([val_b[c] for c in shared], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n = len(shared)
    deltas = np.empty(int(draws), dtype=np.float64)
    for k in range(int(draws)):
        idx = rng.integers(0, n, size=n)
        deltas[k] = float(np.median(arr_a[idx]) - np.median(arr_b[idx]))
    per_cluster_deltas = {
        c: float(val_a[c] - val_b[c]) for c in shared}
    return {
        "schema_version": SCHEMA_VERSION,
        "draws": int(draws),
        "seed": int(seed),
        "n_clusters": n,
        "cluster_ids": list(shared),
        "point_delta": float(np.median(arr_a) - np.median(arr_b)),
        "median_delta": float(np.median(deltas)),
        "ci_lo_2_5": float(np.quantile(deltas, 0.025)),
        "ci_hi_97_5": float(np.quantile(deltas, 0.975)),
        "per_cluster_deltas": per_cluster_deltas,
    }
