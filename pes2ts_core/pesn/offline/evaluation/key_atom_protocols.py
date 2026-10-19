"""Lane D — V5 sprint evaluation gates G5.2/G5.3/G5.4 + sealed guard (plan §6).

Implements the sprint plan §6 hard gates and ADR-0026 §5/§7 for the V5
phase-1 campaign (``V5_SPRINT``):

* **G5.2 key-vector gate** — per-reaction-cluster MAE/RMSE with ratios vs
  the zero action and the M0 Classic baseline (undefined denominators
  <1e-6 Å are listed separately, NEVER coerced to 0), cluster median/Q75,
  % clusters better than zero, effective-atom (‖y‖>0.02 Å) cosine medians
  (overall and the near-TS ≤0.20 slice), and cluster-level bootstrap 95%
  CIs (2000 draws, seed=20261018).  Aggregation order is normative:
  frame key-atom mean → branch mean → reaction two-branch equal weight →
  cluster mean; cross-cluster summaries are median/IQR.
* **G5.3 derived-chemistry gate** — EXACT finite-step distance changes
  ``Δd_l(u)=‖r_l+u_j−u_i‖−‖r_l‖`` with ``r_l`` from the CURRENT frame over
  key pairs only; sign accuracy on teacher-significant pairs
  (|Δd_l(y)|≥0.005 Å) where a predicted |Δd_l(u)|<0.005 Å counts as
  INCORRECT in the main denominator (abstention never hides); SRE =
  0.5·Σ|p̂−p| over multi-bond frames (≥2 pairs and total teacher change
  ≥0.01 Å; a fully-zero prediction gets SRE=1); the first-order pair
  projection J_l(u)=e_l·(u_j−u_i) vs exact Δd deviation is reported so
  short-bond / large-step nonlinearity cannot be masked by the loss.
* **G5.4 static executability gate** — strict-copy key-atom patch (non-key
  coordinates must stay bit-identical; ``torch.equal`` semantics), clash
  detection at 0.75·(rcov_i+rcov_j) INCLUDING bonded pairs (Cordero 2008
  table; only pre-registered exceptions — none this sprint), protected
  edges ``(G_start bonds)\\K`` (including key–non-key X–H) with
  stretch >1.35·d_eq or shorten <0.70·d_eq (d_eq from the deployment
  START frame per branch), per-cluster violation-frame fractions, total
  and NEW-vs-input rates, one-sided 95% cluster-bootstrap upper bounds,
  and the refusal rate.  :func:`oracle_static_ceiling` re-runs the same
  protocol with u=teacher and returns a typed ``ORACLE_STATIC_FAILED``
  when the ACTION REPRESENTATION (not the network) lacks static
  feasibility.  The follower counterfactual (rigidly translate a terminal
  H bonded to a unique key heavy atom) is a DIAGNOSTIC marked
  ``contract_counterfactual`` — never counted in the G5.4 pass.
* **Sealed-eval guard** — :func:`sealed_eval_request` returns a typed
  REFUSED decision (``sealed_confirm_not_opened``) unless the crossfit is
  complete for all 15 runs AND model/config/protocol hashes are frozen AND
  the evaluation protocol explicitly allows unsealing.  This sprint never
  unseals (plan §5.5 / ADR-0026 §4): the campaign always calls with
  ``allow_unseal=False`` and the refusal echoes the registration.
* **Gate report writer** — ``gate_report.json`` + ``gate_report.md``
  (schema ``pesn_v5_gate_report_v1``) distinguishing measured / planned /
  not_executed / censored; artifact rows carry schema/config/source/parent
  sha256 plus the four-layer capability status; absent QC is written as
  ``QC not scheduled`` (never a fabricated 0).

Truth-guard safe by construction: pure numpy/torch numerics, no
subprocess, no dynamic imports, no ``model.eval()`` attribute calls.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.features.geometry import COVALENT_RADII

# ---------------------------------------------------------------------------
# Schemas / frozen constants (plan §6; ADR-0026 §5)
# ---------------------------------------------------------------------------

GATE_REPORT_SCHEMA: Final[str] = "pesn_v5_gate_report_v1"
G52_EVAL_SCHEMA: Final[str] = "pesn_v5_g52_eval_v1"
G53_EVAL_SCHEMA: Final[str] = "pesn_v5_g53_eval_v1"
G54_EVAL_SCHEMA: Final[str] = "pesn_v5_g54_eval_v1"
SEALED_DECISION_SCHEMA: Final[str] = "pesn_v5_sealed_eval_v1"

#: Plan §6 header: cluster bootstrap, never frame-level CIs.
BOOTSTRAP_DRAWS: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 20261018

#: Plan §6 G5.2: effective-atom gate and near-TS slice.
EFFECTIVE_ATOM_TAU: Final[float] = 0.02
NEAR_TS_CUT: Final[float] = 0.20
#: Soft-norm epsilon of the cosine (plan §4 cos_soft_norm).
COS_SOFT_NORM: Final[float] = 1e-4

#: Plan §6 G5.2 thresholds (engineering gates, frozen before unsealing).
G52_MAE_OVER_ZERO_MAX: Final[float] = 0.80
G52_RMSE_OVER_ZERO_MAX: Final[float] = 0.80
G52_RMSE_OVER_CLASSIC_MAX: Final[float] = 0.90
G52_Q75_RMSE_OVER_ZERO_MAX: Final[float] = 1.00
G52_BETTER_THAN_ZERO_FRACTION_MIN: Final[float] = 0.70
G52_CLASSIC_IMPROVEMENT_LOWER_MIN: Final[float] = 0.0
G52_COS_OVERALL_MIN: Final[float] = 0.60
G52_COS_NEAR_TS_MIN: Final[float] = 0.50
G52_MIN_CLUSTERS: Final[int] = 20
G52_MIN_NEAR_TS_CLUSTERS: Final[int] = 10

#: Plan §6 G5.3 thresholds.
G53_SIGN_MEDIAN_MIN: Final[float] = 0.90
G53_SIGN_Q25_MIN: Final[float] = 0.80
G53_SRE_MEDIAN_MAX: Final[float] = 0.15
G53_SRE_Q75_MAX: Final[float] = 0.25
G53_SIGN_PAIR_THRESHOLD: Final[float] = 0.005
G53_SRE_MIN_TOTAL_CHANGE: Final[float] = 0.01
#: V5.1a (plan §5.1 G5.3 死区弃权率 <10%): fraction of teacher-active
#: pairs (|Δd_true| ≥ 0.005 Å) whose |Δd_pred| stays inside the dead
#: zone (|Δd_pred| < 0.005 Å).
G53_DEAD_ZONE_MAX: Final[float] = 0.10

#: Plan §6 G5.4 thresholds and geometry factors.
G54_CLASH_FACTOR: Final[float] = 0.75
G54_STRETCH_FACTOR: Final[float] = 1.35
G54_SHORTEN_FACTOR: Final[float] = 0.70
G54_NEW_CLASH_MAX: Final[float] = 0.005
G54_NEW_PROTECTED_MAX: Final[float] = 0.01
G54_UPPER_BOUND_MAX: Final[float] = 0.02
G54_CLUSTER_MAX: Final[float] = 0.05
G54_REFUSAL_MAX: Final[float] = 0.05
#: Pre-registered bonded-short-contact exceptions — NONE this sprint
#: (plan §6 G5.4: exceptions can only be registered before test access).
PREREGISTERED_CLASH_EXCEPTIONS: Final[frozenset] = frozenset()

#: Undefined-denominator cutoff (plan §6 G5.2).
DENOM_EPS: Final[float] = 1e-6

#: Four-layer capability statuses (ADR-0026 §7 vocabulary).
CAPABILITY_LAYERS: Final[tuple[str, ...]] = (
    "code", "math_contract", "real_data", "sealed")
CAPABILITY_DEFAULTS: Final[dict] = {
    "code": "implemented",
    "math_contract": "fixture_passed",
    "real_data": "unknown",
    "sealed": "unknown",
}

#: Gate-report section statuses (plan §8: distinguish measured / planned /
#: not executed / censored).
SECTION_STATUSES: Final[tuple[str, ...]] = (
    "measured", "planned", "not_executed", "censored")

#: QC placeholder when no quantum-chemistry work was scheduled (plan §8 —
#: never fabricate a zero cost).
QC_NOT_SCHEDULED: Final[str] = "QC not scheduled"


class KeyAtomProtocolError(RuntimeError):
    """Typed gate-protocol refusal (machine-readable code)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise KeyAtomProtocolError(code, detail)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _median(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _quantile(values: Sequence[float], q: float) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.quantile(np.asarray(vals, dtype=np.float64), q))


def _ratio(numerator: float, denominator: float) -> float | None:
    """Plan §6 G5.2: denominators <1e-6 Å are undefined, never 0."""
    if not math.isfinite(denominator) or abs(denominator) < DENOM_EPS:
        return None
    return float(numerator / denominator)


def _soft_cosine_matrix(u: np.ndarray, y: np.ndarray,
                        soft: float = COS_SOFT_NORM) -> np.ndarray:
    """Per-atom soft-norm cosine (plan §4), vectorized over ``[m, 3]``."""
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    num = np.sum(u * y, axis=-1)
    den = np.sqrt(np.sum(u * u, axis=-1) + soft * soft) * np.sqrt(
        np.sum(y * y, axis=-1) + soft * soft)
    return num / np.maximum(den, 1e-12)


# ---------------------------------------------------------------------------
# Hierarchical aggregation (plan §6 header — NORMATIVE order)
# ---------------------------------------------------------------------------


def hierarchical_means(
    frame_values: np.ndarray,
    cluster_of: Sequence[str],
    reaction_of: Sequence[str],
    branch_of: Sequence[int],
) -> dict[str, float]:
    """Aggregate per-frame scalars in the plan §6 order.

    frame → branch mean → reaction two-branch EQUAL weight → cluster mean.
    ``frame_values[i]`` belongs to ``(cluster_of[i], reaction_of[i],
    branch_of[i])``.  Returns ``{cluster_id: value}``.
    """
    _require(len(frame_values) == len(cluster_of) == len(reaction_of)
             == len(branch_of), "HIERARCHY_SHAPE",
             "frame values and identity arrays must have equal length")
    branch_acc: dict[tuple[str, str, int], list[float]] = {}
    for value, cluster, reaction, branch in zip(
            np.asarray(frame_values, dtype=np.float64).tolist(),
            cluster_of, reaction_of, branch_of):
        if not math.isfinite(float(value)):
            continue
        branch_acc.setdefault(
            (str(cluster), str(reaction), int(branch)), []).append(float(value))
    reaction_acc: dict[tuple[str, str], list[float]] = {}
    for (cluster, reaction, _branch), values in branch_acc.items():
        reaction_acc.setdefault((cluster, reaction), []).append(
            float(np.mean(values)))
    cluster_acc: dict[str, list[float]] = {}
    for (cluster, _reaction), values in reaction_acc.items():
        cluster_acc.setdefault(cluster, []).append(float(np.mean(values)))
    return {cluster: float(np.mean(values))
            for cluster, values in sorted(cluster_acc.items())}


# ---------------------------------------------------------------------------
# G5.2 — key-vector gate
# ---------------------------------------------------------------------------


def _as_prediction_blocks(predictions) -> dict:
    """Normalize the predictions argument to ``{"raw": ..., "bounded": ...}``
    (a bare array becomes raw-only and the bounded block mirrors raw so
    downstream code stays uniform).  Values may be stacked ``[S,m,3]``
    arrays OR ragged sequences of per-sample ``[m,3]`` arrays (molecules
    with different key-atom counts)."""
    if isinstance(predictions, Mapping):
        blocks = {str(k): v for k, v in predictions.items()}
        _require(bool(blocks), "G52_PREDICTIONS_EMPTY", "no prediction blocks")
        if "raw" not in blocks:
            blocks["raw"] = next(iter(blocks.values()))
        if "bounded" not in blocks:
            blocks["bounded"] = blocks["raw"]
        return blocks
    return {"raw": predictions, "bounded": predictions}


def _coerce_samples(x) -> list[np.ndarray]:
    """[S,m,3] array OR ragged sequence -> list of [m,3] float64 arrays."""
    if isinstance(x, np.ndarray) and x.ndim == 3:
        return [np.asarray(x[s], dtype=np.float64)
                for s in range(x.shape[0])]
    seq = list(x)
    out = []
    for s in seq:
        a = np.asarray(s, dtype=np.float64)
        _require(a.ndim == 2 and a.shape[-1] == 3, "G52_PREDICTION_SHAPE",
                 f"per-sample block must be [m,3], got {a.shape}")
        out.append(a)
    return out


def _g52_block(
    u_by_sample: np.ndarray,
    y: np.ndarray,
    clusters: Sequence[str],
    reactions: Sequence[str],
    branches: Sequence[int],
    classic: np.ndarray | None,
    near_ts: np.ndarray,
    *,
    tau: float,
    near_ts_cut: float,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict:
    """One raw/bounded G5.2 metric block (plan §6 G5.2)."""
    u_list = _coerce_samples(u_by_sample)
    y_list = _coerce_samples(y)
    n_samples = len(y_list)
    _require(len(u_list) == n_samples, "G52_SHAPE_MISMATCH",
             f"teacher samples {n_samples} vs prediction samples "
             f"{len(u_list)}")
    _require(len(clusters) == n_samples, "G52_SHAPE_MISMATCH",
             "cluster ids must align with samples")

    # --- per-frame reductions over key atoms (ragged-safe) ------------------
    mae_frame = np.asarray([
        np.linalg.norm(u - yy, axis=-1).mean() for u, yy in
        zip(u_list, y_list)])
    mse_frame = np.asarray([
        (np.linalg.norm(u - yy, axis=-1) ** 2).mean() for u, yy in
        zip(u_list, y_list)])
    y_norms = [np.linalg.norm(yy, axis=-1) for yy in y_list]
    mae_zero_frame = np.asarray([yn.mean() for yn in y_norms])
    mse_zero_frame = np.asarray([(yn ** 2).mean() for yn in y_norms])

    # effective-atom cosines (pooled over atoms, plan §6 "cos 中位全场")
    cos_eff: list[float] = []
    cos_near: list[float] = []
    for s in range(n_samples):
        eff = y_norms[s] > tau
        if not bool(eff.any()):
            continue
        values = _soft_cosine_matrix(u_list[s][eff], y_list[s][eff]).tolist()
        cos_eff.extend(values)
        if float(near_ts[s]) <= near_ts_cut:
            cos_near.extend(values)

    def _clusterize(mae_f, mse_f, mae_z, mse_z):
        c_mae = hierarchical_means(mae_f, clusters, reactions, branches)
        c_mse = hierarchical_means(mse_f, clusters, reactions, branches)
        c_mae_zero = hierarchical_means(mae_z, clusters, reactions, branches)
        c_mse_zero = hierarchical_means(mse_z, clusters, reactions, branches)
        per_cluster: dict[str, dict] = {}
        for cluster in sorted(c_mse):
            rmse = math.sqrt(c_mse[cluster])
            rmse_zero = math.sqrt(c_mse_zero[cluster])
            entry = {
                "mae": c_mae[cluster],
                "rmse": rmse,
                "mae_over_zero": _ratio(c_mae[cluster], c_mae_zero[cluster]),
                "rmse_over_zero": _ratio(rmse, rmse_zero),
                "rmse_zero": rmse_zero,
            }
            per_cluster[cluster] = entry
        return per_cluster

    per_cluster = _clusterize(mae_frame, mse_frame, mae_zero_frame,
                              mse_zero_frame)

    # --- Classic ratios (per cluster; undefined listed separately) ---------
    undefined_zero: list[str] = []
    undefined_classic: list[str] = []
    better_than_zero = 0
    classic_defined = 0
    classic_improvements: list[float] = []
    if classic is not None:
        cls_list = _coerce_samples(classic)
        _require(len(cls_list) == n_samples, "G52_SHAPE_MISMATCH",
                 "classic predictions must align with model predictions")
        mse_cls_frame = np.asarray([
            (np.linalg.norm(c - yy, axis=-1) ** 2).mean() for c, yy in
            zip(cls_list, y_list)])
        c_mse_cls = hierarchical_means(mse_cls_frame, clusters, reactions,
                                       branches)
    else:
        c_mse_cls = {}
    for cluster, entry in per_cluster.items():
        if entry["rmse_over_zero"] is None:
            undefined_zero.append(cluster)
        elif entry["rmse_over_zero"] < 1.0:
            better_than_zero += 1
        if cluster in c_mse_cls:
            rmse_cls = math.sqrt(c_mse_cls[cluster])
            ratio = _ratio(entry["rmse"], rmse_cls)
            entry["rmse_classic"] = rmse_cls
            entry["rmse_over_classic"] = ratio
            if ratio is None:
                undefined_classic.append(cluster)
            else:
                classic_defined += 1
                classic_improvements.append(1.0 - ratio)
        else:
            entry["rmse_classic"] = None
            entry["rmse_over_classic"] = None
            undefined_classic.append(cluster)

    n_defined_zero = sum(1 for e in per_cluster.values()
                         if e["rmse_over_zero"] is not None)
    aggregate = {
        "n_clusters": len(per_cluster),
        "n_clusters_rmse_over_zero_defined": n_defined_zero,
        "median_mae": _median([e["mae"] for e in per_cluster.values()]),
        "median_rmse": _median([e["rmse"] for e in per_cluster.values()]),
        "median_mae_over_zero": _median(
            [e["mae_over_zero"] for e in per_cluster.values()]),
        "median_rmse_over_zero": _median(
            [e["rmse_over_zero"] for e in per_cluster.values()]),
        "q75_rmse_over_zero": _quantile(
            [e["rmse_over_zero"] for e in per_cluster.values()], 0.75),
        "fraction_better_than_zero": (
            better_than_zero / n_defined_zero if n_defined_zero else None),
        "n_better_than_zero": better_than_zero,
        "median_rmse_over_classic": _median(
            [e["rmse_over_classic"] for e in per_cluster.values()]),
        "n_clusters_rmse_over_classic_defined": classic_defined,
        "cos_eff_median": _median(cos_eff),
        "cos_eff_near_ts_median": _median(cos_near),
        "n_effective_atoms": len(cos_eff),
        "n_effective_atoms_near_ts": len(cos_near),
        "n_near_ts_clusters": len({
            str(c) for c, nt in zip(clusters, near_ts)
            if float(nt) <= near_ts_cut}),
    }

    # --- cluster-level bootstrap (2000 draws, seed 20261018) ----------------
    cluster_ids = sorted(per_cluster)
    rng = np.random.default_rng(bootstrap_seed)
    med_zero_samples: list[float] = []
    improvement_samples: list[float] = []
    if cluster_ids:
        ratios = np.asarray(
            [per_cluster[c]["rmse_over_zero"] if per_cluster[c]["rmse_over_zero"]
             is not None else np.nan for c in cluster_ids], dtype=np.float64)
        impr = np.asarray(
            [1.0 - per_cluster[c]["rmse_over_classic"]
             if per_cluster[c]["rmse_over_classic"] is not None else np.nan
             for c in cluster_ids], dtype=np.float64)
        n = len(cluster_ids)
        for _ in range(int(bootstrap_draws)):
            pick = rng.integers(0, n, size=n)
            sample_ratios = ratios[pick]
            sample_ratios = sample_ratios[np.isfinite(sample_ratios)]
            if sample_ratios.size:
                med_zero_samples.append(float(np.median(sample_ratios)))
            sample_impr = impr[pick]
            sample_impr = sample_impr[np.isfinite(sample_impr)]
            if sample_impr.size:
                improvement_samples.append(float(np.median(sample_impr)))
    bootstrap = {
        "draws": int(bootstrap_draws),
        "seed": int(bootstrap_seed),
        "unit": "reaction_cluster",
        "median_rmse_over_zero_ci95": (
            [_quantile(med_zero_samples, 0.025),
             _quantile(med_zero_samples, 0.975)]
            if med_zero_samples else [None, None]),
        "classic_improvement_median": _median(classic_improvements),
        "classic_improvement_lower95": (
            _quantile(improvement_samples, 0.025)
            if improvement_samples else None),
        "n_classic_improvement_defined": len(classic_improvements),
    }

    undefined = {"rmse_over_zero": sorted(undefined_zero),
                 "rmse_over_classic": sorted(undefined_classic),
                 "note": ("clusters with denominator <1e-6 Å are listed "
                          "here and excluded from ratio aggregates; they "
                          "are never coerced to 0 (plan §6 G5.2)")}

    # --- gate evaluation ----------------------------------------------------
    insufficient = (
        aggregate["n_clusters_rmse_over_zero_defined"] < G52_MIN_CLUSTERS
        or aggregate["n_near_ts_clusters"] < G52_MIN_NEAR_TS_CLUSTERS)
    checks = {
        "median_mae_over_zero_le_0.80": (
            aggregate["median_mae_over_zero"] is not None
            and aggregate["median_mae_over_zero"] <= G52_MAE_OVER_ZERO_MAX),
        "median_rmse_over_zero_le_0.80": (
            aggregate["median_rmse_over_zero"] is not None
            and aggregate["median_rmse_over_zero"] <= G52_RMSE_OVER_ZERO_MAX),
        "median_rmse_over_classic_le_0.90": (
            aggregate["median_rmse_over_classic"] is not None
            and aggregate["median_rmse_over_classic"]
            <= G52_RMSE_OVER_CLASSIC_MAX),
        "q75_rmse_over_zero_le_1.00": (
            aggregate["q75_rmse_over_zero"] is not None
            and aggregate["q75_rmse_over_zero"] <= G52_Q75_RMSE_OVER_ZERO_MAX),
        "fraction_better_than_zero_ge_0.70": (
            aggregate["fraction_better_than_zero"] is not None
            and aggregate["fraction_better_than_zero"]
            >= G52_BETTER_THAN_ZERO_FRACTION_MIN),
        "classic_improvement_lower95_gt_0": (
            bootstrap["classic_improvement_lower95"] is not None
            and bootstrap["classic_improvement_lower95"]
            > G52_CLASSIC_IMPROVEMENT_LOWER_MIN),
        "cos_overall_ge_0.60": (
            aggregate["cos_eff_median"] is not None
            and aggregate["cos_eff_median"] >= G52_COS_OVERALL_MIN),
        "cos_near_ts_ge_0.50": (
            aggregate["cos_eff_near_ts_median"] is not None
            and aggregate["cos_eff_near_ts_median"] >= G52_COS_NEAR_TS_MIN),
    }
    gate = {
        "checks": checks,
        "insufficient_clusters": insufficient,
        "pass": (not insufficient) and all(checks.values()),
        "thresholds": {
            "median_mae_over_zero_max": G52_MAE_OVER_ZERO_MAX,
            "median_rmse_over_zero_max": G52_RMSE_OVER_ZERO_MAX,
            "median_rmse_over_classic_max": G52_RMSE_OVER_CLASSIC_MAX,
            "q75_rmse_over_zero_max": G52_Q75_RMSE_OVER_ZERO_MAX,
            "fraction_better_than_zero_min": G52_BETTER_THAN_ZERO_FRACTION_MIN,
            "classic_improvement_lower_min": G52_CLASSIC_IMPROVEMENT_LOWER_MIN,
            "cos_overall_min": G52_COS_OVERALL_MIN,
            "cos_near_ts_min": G52_COS_NEAR_TS_MIN,
            "min_clusters": G52_MIN_CLUSTERS,
            "min_near_ts_clusters": G52_MIN_NEAR_TS_CLUSTERS,
            "effective_atom_tau_angstrom": tau,
            "near_ts_cut": near_ts_cut,
        },
    }
    return {
        "per_cluster": per_cluster,
        "aggregate": aggregate,
        "bootstrap": bootstrap,
        "undefined": undefined,
        "gate": gate,
    }


def evaluate_g52(
    predictions,
    teacher,
    groups,
    *,
    classic=None,
    reactions=None,
    branches=None,
    near_ts=None,
    tau: float = EFFECTIVE_ATOM_TAU,
    near_ts_cut: float = NEAR_TS_CUT,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """G5.2 key-vector gate evaluation (plan §6 G5.2; ADR-0026 §5).

    ``predictions`` is either a ``[S, m, 3]`` array (raw only) or a mapping
    with ``"raw"`` and ``"bounded"`` ``[S, m, 3]`` blocks — the sprint
    reports BOTH and the gate uses raw accuracy (plan §4).  ``teacher`` is
    ``y [S, m, 3]``; ``groups`` are reaction-cluster ids per sample;
    ``classic`` (optional) is the M0 baseline ``[S, m, 3]``; ``reactions``
    / ``branches`` enable the plan §6 hierarchical aggregation (defaulting
    to one reaction per sample on branch 0); ``near_ts`` is the remaining
    key-path arc fraction per sample (isolated-evaluator field, never a
    model input).

    Returns a dict with schema ``pesn_v5_g52_eval_v1``: ``per_cluster``,
    ``aggregate`` (cluster medians/IQR), ``bootstrap`` (cluster-level 95%
    CIs), ``undefined`` (listed denominators), ``gate`` and a ``bounded``
    twin block.
    """
    y = (teacher if isinstance(teacher, (list, tuple))
         else np.asarray(teacher, dtype=np.float64))
    if isinstance(y, np.ndarray):
        _require(y.ndim == 3 and y.shape[-1] == 3, "G52_TEACHER_SHAPE",
                 f"teacher must be [S,m,3], got {y.shape}")
    n_samples = len(y)
    blocks = _as_prediction_blocks(predictions)
    cluster_ids = [str(g) for g in groups]
    _require(len(cluster_ids) == n_samples, "G52_SHAPE_MISMATCH",
             "group ids must align with teacher samples")
    reaction_ids = ([str(r) for r in reactions] if reactions is not None
                    else [f"{c}::sample{i}" for i, c in enumerate(cluster_ids)])
    branch_ids = ([int(b) for b in branches] if branches is not None
                  else [0] * n_samples)
    near = (np.asarray(near_ts, dtype=np.float64) if near_ts is not None
            else np.ones(n_samples, dtype=np.float64))
    classic_array = classic  # ndarray [S,m,3] or ragged sequence; None ok

    raw = _g52_block(
        blocks["raw"], y, cluster_ids, reaction_ids, branch_ids,
        classic_array, near, tau=tau, near_ts_cut=near_ts_cut,
        bootstrap_draws=bootstrap_draws, bootstrap_seed=bootstrap_seed)
    bounded = _g52_block(
        blocks["bounded"], y, cluster_ids, reaction_ids, branch_ids,
        classic_array, near, tau=tau, near_ts_cut=near_ts_cut,
        bootstrap_draws=bootstrap_draws, bootstrap_seed=bootstrap_seed)
    return {
        "schema_version": G52_EVAL_SCHEMA,
        "aggregation_order": ("frame key-atom mean -> branch mean -> "
                              "reaction two-branch equal weight -> cluster "
                              "mean; cross-cluster median/IQR"),
        "raw": raw,
        "bounded": bounded,
        "gate": raw["gate"],
    }


# ---------------------------------------------------------------------------
# G5.3 — derived-chemistry gate (exact finite-step Δd)
# ---------------------------------------------------------------------------


def exact_delta_d(
    pos: np.ndarray,
    key_rows: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    u: np.ndarray,
) -> np.ndarray:
    """EXACT finite-step bond-distance change per key pair (plan §6 G5.3).

    ``Δd_l(u) = ‖r_l + u_j − u_i‖ − ‖r_l‖`` with ``r_l`` taken from the
    CURRENT frame ``pos`` (never the start frame) and ``(i, j)`` given as
    global atom rows resolved through ``key_rows`` into u-slot indices.
    """
    pos = np.asarray(pos, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    key = np.asarray(key_rows, dtype=np.int64)
    slot = {int(row): k for k, row in enumerate(key)}
    out = np.zeros(len(list(pairs)), dtype=np.float64)
    for l, (i, j) in enumerate(pairs):
        _require(int(i) in slot and int(j) in slot, "G53_PAIR_NOT_KEY",
                 f"pair ({i},{j}) has a non-key endpoint")
        si, sj = slot[int(i)], slot[int(j)]
        r = pos[int(j)] - pos[int(i)]
        out[l] = np.linalg.norm(r + u[sj] - u[si]) - np.linalg.norm(r)
    return out


def first_order_pair_projection(
    pos: np.ndarray,
    key_rows: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    u: np.ndarray,
) -> np.ndarray:
    """First-order J_l(u) = e_l·(u_j − u_i) with e_l from the CURRENT frame.

    The BC loss fits this first-order quantity (plan §4); G5.3 reports its
    deviation from the exact Δd so nonlinearity stays visible.
    """
    pos = np.asarray(pos, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    key = np.asarray(key_rows, dtype=np.int64)
    slot = {int(row): k for k, row in enumerate(key)}
    out = np.zeros(len(list(pairs)), dtype=np.float64)
    for l, (i, j) in enumerate(pairs):
        si, sj = slot[int(i)], slot[int(j)]
        r = pos[int(j)] - pos[int(i)]
        norm = float(np.linalg.norm(r))
        e = r / norm if norm > 1e-12 else np.zeros(3, dtype=np.float64)
        out[l] = float(e @ (u[sj] - u[si]))
    return out


def g53_sample_enumeration(
    *,
    pos: np.ndarray,
    ptr: np.ndarray,
    key_index: np.ndarray,
    key_ptr: np.ndarray,
    pair_index: np.ndarray,
    pair_edit: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    group_ids: Sequence[str],
    reaction_ids: Sequence[str] | None = None,
    branches: Sequence[int] | None = None,
) -> list[dict]:
    """Shared key-pair enumeration for G5.3 metrics (V5.1 plan §3.3/§3.4).

    Produces one ``evaluate_g53``-ready sample mapping per frame that has
    at least one changed key pair.  Pair semantics are the frozen V5
    sprint ``_static_diagnostics`` rule: a ``pair_index`` column belongs
    to frame ``b`` when BOTH global node-row endpoints fall inside the
    frame's node range ``[ptr[b], ptr[b+1])`` AND ``pair_edit != 0``;
    key rows and pair rows are re-based to the frame.  The trainer's
    inner-val SRE / dead-zone metrics and the campaign static diagnostics
    MUST share this helper so the derived-chemistry numbers stay
    comparable (plan §3.4: selection metric == gate metric).
    """
    pos = np.asarray(pos, dtype=np.float64)
    ptr = np.asarray(ptr, dtype=np.int64)
    key_index = np.asarray(key_index, dtype=np.int64)
    key_ptr = np.asarray(key_ptr, dtype=np.int64)
    pair_index = np.asarray(pair_index, dtype=np.int64)
    pair_edit = np.asarray(pair_edit, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_frames = len(ptr) - 1
    out: list[dict] = []
    for b in range(n_frames):
        n_lo, n_hi = int(ptr[b]), int(ptr[b + 1])
        k_lo, k_hi = int(key_ptr[b]), int(key_ptr[b + 1])
        if k_hi <= k_lo:
            continue
        key_rows = [int(v) - n_lo for v in key_index[k_lo:k_hi]]
        pairs: list[tuple[int, int]] = []
        for col in range(pair_index.shape[1]):
            a = int(pair_index[0, col])
            c = int(pair_index[1, col])
            if n_lo <= a < n_hi and n_lo <= c < n_hi and \
                    float(pair_edit[col]) != 0.0:
                pairs.append((a - n_lo, c - n_lo))
        if not pairs:
            continue
        out.append({
            "group_id": str(group_ids[b]),
            "reaction_id": str(
                reaction_ids[b] if reaction_ids is not None
                else group_ids[b]),
            "branch": int(branches[b]) if branches is not None else 0,
            "pos": pos[n_lo:n_hi],
            "key_rows": key_rows,
            "pairs": pairs,
            "u": u[k_lo:k_hi],
            "y": y[k_lo:k_hi],
        })
    return out


def evaluate_g53(
    samples: Sequence[Mapping],
    *,
    sign_threshold: float = G53_SIGN_PAIR_THRESHOLD,
    sre_min_total_change: float = G53_SRE_MIN_TOTAL_CHANGE,
) -> dict:
    """G5.3 derived-chemistry gate (plan §6 G5.3).

    Each sample mapping carries: ``pos`` [n,3] CURRENT frame, ``key_rows``
    [m], ``pairs`` (global-row key pairs), ``u`` [m,3], ``y`` [m,3],
    ``group_id`` (reaction cluster), ``reaction_id``, ``branch``.

    Sign accuracy counts only teacher-significant pairs (|Δd_l(y)|≥0.005
    Å); a predicted |Δd_l(u)|<0.005 Å counts as INCORRECT in the main
    denominator (abstention is never free).  SRE is computed on multi-bond
    frames (≥2 pairs, total teacher |Δd| ≥ 0.01 Å) with a fully-zero
    prediction mapped to SRE=1.  Aggregation follows the plan §6 order;
    cross-cluster summaries are median (sign) and median/Q75 (SRE).
    """
    sign_frame: list[float] = []
    sign_cluster: list[str] = []
    sign_reaction: list[str] = []
    sign_branch: list[int] = []
    sre_frame: list[float] = []
    sre_cluster: list[str] = []
    sre_reaction: list[str] = []
    sre_branch: list[int] = []
    deviations: list[float] = []
    n_pairs_sign = 0
    n_pairs_correct = 0
    n_active_pairs = 0
    n_dead_pairs = 0

    for sample in samples:
        pos = np.asarray(sample["pos"], dtype=np.float64)
        key_rows = [int(r) for r in sample["key_rows"]]
        pairs = [(int(p[0]), int(p[1])) for p in sample["pairs"]]
        u = np.asarray(sample["u"], dtype=np.float64)
        y = np.asarray(sample["y"], dtype=np.float64)
        cluster = str(sample["group_id"])
        reaction = str(sample.get("reaction_id", cluster))
        branch = int(sample.get("branch", 0))
        if not pairs:
            continue
        dd_u = exact_delta_d(pos, key_rows, pairs, u)
        dd_y = exact_delta_d(pos, key_rows, pairs, y)
        j_u = first_order_pair_projection(pos, key_rows, pairs, u)
        deviations.extend(np.abs(j_u - dd_u).tolist())

        significant = np.abs(dd_y) >= sign_threshold
        if bool(significant.any()):
            correct = (np.sign(dd_u[significant]) == np.sign(dd_y[significant])) \
                & (np.abs(dd_u[significant]) >= sign_threshold)
            n_pairs_sign += int(significant.sum())
            n_pairs_correct += int(correct.sum())
            n_active_pairs += int(significant.sum())
            n_dead_pairs += int(
                (np.abs(dd_u[significant]) < sign_threshold).sum())
            sign_frame.append(float(correct.mean()))
            sign_cluster.append(cluster)
            sign_reaction.append(reaction)
            sign_branch.append(branch)

        total_change = float(np.abs(dd_y).sum())
        if len(pairs) >= 2 and total_change >= sre_min_total_change:
            p_teacher = np.abs(dd_y) / total_change
            pred_change = float(np.abs(dd_u).sum())
            if pred_change <= 0.0:
                sre = 1.0
            else:
                p_hat = np.abs(dd_u) / pred_change
                sre = float(0.5 * np.abs(p_hat - p_teacher).sum())
            sre_frame.append(sre)
            sre_cluster.append(cluster)
            sre_reaction.append(reaction)
            sre_branch.append(branch)

    sign_clusters = hierarchical_means(
        np.asarray(sign_frame, dtype=np.float64) if sign_frame else np.zeros(0),
        sign_cluster, sign_reaction, sign_branch)
    sre_clusters = hierarchical_means(
        np.asarray(sre_frame, dtype=np.float64) if sre_frame else np.zeros(0),
        sre_cluster, sre_reaction, sre_branch)
    sign_median = _median(list(sign_clusters.values()))
    sign_q25 = _quantile(list(sign_clusters.values()), 0.25)
    sre_median = _median(list(sre_clusters.values()))
    sre_q75 = _quantile(list(sre_clusters.values()), 0.75)

    checks = {
        "sign_median_ge_0.90": (sign_median is not None
                                and sign_median >= G53_SIGN_MEDIAN_MIN),
        "sign_q25_ge_0.80": (sign_q25 is not None and sign_q25 >= G53_SIGN_Q25_MIN),
        "sre_median_le_0.15": (sre_median is not None
                               and sre_median <= G53_SRE_MEDIAN_MAX),
        "sre_q75_le_0.25": (sre_q75 is not None and sre_q75 <= G53_SRE_Q75_MAX),
    }
    return {
        "schema_version": G53_EVAL_SCHEMA,
        "per_cluster_sign_accuracy": sign_clusters,
        "per_cluster_sre": sre_clusters,
        "aggregate": {
            "sign_accuracy_cluster_median": sign_median,
            "sign_accuracy_cluster_q25": sign_q25,
            "sre_cluster_median": sre_median,
            "sre_cluster_q75": sre_q75,
            "n_sign_pairs": n_pairs_sign,
            "n_sign_pairs_correct": n_pairs_correct,
            "n_active_pairs": n_active_pairs,
            "n_dead_pairs": n_dead_pairs,
            "dead_zone_fraction": (
                float(n_dead_pairs) / float(n_active_pairs)
                if n_active_pairs > 0 else None),
            "n_sre_frames": len(sre_frame),
            "n_clusters_sign": len(sign_clusters),
            "n_clusters_sre": len(sre_clusters),
            "j_vs_exact_dd_abs_deviation_mean": (
                float(np.mean(deviations)) if deviations else None),
            "j_vs_exact_dd_abs_deviation_max": (
                float(np.max(deviations)) if deviations else None),
        },
        "thresholds": {
            "sign_pair_threshold_angstrom": sign_threshold,
            "sign_median_min": G53_SIGN_MEDIAN_MIN,
            "sign_q25_min": G53_SIGN_Q25_MIN,
            "sre_median_max": G53_SRE_MEDIAN_MAX,
            "sre_q75_max": G53_SRE_Q75_MAX,
            "sre_min_total_change_angstrom": sre_min_total_change,
            "abstention_policy": ("predicted |Δd|<0.005 A counts incorrect "
                                  "in the main denominator"),
        },
        "gate": {"checks": checks, "pass": all(checks.values())},
    }


# ---------------------------------------------------------------------------
# G5.4 — static executability gate (strict-copy, no QC)
# ---------------------------------------------------------------------------


def assert_strict_copy(
    original: np.ndarray, patched: np.ndarray, key_rows: Sequence[int]
) -> None:
    """Non-key coordinates must be BIT-identical after the patch (plan §6
    G5.1/G5.4 strict-copy contract, ``torch.equal`` semantics)."""
    original = np.asarray(original)
    patched = np.asarray(patched)
    _require(original.shape == patched.shape, "STRICT_COPY_SHAPE",
             f"shape changed: {original.shape} -> {patched.shape}")
    key_mask = np.zeros(original.shape[0], dtype=bool)
    key_mask[np.asarray(list(key_rows), dtype=np.int64)] = True
    non_key_equal = bool(np.array_equal(patched[~key_mask],
                                        original[~key_mask]))
    if not non_key_equal:
        raise KeyAtomProtocolError(
            "STRICT_COPY_VIOLATION",
            "non-key atom coordinates changed under a key-atom-only patch")


def patch_strict_copy(
    pos: np.ndarray, u: np.ndarray, key_rows: Sequence[int],
) -> np.ndarray:
    """Clone ``pos`` and add ``u`` to the key-atom rows ONLY.

    Asserts the strict-copy contract (non-key coordinates bit-identical)
    and that ``u`` aligns with ``key_rows``.  Works for numpy arrays and
    torch tensors (``torch.equal``-checked for the latter).
    """
    key = np.asarray(list(key_rows), dtype=np.int64)
    u_array = np.asarray(u)
    if hasattr(pos, "detach") and hasattr(u, "detach"):
        import torch  # local import: optional heavy backend

        _require(int(key.numel() if hasattr(key, "numel") else len(key))
                 == int(u.detach().shape[0]), "PATCH_SHAPE",
                 "u rows must match key_rows length")
        patched = pos.detach().clone()
        patched[key] = patched[key] + u.detach()
        key_mask = torch.zeros(pos.shape[0], dtype=torch.bool,
                               device=pos.device)
        key_mask[key] = True
        if not torch.equal(patched[~key_mask], pos[~key_mask]):
            raise KeyAtomProtocolError(
                "STRICT_COPY_VIOLATION",
                "non-key atom coordinates changed under a key-atom-only patch")
        return patched
    pos_array = np.asarray(pos, dtype=np.float64)
    u64 = np.asarray(u, dtype=np.float64)
    _require(u64.shape == (key.size, 3), "PATCH_SHAPE",
             f"u must be [{key.size},3], got {u64.shape}")
    patched = pos_array.copy()
    patched[key] = patched[key] + u64
    assert_strict_copy(pos_array, patched, key)
    return patched


def _covariant_radii_or_raise(atomic_numbers: Sequence[int]) -> np.ndarray:
    radii = np.zeros(len(list(atomic_numbers)), dtype=np.float64)
    for i, z in enumerate(atomic_numbers):
        radius = COVALENT_RADII.get(int(z))
        _require(radius is not None, "G54_UNKNOWN_ELEMENT",
                 f"no Cordero covalent radius for Z={z} at row {i}")
        radii[i] = float(radius)
    return radii


def clash_pairs_of(
    pos: np.ndarray,
    atomic_numbers: Sequence[int],
    *,
    factor: float = G54_CLASH_FACTOR,
    exceptions: frozenset = PREREGISTERED_CLASH_EXCEPTIONS,
) -> list[list[int]]:
    """Interatomic pairs closer than ``factor·(rcov_i+rcov_j)`` (Å).

    Plan §6 G5.4: this INCLUDES bonded pairs — the only legitimate short
    contacts are PRE-REGISTERED exceptions (none exist this sprint); a
    post-hoc exemption is impossible by construction.
    """
    pos = np.asarray(pos, dtype=np.float64)
    radii = _covariant_radii_or_raise(atomic_numbers)
    n = pos.shape[0]
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    limit = factor * (radii[:, None] + radii[None, :])
    out: list[list[int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if {int(i), int(j)} in exceptions:
                continue
            if dist[i, j] < limit[i, j]:
                out.append([int(i), int(j)])
    return out


def protected_edges_of(
    start_bonds: Sequence[tuple[int, int, float]],
    k_pairs: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """``(G_start bonds) \\ K`` — plan §6 G5.4 protected edges.

    Includes key–non-key edges such as X–H: checking only bonds whose BOTH
    endpoints are non-key would be vacuous under strict-copy (their length
    can never change) and would fake safety.
    """
    k_set = {frozenset((int(i), int(j))) for i, j, *_ in k_pairs}
    edges: list[tuple[int, int]] = []
    seen: set[frozenset] = set()
    for bond in start_bonds:
        i, j = int(bond[0]), int(bond[1])
        pair = frozenset((i, j))
        if pair in k_set or pair in seen or i == j:
            continue
        seen.add(pair)
        edges.append((min(i, j), max(i, j)))
    return edges


def protected_violations_of(
    pos: np.ndarray,
    protected_edges: Sequence[tuple[int, int]],
    pos_start: np.ndarray,
    *,
    stretch_factor: float = G54_STRETCH_FACTOR,
    shorten_factor: float = G54_SHORTEN_FACTOR,
) -> list[dict]:
    """Protected edges stretched >1.35·d_eq or shortened <0.70·d_eq.

    ``d_eq`` is the deployment START-frame bond length per branch (plan §6
    G5.4: never picked from a future minimum-energy frame); start-frame
    anomalies (non-positive or sub-0.5 Å bonds) are refused, not silently
    repaired.
    """
    pos = np.asarray(pos, dtype=np.float64)
    start = np.asarray(pos_start, dtype=np.float64)
    out: list[dict] = []
    for i, j in protected_edges:
        d_eq = float(np.linalg.norm(start[int(j)] - start[int(i)]))
        _require(d_eq > 0.5, "G54_DEQ_ANOMALOUS",
                 f"start-frame bond ({i},{j}) length {d_eq:.4f} A <= 0.5 A")
        d_now = float(np.linalg.norm(pos[int(j)] - pos[int(i)]))
        if d_now > stretch_factor * d_eq:
            out.append({"edge": [int(i), int(j)], "kind": "stretch",
                        "d": d_now, "d_eq": d_eq})
        elif d_now < shorten_factor * d_eq:
            out.append({"edge": [int(i), int(j)], "kind": "shorten",
                        "d": d_now, "d_eq": d_eq})
    return out


def _sample_geometry_violations(
    sample: Mapping, u: np.ndarray
) -> tuple[dict, dict]:
    """(input, patched) violation summaries for one G5.4 sample.

    Sample fields: ``pos`` [n,3] CURRENT input frame, ``pos_start`` [n,3]
    deployment start frame (for d_eq), ``atomic_numbers`` [n],
    ``key_rows`` [m], ``k_pairs`` (global rows of the K changed pairs),
    ``start_bonds`` (branch start graph ``(i, j, order)``).
    """
    pos = np.asarray(sample["pos"], dtype=np.float64)
    pos_start = np.asarray(sample.get("pos_start", sample["pos"]),
                           dtype=np.float64)
    key_rows = [int(r) for r in sample["key_rows"]]
    atomic_numbers = [int(z) for z in sample["atomic_numbers"]]
    k_pairs = [(int(p[0]), int(p[1])) for p in sample.get("k_pairs", ())]
    start_bonds = [tuple(b) for b in sample.get("start_bonds", ())]
    protected = protected_edges_of(start_bonds, k_pairs)
    patched = patch_strict_copy(pos, u, key_rows)

    def _summarize(geometry: np.ndarray) -> dict:
        clashes = clash_pairs_of(geometry, atomic_numbers)
        prot = protected_violations_of(geometry, protected, pos_start)
        return {
            "clash_pairs": clashes,
            "n_clash_pairs": len(clashes),
            "protected_violations": prot,
            "n_protected_violations": len(prot),
            "has_clash": bool(clashes),
            "has_protected_violation": bool(prot),
        }

    return _summarize(pos), _summarize(patched)


def _one_sided_upper_bootstrap(
    cluster_values: Mapping[str, float], *, draws: int, seed: int,
) -> float | None:
    """One-sided 95% cluster-bootstrap upper bound of the cluster mean."""
    values = np.asarray([v for v in cluster_values.values()
                         if math.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    n = values.size
    stats = np.empty(draws, dtype=np.float64)
    for k in range(draws):
        pick = rng.integers(0, n, size=n)
        stats[k] = values[pick].mean()
    return float(np.quantile(stats, 0.95))


def evaluate_g54(
    samples: Sequence[Mapping],
    predictions: Sequence[np.ndarray],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict:
    """G5.4 static executability gate (plan §6 G5.4; ADR-0026 §4).

    ``samples[i]`` describes the input geometry of one frame (see
    :func:`_sample_geometry_violations`); ``predictions[i]`` is the
    key-atom displacement ``u [m,3]`` (bounded export — executability is
    a bounded-output property).  Frames whose u is missing or non-finite
    count as REFUSALS and stay in the denominator (plan §6: refusal ≤5%).

    Gates: new-clash rate ≤0.5%, new-protected rate ≤1%, each with a
    one-sided 95% cluster-bootstrap upper bound ≤2%, no single cluster
    >5%, refusal ≤5%.  "new" means violated AFTER the patch but not in the
    INPUT frame — input violations remain in the denominator and are
    never laundered away.
    """
    _require(len(samples) == len(predictions), "G54_SHAPE_MISMATCH",
             "samples and predictions must align")
    cluster_of: list[str] = []
    rows: list[dict] = []
    n_refused = 0
    for sample, u_raw in zip(samples, predictions):
        cluster = str(sample["group_id"])
        u = None if u_raw is None else np.asarray(u_raw, dtype=np.float64)
        if u is None or not bool(np.isfinite(u).all()):
            n_refused += 1
            rows.append({"cluster": cluster, "refused": True,
                         "input_clash": False, "input_protected": False,
                         "patched_clash": False, "patched_protected": False,
                         "new_clash": False, "new_protected": False})
            cluster_of.append(cluster)
            continue
        source, patched = _sample_geometry_violations(sample, u)
        new_clash = patched["has_clash"] and not source["has_clash"]
        new_protected = (patched["has_protected_violation"]
                         and not source["has_protected_violation"])
        rows.append({
            "cluster": cluster,
            "refused": False,
            "input_clash": source["has_clash"],
            "input_protected": source["has_protected_violation"],
            "patched_clash": patched["has_clash"],
            "patched_protected": patched["has_protected_violation"],
            "new_clash": bool(new_clash),
            "new_protected": bool(new_protected),
            "n_clash_pairs": patched["n_clash_pairs"],
            "n_protected_violations": patched["n_protected_violations"],
        })
        cluster_of.append(cluster)

    n_frames = len(rows)
    _require(n_frames > 0, "G54_NO_FRAMES", "no samples supplied")

    def _rate(key: str) -> float:
        return sum(1 for r in rows if r[key]) / n_frames

    clusters = sorted(set(cluster_of))
    new_clash_per_cluster: dict[str, float] = {}
    new_protected_per_cluster: dict[str, float] = {}
    for cluster in clusters:
        members = [r for r in rows if r["cluster"] == cluster]
        new_clash_per_cluster[cluster] = (
            sum(1 for r in members if r["new_clash"]) / len(members))
        new_protected_per_cluster[cluster] = (
            sum(1 for r in members if r["new_protected"]) / len(members))

    new_clash_upper = _one_sided_upper_bootstrap(
        new_clash_per_cluster, draws=bootstrap_draws, seed=bootstrap_seed)
    new_protected_upper = _one_sided_upper_bootstrap(
        new_protected_per_cluster, draws=bootstrap_draws,
        seed=bootstrap_seed + 1)
    refusal_rate = n_refused / n_frames
    max_cluster_clash = max(new_clash_per_cluster.values())
    max_cluster_protected = max(new_protected_per_cluster.values())

    checks = {
        "new_clash_rate_le_0.005": _rate("new_clash") <= G54_NEW_CLASH_MAX,
        "new_protected_rate_le_0.01": (
            _rate("new_protected") <= G54_NEW_PROTECTED_MAX),
        "new_clash_upper95_le_0.02": (
            new_clash_upper is not None and new_clash_upper <= G54_UPPER_BOUND_MAX),
        "new_protected_upper95_le_0.02": (
            new_protected_upper is not None
            and new_protected_upper <= G54_UPPER_BOUND_MAX),
        "no_cluster_clash_gt_0.05": max_cluster_clash <= G54_CLUSTER_MAX,
        "no_cluster_protected_gt_0.05": (
            max_cluster_protected <= G54_CLUSTER_MAX),
        "refusal_rate_le_0.05": refusal_rate <= G54_REFUSAL_MAX,
    }
    return {
        "schema_version": G54_EVAL_SCHEMA,
        "n_frames": n_frames,
        "n_clusters": len(clusters),
        "n_refused": n_refused,
        "rates": {
            "new_clash_rate": _rate("new_clash"),
            "new_protected_rate": _rate("new_protected"),
            "total_clash_rate": _rate("patched_clash") if n_frames else 0.0,
            "total_protected_rate": _rate("patched_protected"),
            "input_clash_rate": _rate("input_clash"),
            "input_protected_rate": _rate("input_protected"),
            "refusal_rate": refusal_rate,
        },
        "per_cluster": {
            cluster: {
                "new_clash_fraction": new_clash_per_cluster[cluster],
                "new_protected_fraction": new_protected_per_cluster[cluster],
            }
            for cluster in clusters},
        "bootstrap": {
            "draws": int(bootstrap_draws),
            "seed": int(bootstrap_seed),
            "unit": "reaction_cluster",
            "new_clash_upper95": new_clash_upper,
            "new_protected_upper95": new_protected_upper,
        },
        "thresholds": {
            "clash_factor": G54_CLASH_FACTOR,
            "clash_rule": "d < 0.75*(rcov_i+rcov_j), bonded pairs INCLUDED",
            "preregistered_exceptions": sorted(
                sorted(pair) for pair in PREREGISTERED_CLASH_EXCEPTIONS),
            "stretch_factor": G54_STRETCH_FACTOR,
            "shorten_factor": G54_SHORTEN_FACTOR,
            "d_eq_source": "deployment start frame per branch",
            "new_clash_rate_max": G54_NEW_CLASH_MAX,
            "new_protected_rate_max": G54_NEW_PROTECTED_MAX,
            "upper95_max": G54_UPPER_BOUND_MAX,
            "cluster_max": G54_CLUSTER_MAX,
            "refusal_max": G54_REFUSAL_MAX,
        },
        "gate": {"checks": checks, "pass": all(checks.values())},
        "rows": rows,
    }


def oracle_static_ceiling(
    samples: Sequence[Mapping],
    teacher: Sequence[np.ndarray],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict:
    """Static teacher ceiling under the SAME strict-copy protocol.

    Plan §0/§6 and ADR-0026 §4: when the oracle teacher itself fails G5.4
    the ACTION REPRESENTATION (d0 / contract) is flagged via a typed
    ``ORACLE_STATIC_FAILED`` status — the failure is never blamed on the
    network, and model promotion is blocked until the contract is revised.
    """
    report = evaluate_g54(samples, list(teacher),
                          bootstrap_seed=bootstrap_seed,
                          bootstrap_draws=bootstrap_draws)
    status = ("ORACLE_STATIC_OK" if report["gate"]["pass"]
              else "ORACLE_STATIC_FAILED")
    return {
        "status": status,
        "action_representation_flagged": status == "ORACLE_STATIC_FAILED",
        "note": ("oracle teacher evaluated under the identical strict-copy "
                 "static protocol; a failure flags the action "
                 "representation (step size / contract), not the model "
                 "(plan §6 G5.4)"),
        "report": report,
    }


# ---------------------------------------------------------------------------
# G5.4 follower counterfactual (contract_counterfactual — NOT in the pass)
# ---------------------------------------------------------------------------


def follower_h_rows(
    atomic_numbers: Sequence[int],
    key_rows: Sequence[int],
    start_bonds: Sequence[tuple[int, int, float]],
) -> list[tuple[int, int]]:
    """Terminal migrating hydrogens: ``(h_row, heavy_key_row)`` pairs.

    Plan §6 G5.4 / plan §0 follower risk: only an H that (a) is NOT itself
    a key atom, (b) has exactly ONE start-graph neighbor, and (c) that
    neighbor is a KEY HEAVY atom, qualifies.  Bridging/transferring H
    (degree ≥2 or key itself) is excluded — moving it would silently
    change the executed reaction object.
    """
    key = {int(r) for r in key_rows}
    degree: dict[int, set[int]] = {}
    for bond in start_bonds:
        i, j = int(bond[0]), int(bond[1])
        degree.setdefault(i, set()).add(j)
        degree.setdefault(j, set()).add(i)
    out: list[tuple[int, int]] = []
    for row, z in enumerate(atomic_numbers):
        if int(z) != 1 or row in key:
            continue
        neighbors = degree.get(row, set())
        if len(neighbors) != 1:
            continue  # bridging / multi-attached H
        heavy = next(iter(neighbors))
        if heavy in key and int(atomic_numbers[heavy]) > 1:
            out.append((int(row), int(heavy)))
    return sorted(out)


def follower_counterfactual_diagnostics(
    samples: Sequence[Mapping],
    predictions: Sequence[np.ndarray],
) -> list[dict]:
    """Paired strict-copy vs follower-translated static diagnostics.

    For each sample the follower H (if any) is translated by its key heavy
    atom's displacement ON TOP of the strict-copy patch; the violation
    summaries of both geometries are reported as a paired difference.
    Results are marked ``contract_counterfactual`` and are NEVER counted
    in the G5.4 pass (plan §6: the follower branch cannot enter the gate
    denominator nor silently grow the frozen set).
    """
    out: list[dict] = []
    for sample, u_raw in zip(samples, predictions):
        entry: dict = {
            "contract_counterfactual": True,
            "group_id": str(sample["group_id"]),
            "n_followers": 0,
            "strict_copy": None,
            "counterfactual": None,
            "paired_diff": None,
            "note": ("diagnostic only — excluded from the G5.4 gate "
                     "(plan §6 G5.4 follower counterfactual)"),
        }
        if u_raw is None:
            out.append(entry)
            continue
        u = np.asarray(u_raw, dtype=np.float64)
        if not bool(np.isfinite(u).all()):
            out.append(entry)
            continue
        pos = np.asarray(sample["pos"], dtype=np.float64)
        key_rows = [int(r) for r in sample["key_rows"]]
        followers = follower_h_rows(
            [int(z) for z in sample["atomic_numbers"]], key_rows,
            [tuple(b) for b in sample.get("start_bonds", ())])
        entry["n_followers"] = len(followers)
        strict = patch_strict_copy(pos, u, key_rows)
        slot = {row: k for k, row in enumerate(key_rows)}
        cf = strict.copy()
        for h_row, heavy_row in followers:
            cf[h_row] = cf[h_row] + u[slot[heavy_row]]

        def _summarize(geometry: np.ndarray) -> dict:
            clashes = clash_pairs_of(
                geometry, [int(z) for z in sample["atomic_numbers"]])
            protected = protected_violations_of(
                geometry,
                protected_edges_of(
                    [tuple(b) for b in sample.get("start_bonds", ())],
                    [(int(p[0]), int(p[1]))
                     for p in sample.get("k_pairs", ())]),
                np.asarray(sample.get("pos_start", sample["pos"]),
                           dtype=np.float64))
            return {"has_clash": bool(clashes), "n_clash_pairs": len(clashes),
                    "has_protected_violation": bool(protected),
                    "n_protected_violations": len(protected)}

        strict_summary = _summarize(strict)
        cf_summary = _summarize(cf)
        entry["strict_copy"] = strict_summary
        entry["counterfactual"] = cf_summary
        entry["paired_diff"] = {
            "new_clash_from_follower": (
                cf_summary["has_clash"] and not strict_summary["has_clash"]),
            "new_protected_from_follower": (
                cf_summary["has_protected_violation"]
                and not strict_summary["has_protected_violation"]),
        }
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Sealed-eval guard (plan §5.5; ADR-0026 §4 — never unsealed this sprint)
# ---------------------------------------------------------------------------


def sealed_eval_request(
    registration: Mapping,
    *,
    crossfit_complete_runs: int,
    model_hash: str | None,
    config_hash: str | None,
    protocol_hash: str | None,
    unseal: bool,
    allow_unseal: bool,
    sprint_policy: str = "v5_sprint_phase1_never_unseals",
) -> dict:
    """Sealed-confirm access request — typed REFUSED unless fully qualified.

    Qualification (plan §5.5): the formal crossfit must be COMPLETE for all
    15 runs (5 folds × 3 seeds; a 5-run downgrade does not qualify), the
    model/config/protocol sha256 hashes must all be frozen, and the caller
    must both pass ``unseal=True`` AND hold a protocol-level
    ``allow_unseal=True``.  The V5 phase-1 sprint ALWAYS calls with
    ``allow_unseal=False`` (plan §5.5 / ADR-0026 §4: opening the sealed
    set is out of scope for this window), so every request during this
    sprint is refused with code ``sealed_confirm_not_opened`` and the
    registration echoed verbatim.
    """
    registration_echo = {str(k): v for k, v in registration.items()}
    reasons: list[str] = []
    if int(crossfit_complete_runs) < 15:
        reasons.append(
            f"crossfit complete runs {int(crossfit_complete_runs)} < 15 "
            "(5 folds x 3 seeds; a 1-seed downgrade never qualifies)")
    for name, value in (("model", model_hash), ("config", config_hash),
                        ("protocol", protocol_hash)):
        if not value:
            reasons.append(f"{name} hash not frozen")
    if not unseal:
        reasons.append("explicit --unseal not passed")
    if not allow_unseal:
        reasons.append(
            f"evaluation protocol policy {sprint_policy!r} forbids "
            "unsealing in this sprint (plan §5.5 / ADR-0026 §4)")
    qualified = not reasons
    return {
        "schema_version": SEALED_DECISION_SCHEMA,
        "status": "REFUSED" if not qualified else "QUALIFIED",
        "code": (None if qualified else "sealed_confirm_not_opened"),
        "reasons": reasons,
        "registration": registration_echo,
        "crossfit_complete_runs": int(crossfit_complete_runs),
        "hashes_frozen": {
            "model_sha256": model_hash,
            "config_sha256": config_hash,
            "protocol_sha256": protocol_hash,
        },
        "unseal_requested": bool(unseal),
        "allow_unseal": bool(allow_unseal),
        "sprint_policy": sprint_policy,
        "note": ("the V5 phase-1 sprint never opens the sealed confirm set; "
                 "Confirm/OOD stay registered-but-unopened (ADR-0026 §4)"),
    }


# ---------------------------------------------------------------------------
# Gate report writer (plan §8 deliverables)
# ---------------------------------------------------------------------------


def normalize_capability(status: Mapping | None) -> dict:
    """Four-layer capability block (ADR-0026 §7), missing layers defaulted."""
    merged = dict(CAPABILITY_DEFAULTS)
    for key, value in dict(status or {}).items():
        if key in CAPABILITY_LAYERS:
            merged[key] = str(value)
    return merged


def normalize_artifact(
    artifact: Mapping, *, repo_root: str | Path | None = None
) -> dict:
    """One artifact row: schema/config/source/parent sha256 + capability.

    Plan §8: every deliverable carries schema/config/source/parent hashes
    and the four-layer capability status; the file digest is computed when
    the path exists (missing files stay ``sha256=None`` — never faked).
    """
    path = artifact.get("path")
    digest = None
    if path:
        candidate = Path(path)
        if not candidate.is_absolute() and repo_root is not None:
            candidate = Path(repo_root) / candidate
        if candidate.is_file():
            digest = _sha256_file(candidate)
    return {
        "name": str(artifact.get("name", "")),
        "schema": artifact.get("schema"),
        "path": str(path) if path is not None else None,
        "sha256": digest,
        "config_sha256": artifact.get("config_sha256"),
        "source": artifact.get("source"),
        "parent": artifact.get("parent"),
        "capability": normalize_capability(artifact.get("capability")),
        "status": str(artifact.get("status", "measured")),
        "note": artifact.get("note"),
    }


def render_gate_report_md(report: Mapping) -> str:
    """Human-readable rendering of a ``pesn_v5_gate_report_v1`` document."""
    lines: list[str] = []
    lines.append("# PES2TS / PES-N V5 冲刺 Gate 报告（G5.1–G5.4 + 封存评测）")
    lines.append("")
    lines.append(f"> schema: `{report.get('schema_version')}`")
    lines.append(f"> generated: {report.get('generated_at')}")
    lines.append(f"> campaign: {report.get('campaign_id', 'V5_SPRINT')}")
    lines.append("")
    lines.append("状态词汇：measured / planned / not_executed / censored；"
                 "无 QC 计划处写 `QC not scheduled`（不填 0）。")
    lines.append("")
    lines.append("## 1. Gate 状态总览")
    lines.append("")
    lines.append("| Gate | 状态 | pass | 说明 |")
    lines.append("|---|---|---|---|")
    for gate_id in ("G5.1", "G5.2", "G5.3", "G5.4", "oracle_static",
                    "sealed_eval"):
        section = (report.get("gates") or {}).get(gate_id) or {}
        status = section.get("status", "not_executed")
        verdict = section.get("verdict", "—")
        note = str(section.get("note", "")).replace("|", "/")
        lines.append(f"| {gate_id} | {status} | {verdict} | {note} |")
    lines.append("")
    for gate_id in ("G5.1", "G5.2", "G5.3", "G5.4", "oracle_static",
                    "sealed_eval"):
        section = (report.get("gates") or {}).get(gate_id)
        if not section:
            continue
        lines.append(f"## {gate_id}（{section.get('status', 'not_executed')}）")
        lines.append("")
        payload = section.get("payload")
        if payload is None:
            lines.append(str(section.get("note", "not executed in this window")))
        else:
            lines.append("```json")
            lines.append(json.dumps(payload, indent=2, ensure_ascii=False,
                                    sort_keys=True))
            lines.append("```")
        lines.append("")
    lines.append("## 2. 产物登记")
    lines.append("")
    lines.append("| 产物 | schema | sha256 | 四层能力 | 状态 |")
    lines.append("|---|---|---|---|---|")
    for artifact in report.get("artifacts") or []:
        cap = artifact.get("capability") or {}
        cap_text = "/".join(str(cap.get(k, "unknown"))
                            for k in CAPABILITY_LAYERS)
        digest = artifact.get("sha256") or "—"
        if len(str(digest)) > 16:
            digest = str(digest)[:16] + "…"
        lines.append(
            f"| {artifact.get('name')} | {artifact.get('schema')} | "
            f"`{digest}` | {cap_text} | {artifact.get('status')} |")
    lines.append("")
    lines.append(f"QC：{report.get('qc', QC_NOT_SCHEDULED)}")
    lines.append("")
    statuses = report.get("stage_statuses") or {}
    if statuses:
        lines.append("## 3. 战役阶段状态（方案 §5.7 词汇）")
        lines.append("")
        lines.append("| stage | status |")
        lines.append("|---|---|")
        for stage in sorted(statuses):
            lines.append(f"| {stage} | {statuses[stage]} |")
        lines.append("")
    return "\n".join(lines)


def write_gate_report(
    out_dir: str | Path,
    *,
    gates: Mapping[str, Mapping],
    artifacts: Sequence[Mapping],
    stage_statuses: Mapping[str, str] | None = None,
    campaign_id: str = "V5_SPRINT",
    qc: str = QC_NOT_SCHEDULED,
    config_sha256: str | None = None,
    extra: Mapping | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    """Write ``gate_report.json`` + ``gate_report.md`` (plan §8).

    Gate sections carry ``status`` ∈ {measured, planned, not_executed,
    censored} plus a ``verdict`` and optional ``payload``; artifacts are
    normalized through :func:`normalize_artifact`.
    """
    for gate_id, section in dict(gates).items():
        status = str(section.get("status", "not_executed"))
        _require(status in SECTION_STATUSES, "GATE_REPORT_STATUS",
                 f"gate {gate_id}: status {status!r} not in "
                 f"{SECTION_STATUSES}")
    report = {
        "schema_version": GATE_REPORT_SCHEMA,
        "generated_at": _utc_now(),
        "campaign_id": campaign_id,
        "config_sha256": config_sha256,
        "gates": {str(k): dict(v) for k, v in dict(gates).items()},
        "artifacts": [normalize_artifact(a, repo_root=repo_root)
                      for a in artifacts],
        "stage_statuses": dict(stage_statuses or {}),
        "qc": qc,
        "section_statuses": list(SECTION_STATUSES),
        "capability_layers": list(CAPABILITY_LAYERS),
    }
    if extra:
        report["extra"] = dict(extra)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "gate_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    md_path = out / "gate_report.md"
    md_path.write_text(render_gate_report_md(report), encoding="utf-8")
    return {
        "gate_report_json": {"path": str(json_path),
                             "sha256": _sha256_file(json_path)},
        "gate_report_md": {"path": str(md_path),
                           "sha256": _sha256_file(md_path)},
    }


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CAPABILITY_LAYERS",
    "EFFECTIVE_ATOM_TAU",
    "GATE_REPORT_SCHEMA",
    "G52_EVAL_SCHEMA",
    "G53_EVAL_SCHEMA",
    "G54_EVAL_SCHEMA",
    "KeyAtomProtocolError",
    "NEAR_TS_CUT",
    "PREREGISTERED_CLASH_EXCEPTIONS",
    "QC_NOT_SCHEDULED",
    "SECTION_STATUSES",
    "SEALED_DECISION_SCHEMA",
    "assert_strict_copy",
    "clash_pairs_of",
    "evaluate_g52",
    "evaluate_g53",
    "evaluate_g54",
    "exact_delta_d",
    "first_order_pair_projection",
    "follower_counterfactual_diagnostics",
    "follower_h_rows",
    "hierarchical_means",
    "normalize_artifact",
    "oracle_static_ceiling",
    "patch_strict_copy",
    "protected_edges_of",
    "protected_violations_of",
    "render_gate_report_md",
    "sealed_eval_request",
    "write_gate_report",
]
