"""Monitor8 evaluation (WP4-A, plan v4 §I.9/§I.13; G4 §5 pre-registration).

Everything here is gated on a content-digested pre-registration artifact
(``monitor_prereg_v1.json``): **any** monitor evaluation refuses to run
unless the prereg file exists AND its sha256 matches the digest recorded by
the caller (thresholds file / run record).  This is binding condition C4:
pre-registration is frozen BEFORE any monitor evaluation of trained models,
and the quantitative gate thresholds are frozen AFTER it, before signing.

Components:

* prereg load + digest gate (``load_prereg``, ``gate_prereg``);
* population / leakage assertions (``assert_fit_monitor_disjoint`` —
  fit ∩ monitor = ∅, disputed-5 membership, sealed-family overlap);
* same-density baseline: a state-independent radial-tanh Gaussian with
  GLOBAL μ̄/σ̄ fit by MLE on the fit50 teacher latents ONLY (the log-det is
  a data constant, so the MLE is the plain Gaussian moment match on
  ``u* = R⁻¹(a*/a_max)``), evaluated with the identical transform and the
  identical fixed §I.10 reduction semantics (per-atom here: the monitor is
  a full-population evaluation, the per-group values ARE the registered
  unit of report — n=8 admits no pooled-only claims);
* per-group metrics (all 8 groups reported INDIVIDUALLY): per-atom MSE +
  ratio vs the zero-action baseline, |a|-weighted direction cosine,
  per-atom NLL (model + same-density), strata near-TS / moving / near-rest
  with FROZEN fit-measured thresholds (no re-measurement — C5; the near-TS
  arc rule is the frozen per-branch definition), the disputed-5 sensitivity
  slice, and the σ-vs-|u*| summary with the two registered σ-pathology
  flags;
* aggregate: median over groups + per-group sign consistency;
* C3 trigger evaluation (joint |a|-w cosine < 0.30 OR near-TS cosine
  < 0.40 at budget end) and the frozen model-selection rule.

Free-run rollouts are WP4-B scope and do NOT live here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from pes2ts_core.pesn.offline.training.dual_ts_bc import (
    build_targets,
    load_dual_ts_states,
    per_atom_nll,
    sha256_file,
)

MONITOR_PREREG_SCHEMA: Final[str] = "pesn_wp4_monitor_prereg_v1"
MONITOR_THRESHOLDS_SCHEMA: Final[str] = "pesn_wp4_monitor_thresholds_v1"
MONITOR_EVAL_SCHEMA: Final[str] = "pesn_wp4_monitor_eval_v1"
SELECTION_RECORD_SCHEMA: Final[str] = "pesn_wp4_selection_record_v1"

#: C3 registered trigger (G4 §4): fired ONLY by monitor evidence at budget
#: end; never by fit-side numbers.
C3_TRIGGER_JOINT_COS: Final[float] = 0.30
C3_TRIGGER_NEAR_TS_COS: Final[float] = 0.40

#: Registered σ-pathology flags (G4 §3 "σ vs |u*| scatter summary"):
#: (i) σ pinned at the floor on (nearly) all atoms; (ii) σ ≫ |u*| on the
#: near-rest stratum (registered factor 3×; on fit, RMS|a| ≈ 0.0225 Å ⇒
#: |u*| ≈ 0.23 ≫ σ_min = 0.02 — the floor protects, it does not block).
SIGMA_FLOOR_FRACTION_FLAG: Final[float] = 0.9
SIGMA_FLOOR_VALUE: Final[float] = 0.02
SIGMA_DOMINATES_FACTOR: Final[float] = 3.0

_LOG_2PI: Final[float] = math.log(2.0 * math.pi)


class MonitorGateError(RuntimeError):
    """Typed prereg/data gate refusal (carries a machine-readable code)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# Prereg + leakage assertions
# ---------------------------------------------------------------------------

def load_prereg(path: str | Path) -> tuple[dict, str]:
    """Load the prereg payload and its sha256 (bytes digest)."""
    path = Path(path)
    if not path.is_file():
        raise MonitorGateError(
            "PREREG_MISSING",
            f"monitor pre-registration missing: {path} — C4 forbids any "
            f"monitor evaluation without it")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != MONITOR_PREREG_SCHEMA:
        raise MonitorGateError(
            "PREREG_SCHEMA_UNKNOWN",
            f"expected schema {MONITOR_PREREG_SCHEMA}, got "
            f"{payload.get('schema_version')!r}")
    for key in ("population", "baselines", "metrics", "sampling",
                "reduction", "recipes", "budget", "model_selection"):
        if key not in payload:
            raise MonitorGateError(
                "PREREG_INCOMPLETE", f"missing section {key!r}")
    return payload, sha256_file(path)


def gate_prereg(path: str | Path, expected_sha256: str) -> tuple[dict, str]:
    """C4 gate: prereg must exist AND digest-match the caller's record."""
    payload, digest = load_prereg(path)
    if not expected_sha256 or digest != expected_sha256:
        raise MonitorGateError(
            "PREREG_DIGEST_CHANGED",
            f"prereg digest {digest} != recorded {expected_sha256!r} — "
            f"refusing monitor evaluation (C4)")
    return payload, digest


def assert_fit_monitor_disjoint(
    *,
    fit_group_ids: Sequence[str],
    monitor_group_ids: Sequence[str],
    lineage: dict,
    disputed_ids: Sequence[str] = (),
) -> dict:
    """Leakage assertions for the monitor population (C4 §5 item 1).

    Asserts fit ∩ monitor = ∅, every monitor id sits in the lineage monitor
    partition, no disputed-5 membership, and no sealed-family overlap.
    Raises :class:`MonitorGateError` on any violation; returns the evidence
    payload recorded into the pre-registration.
    """
    fit, mon = set(fit_group_ids), set(monitor_group_ids)
    inter = sorted(fit & mon)
    if inter:
        raise MonitorGateError(
            "FIT_MONITOR_LEAK", f"groups in both partitions: {inter}")
    partition = set(lineage["partitions"]["monitor"]["group_ids"])
    outside = sorted(mon - partition)
    if outside:
        raise MonitorGateError(
            "MONITOR_PARTITION_MISMATCH",
            f"monitor ids outside the lineage monitor partition: {outside}")
    disputed = sorted(mon & set(disputed_ids))
    if disputed:
        raise MonitorGateError(
            "DISPUTED_IN_MONITOR",
            f"disputed-5 reactions inside the monitor wave: {disputed}")
    sealed = set(
        lineage.get("family_audit", {}).get("sealed_reaction_ids", []))
    overlap = sorted(mon & sealed)
    if overlap:
        raise MonitorGateError(
            "FAMILY_SEAL_OVERLAP",
            f"monitor ids inside sealed families: {overlap}")
    return {
        "fit_intersection": [],
        "all_in_monitor_partition": True,
        "disputed5_members": disputed,
        "sealed_family_overlaps": overlap,
        "n_monitor_groups": len(mon),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def numpy_radial_log_det(u: np.ndarray) -> np.ndarray:
    """log sech²(r) + 2 log(tanh(r)/r) per atom (small-r series), numpy."""
    u = np.asarray(u, dtype=np.float64)
    r = np.linalg.norm(u, axis=-1)
    small = -(5.0 / 3.0) * r ** 2
    ratio = np.tanh(r) / np.where(r > 1e-12, r, 1.0)
    naive = np.log(1.0 - np.tanh(r) ** 2) + 2.0 * np.log(ratio)
    return np.where(r < 1e-3, small, naive)


def latent_targets_np(actions: np.ndarray, action_max: float) -> np.ndarray:
    """``u* = R⁻¹(a*/a_max)`` (numpy; loud rejection on a_max violations)."""
    z = np.asarray(actions, dtype=np.float64) / float(action_max)
    norms = np.linalg.norm(z, axis=-1, keepdims=True)
    if norms.size and float(norms.max()) >= 1.0:
        raise MonitorGateError(
            "A_MAX_VIOLATION",
            f"|a*|/a_max = {float(norms.max()):.12f} >= 1")
    r = np.arctanh(np.clip(norms, 0.0, 1.0 - 1e-7))
    return r * z / np.maximum(norms, 1e-12)


def fit_same_density_baseline_mle(
    fit_states: Sequence[dict], *, action_max: float,
) -> dict:
    """Global MLE μ̄/σ̄ of the state-independent radial-tanh Gaussian.

    Fit on fit50 teacher latents ONLY (all valid non-teacher atoms pooled).
    Because the pushforward density ``log π_z(u) = log N(u; μ̄, σ̄) −
    log_det(u)`` has a data-constant log-det, the MLE is the plain Gaussian
    moment match: ``μ̄ = mean(u*)``, ``σ̄² = mean(‖u* − μ̄‖²)/3`` (scalar σ
    shared by all three axes).  Evaluated downstream with the IDENTICAL
    transform (C4 §5 item 2).
    """
    us = []
    for s in fit_states:
        if s["is_teacher"] or s["action"] is None:
            continue
        us.append(latent_targets_np(np.asarray(s["action"]), action_max))
    if not us:
        raise MonitorGateError("BASELINE_EMPTY", "no fit labels to fit")
    u = np.concatenate(us, axis=0)
    mu_bar = u.mean(axis=0)
    sigma_bar = float(math.sqrt(
        float(((u - mu_bar) ** 2).sum() / (3 * u.shape[0]))))
    payload = {
        "mu_bar": [float(v) for v in mu_bar],
        "sigma_bar": sigma_bar,
        "n_atoms": int(u.shape[0]),
        "fit_rule": "global MLE on fit50 teacher latents only: "
                    "mu_bar = mean(u*), sigma_bar^2 = mean(|u*-mu_bar|^2)/3 "
                    "(log_det(u*) is a data constant)",
        "action_max": float(action_max),
    }
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def same_density_per_atom(mu_bar: np.ndarray, sigma_bar: float,
                          u_star: np.ndarray) -> np.ndarray:
    """Per-atom NLL of the same-density baseline AT the teacher latent."""
    resid = u_star - mu_bar
    return (_LOG_2PI * 1.5 + 3.0 * math.log(sigma_bar)
            + 0.5 * (resid * resid).sum(-1) / sigma_bar ** 2
            + numpy_radial_log_det(u_star))


def same_density_deterministic(mu_bar, action_max: float) -> np.ndarray:
    """``a_det = a_max·tanh(‖μ̄‖)μ̄/‖μ̄‖`` — constant across states/atoms."""
    mu_bar = np.asarray(mu_bar, dtype=np.float64)
    n = float(np.linalg.norm(mu_bar))
    return action_max * math.tanh(n) * mu_bar / max(n, 1e-12)


# ---------------------------------------------------------------------------
# Per-group evaluation (numpy metric core + torch model forward)
# ---------------------------------------------------------------------------

def _state_atom_meta(states: Sequence[dict]) -> dict:
    """Per-atom alignment arrays for a group's collated state order."""
    n_atoms = [int(np.asarray(s["atomic_numbers"]).size) for s in states]
    branches, ral = [], []
    for s, n in zip(states, n_atoms):
        branches.extend([s["branch_id"]] * n)
        ral.extend([s["remaining_arc_length"]] * n)
    return {"branches": np.asarray(branches), "ral": np.asarray(ral),
            "n_atoms": n_atoms}


def metrics_block(
    *,
    a_det: np.ndarray,
    a_star: np.ndarray,
    u_star: np.ndarray,
    nll_model: np.ndarray,
    nll_same_density: np.ndarray,
    atom_mask: np.ndarray,
    dir_mask: np.ndarray,
    zero_baseline: bool = False,
) -> dict:
    """All registered metric values over ONE atom population.

    Zero-action baseline handling: with ``zero_baseline=True`` the "model"
    action is a=0, so MSE == zero baseline and the ratio is 1 by
    construction; the direction cosine is undefined (a=0) and reported as
    ``None``.  NLL of the zero baseline is undefined (no density) → ``None``.
    """
    n = int(atom_mask.sum())
    if n == 0:
        return {"n_atoms": 0}
    mse_model = float((((a_det - a_star) ** 2).sum(-1))[atom_mask].mean())
    mse_zero = float((a_star[atom_mask] ** 2).sum(-1).mean())
    out: dict = {
        "n_atoms": n,
        "mse_model_A2": mse_model,
        "mse_zero_baseline_A2": mse_zero,
        "mse_ratio_vs_zero": mse_model / mse_zero,
    }
    dmask = atom_mask & dir_mask
    if zero_baseline:
        # a = 0 has no direction and no density: cosine + model NLL undefined
        out["n_atoms_direction"] = int(dmask.sum())
        out["direction_cosine_absa_weighted"] = None
    elif bool(dmask.any()):
        ad, ast = a_det[dmask], a_star[dmask]
        w = np.linalg.norm(ast, axis=1)
        cos = (ad * ast).sum(1) / np.maximum(
            np.linalg.norm(ad, axis=1) * w, 1e-12)
        out["n_atoms_direction"] = int(dmask.sum())
        out["direction_cosine_absa_weighted"] = float(
            (w * cos).sum() / w.sum())
    else:
        out["n_atoms_direction"] = 0
        out["direction_cosine_absa_weighted"] = None
    out["nll_model_per_atom"] = (
        None if zero_baseline else float(nll_model[atom_mask].mean()))
    out["nll_same_density_per_atom"] = float(
        nll_same_density[atom_mask].mean())
    return out


def sigma_pathology_summary(
    sigma: np.ndarray, u_norm: np.ndarray, near_rest: np.ndarray,
) -> dict:
    """σ-vs-|u*| summary + the two registered pathology flags."""
    def q(a, p):
        return float(np.quantile(a, p)) if a.size else float("nan")

    if sigma.size:
        frac_floor = float(
            (sigma <= SIGMA_FLOOR_VALUE * (1.0 + 1e-3)).mean())
        sigma_min = float(sigma.min())
    else:
        frac_floor, sigma_min = 0.0, float("nan")
    pinned = bool(sigma.size
                  and frac_floor >= SIGMA_FLOOR_FRACTION_FLAG
                  and sigma_min <= SIGMA_FLOOR_VALUE * (1.0 + 1e-3))
    if bool(near_rest.any()) and near_rest.size == sigma.size:
        med_sigma = float(np.median(sigma[near_rest]))
        med_unorm = float(np.median(u_norm[near_rest]))
        dominates = bool(med_sigma > SIGMA_DOMINATES_FACTOR * med_unorm)
    else:
        med_sigma = med_unorm = float("nan")
        dominates = False
    return {
        "sigma_mean": float(sigma.mean()) if sigma.size else float("nan"),
        "sigma_median": q(sigma, 0.5),
        "sigma_q90": q(sigma, 0.9),
        "sigma_min": sigma_min,
        "abs_ustar_median": q(u_norm, 0.5),
        "abs_ustar_q90": q(u_norm, 0.9),
        "fraction_at_floor": frac_floor,
        "near_rest_sigma_median": med_sigma,
        "near_rest_abs_ustar_median": med_unorm,
        "pathology": {
            "sigma_pinned_at_floor": pinned,
            "sigma_dominates_near_rest": dominates,
        },
    }


def evaluate_group(
    *,
    group_id: str,
    states: Sequence[dict],
    model_mu: np.ndarray,
    model_sigma: np.ndarray,
    model_nll: np.ndarray,
    action_max: float,
    moving_threshold: float,
    same_density: dict,
    disputed_ids: Sequence[str] = (),
    near_ts_arc_fraction: float,
) -> dict:
    """All preregistered metrics for ONE group.

    ``model_mu``/``model_sigma``/``model_nll`` are per-atom arrays aligned
    with the collated order of ``states``.  Stratum thresholds are the
    FROZEN fit-measured constants (reused, never re-measured — C5); the
    near-TS arc rule is the frozen per-branch definition; near-rest =
    frozen 0.01 Å near-zero mask; moving = frozen fit-population q75 of
    |a*|.
    """
    targets = build_targets(list(states), action_max)
    a_star = targets["a_star"].numpy().astype(np.float64)
    valid = targets["atom_valid"].numpy()
    dir_mask = targets["dir_mask"].numpy()
    u_star = targets["u_star"].numpy().astype(np.float64)
    mu_norm = np.linalg.norm(model_mu, axis=1, keepdims=True)
    a_det = action_max * np.tanh(mu_norm) * model_mu / np.maximum(
        mu_norm, 1e-12)
    nll_sd = same_density_per_atom(
        np.asarray(same_density["mu_bar"]), same_density["sigma_bar"],
        u_star)
    meta = _state_atom_meta(states)
    a_norm = np.linalg.norm(a_star, axis=1)

    near_ts = np.zeros(a_star.shape[0], dtype=bool)
    for branch in set(meta["branches"].tolist()):
        bidx = meta["branches"] == branch
        ral = meta["ral"][bidx]
        near_ts[bidx] = ral <= near_ts_arc_fraction * float(ral.max())
    moving = a_norm >= float(moving_threshold)
    near_rest = valid & ~dir_mask  # the frozen 0.01 Å mask == near-rest

    disputed_set = set(disputed_ids)
    dis_positions = []
    ofs = 0
    for s, n in zip(states, meta["n_atoms"]):
        if s["reaction_group_id"] in disputed_set:
            dis_positions.extend(range(ofs, ofs + n))
        ofs += n
    disputed_mask = np.zeros(a_star.shape[0], dtype=bool)
    if dis_positions:
        disputed_mask[np.asarray(dis_positions, dtype=int)] = True

    def block(extra_mask: np.ndarray, zero_baseline: bool = False) -> dict:
        return metrics_block(
            a_det=a_det if not zero_baseline
            else np.zeros_like(a_det),
            a_star=a_star, u_star=u_star,
            nll_model=model_nll, nll_same_density=nll_sd,
            atom_mask=valid & extra_mask, dir_mask=dir_mask,
            zero_baseline=zero_baseline)

    return {
        "group_id": group_id,
        "n_states": len(states),
        "n_action_states": sum(
            1 for s in states if not s["is_teacher"]),
        "all": block(np.ones_like(valid, dtype=bool)),
        "strata": {
            "near_ts": block(near_ts),
            "moving": block(moving),
            "near_rest": block(near_rest),
        },
        "zero_baseline_reference": block(
            np.ones_like(valid, dtype=bool), zero_baseline=True),
        "sigma_summary": sigma_pathology_summary(
            model_sigma[valid], np.linalg.norm(u_star, axis=1)[valid],
            near_rest[valid]),
        "disputed5_slice": {
            "n_disputed_atoms": int(disputed_mask.sum()),
            "disputed_group_ids": sorted(
                {s["reaction_group_id"] for s in states
                 if s["reaction_group_id"] in disputed_set}),
        },
    }


def summarize_over_groups(per_group: Sequence[dict]) -> dict:
    """Median + per-group sign consistency over the 8 groups (no pooling)."""
    def col(path):
        vals = []
        for g in per_group:
            node = g
            for key in path:
                node = node[key]
            vals.append(node)
        return vals

    ratio = col(["all", "mse_ratio_vs_zero"])
    cos = col(["all", "direction_cosine_absa_weighted"])
    finite_ratio = [v for v in ratio if v is not None and math.isfinite(v)]
    cos_finite = [v for v in cos if v is not None]
    strata_summary = {}
    for stratum in ("near_ts", "moving", "near_rest"):
        sratio = col(["strata", stratum, "mse_ratio_vs_zero"])
        scos = col(["strata", stratum, "direction_cosine_absa_weighted"])
        scos_f = [v for v in scos if v is not None]
        strata_summary[stratum] = {
            "mse_ratio_median": float(np.median(
                [v for v in sratio if v is not None]))
            if any(v is not None for v in sratio) else float("nan"),
            "direction_cosine_median": (
                float(np.median(scos_f)) if scos_f else None),
        }
    pathology = {
        "sigma_pinned_at_floor_groups": [
            g["group_id"] for g in per_group
            if g["sigma_summary"]["pathology"]["sigma_pinned_at_floor"]],
        "sigma_dominates_near_rest_groups": [
            g["group_id"] for g in per_group
            if g["sigma_summary"]["pathology"]["sigma_dominates_near_rest"]],
    }
    return {
        "n_groups": len(per_group),
        "mse_ratio_per_group": ratio,
        "mse_ratio_median": float(np.median(finite_ratio))
        if finite_ratio else float("nan"),
        "mse_ratio_min": float(np.min(finite_ratio))
        if finite_ratio else float("nan"),
        "mse_ratio_max": float(np.max(finite_ratio))
        if finite_ratio else float("nan"),
        "n_groups_ratio_below_one": int(
            sum(1 for v in finite_ratio if v < 1.0)),
        "direction_cosine_per_group": cos,
        "direction_cosine_median": float(np.median(cos_finite))
        if cos_finite else None,
        "n_groups_cosine_positive": int(sum(1 for v in cos_finite if v > 0)),
        "nll_per_atom_per_group": col(["all", "nll_model_per_atom"]),
        "nll_same_density_per_group": col(
            ["all", "nll_same_density_per_atom"]),
        "strata": strata_summary,
        "pathology": pathology,
    }


def c3_trigger_evaluation(summary: dict) -> dict:
    """C3 registered trigger on MONITOR evidence (never fit numbers)."""
    cos_med = summary["direction_cosine_median"]
    near_ts_cos = summary["strata"]["near_ts"]["direction_cosine_median"]
    if cos_med is None and near_ts_cos is None:
        return {"status": "not_evaluable",
                "reason": "no monitor direction evidence"}
    joint = (cos_med is not None and cos_med < C3_TRIGGER_JOINT_COS)
    near_ts = (near_ts_cos is not None
               and near_ts_cos < C3_TRIGGER_NEAR_TS_COS)
    return {
        "status": "triggered" if (joint or near_ts) else "not_triggered",
        "joint_cosine_median": cos_med,
        "joint_threshold": C3_TRIGGER_JOINT_COS,
        "joint_fired": bool(joint),
        "near_ts_cosine_median": near_ts_cos,
        "near_ts_threshold": C3_TRIGGER_NEAR_TS_COS,
        "near_ts_fired": bool(near_ts),
    }


def apply_selection_rule(
    summaries: dict[str, dict],
    thresholds: dict,
) -> dict:
    """Frozen model-selection rule over recipe → summary (monitor only)."""
    rule = thresholds["selection_rule"]
    gate = thresholds["gate"]
    decisions = {}
    eligible: list[tuple[str, float]] = []
    for recipe, s in sorted(summaries.items()):
        ratio = s["mse_ratio_median"]
        cos = s["direction_cosine_median"]
        cos_pos = s["n_groups_cosine_positive"]
        ratio_below = s["n_groups_ratio_below_one"]
        pathology = bool(s["pathology"]["sigma_pinned_at_floor_groups"]
                         or s["pathology"]["sigma_dominates_near_rest_groups"])
        gate_pass = bool(
            math.isfinite(ratio) and ratio < gate["median_mse_ratio_max"]
            and ratio_below >= gate["n_groups_ratio_below_one_min"]
            and cos is not None and cos >= gate["median_cosine_min"]
            and (s["strata"]["near_ts"]["direction_cosine_median"] is not None
                 and s["strata"]["near_ts"]["direction_cosine_median"]
                 >= gate["near_ts_cosine_min"]))
        eligibility_ok = bool(
            cos is not None and cos > rule["eligibility_median_cosine_min"]
            and cos_pos >= rule["eligibility_n_cosine_positive_min"])
        decisions[recipe] = {
            "gate_decision": "PASS" if gate_pass else "BELOW_GATE",
            "selection_eligible": eligibility_ok and not pathology,
            "sigma_pathology": pathology,
        }
        if eligibility_ok and not pathology and math.isfinite(ratio):
            eligible.append((recipe, ratio))
    selected = None
    if eligible:
        best = min(r for _, r in eligible)
        tied = sorted(
            rec for rec, r in eligible if r <= best * (1.0 + rule["tie_band"])
            and r <= best + rule["tie_band"])
        order = rule["simplicity_order"]
        selected = min(tied, key=lambda rec: order.index(rec)
                       if rec in order else len(order))
    return {
        "schema_version": SELECTION_RECORD_SCHEMA,
        "rule": thresholds["selection_rule"],
        "per_recipe": decisions,
        "selected_recipe": selected,
        "selection_metric": "monitor8 median per-group MSE ratio vs the "
                            "zero-action baseline (NEVER min NLL, §I.9)",
    }


def load_monitor_states(
    features_dir: str | Path,
    observed_dir: str | Path,
    *,
    expected_group_ids: Sequence[str],
):
    """Monitor shard loading via the SAME loader/pipeline gate as fit."""
    try:
        return load_dual_ts_states(
            features_dir, observed_dir,
            expected_group_ids=expected_group_ids)
    except FileNotFoundError as exc:
        raise MonitorGateError(
            "MONITOR_DATA_MISSING",
            f"monitor8 features/observations not found ({exc}) — building "
            f"them requires the WP2-style xTB observation batch (separately "
            f"billed QC; forbidden in this lane)") from exc
    except NotADirectoryError as exc:
        raise MonitorGateError(
            "MONITOR_DATA_MISSING", f"monitor data directory missing "
            f"({exc})") from exc


def forward_policy_per_atom(
    torch,
    policy,
    states: Sequence[dict],
    *,
    action_max: float,
    device=None,
    batch_size: int = 256,
) -> dict:
    """Per-atom μ/σ/NLL arrays for a state list (collation order aligned).

    The model NLL is the C1 teacher-latent NLL at ``u* = R⁻¹(a*/a_max)`` —
    identical definition to training.
    """
    mus, sigmas, nlls = [], [], []
    if device is None:
        device = next(policy.parameters()).device
    for i in range(0, len(states), batch_size):
        chunk = list(states[i:i + batch_size])
        targets = build_targets(chunk, action_max)
        for key in ("u_star", "a_star", "atom_valid", "dir_mask",
                    "cell_index", "cell_group", "state_of_atom"):
            targets[key] = targets[key].to(device)
        # reuse the dual-EGNN collation for the model inputs themselves
        from pes2ts_core.pesn.models.dual_egnn import collate_dual_states
        model_inputs = collate_dual_states(torch, chunk)
        for k, v in model_inputs.items():
            if torch.is_tensor(v):
                if v.dtype in (torch.float64, torch.float32):
                    v = v.to(dtype=torch.float32)
                model_inputs[k] = v.to(device)
        out = policy(model_inputs)
        nll = per_atom_nll(policy, out["mu"], out["log_sigma"],
                           targets["u_star"])
        mus.append(out["mu"].detach().cpu().numpy().astype(np.float64))
        sigmas.append(out["sigma"].detach().cpu().numpy().astype(np.float64))
        nlls.append(nll.detach().cpu().numpy().astype(np.float64))
    return {
        "mu": np.concatenate(mus, axis=0),
        "sigma": np.concatenate(sigmas, axis=0),
        "nll": np.concatenate(nlls, axis=0),
    }


def evaluate_groups_with_policy(
    torch,
    policy,
    states: Sequence[dict],
    *,
    group_ids: Sequence[str],
    action_max: float,
    moving_threshold: float,
    same_density: dict,
    disputed_ids: Sequence[str] = (),
    near_ts_arc_fraction: float,
    device=None,
) -> list[dict]:
    """Full preregistered evaluation: forward pass + per-group metrics."""
    by_group: dict[str, list[dict]] = {g: [] for g in group_ids}
    for s in states:
        by_group.setdefault(s["reaction_group_id"], []).append(s)
    per_group = []
    for gid in group_ids:
        gstates = by_group.get(gid, [])
        if not gstates:
            raise MonitorGateError(
                "MONITOR_GROUP_EMPTY", f"no states for group {gid}")
        arrays = forward_policy_per_atom(
            torch, policy, gstates, action_max=action_max, device=device)
        per_group.append(evaluate_group(
            group_id=gid, states=gstates, model_mu=arrays["mu"],
            model_sigma=arrays["sigma"], model_nll=arrays["nll"],
            action_max=action_max, moving_threshold=moving_threshold,
            same_density=same_density, disputed_ids=disputed_ids,
            near_ts_arc_fraction=near_ts_arc_fraction))
    return per_group


__all__ = [
    "C3_TRIGGER_JOINT_COS",
    "C3_TRIGGER_NEAR_TS_COS",
    "MONITOR_EVAL_SCHEMA",
    "MONITOR_PREREG_SCHEMA",
    "MONITOR_THRESHOLDS_SCHEMA",
    "MonitorGateError",
    "SELECTION_RECORD_SCHEMA",
    "SIGMA_DOMINATES_FACTOR",
    "SIGMA_FLOOR_FRACTION_FLAG",
    "apply_selection_rule",
    "assert_fit_monitor_disjoint",
    "c3_trigger_evaluation",
    "evaluate_group",
    "evaluate_groups_with_policy",
    "fit_same_density_baseline_mle",
    "gate_prereg",
    "latent_targets_np",
    "load_dual_ts_states",
    "load_monitor_states",
    "load_prereg",
    "metrics_block",
    "numpy_radial_log_det",
    "same_density_deterministic",
    "same_density_per_atom",
    "sha256_file",
    "sigma_pathology_summary",
    "summarize_over_groups",
]