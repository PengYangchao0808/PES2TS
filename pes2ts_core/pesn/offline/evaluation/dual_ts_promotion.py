"""Lane D — V4.2–V4.7 stage-promotion + 40k gate + raw/EMA selection (§9).

Pure-dict, zero-heavy-dependency mechanization of plan §9.1 / §9.2 so the
Lane C campaign driver can import it directly without pulling torch or any
model/data code.

* :func:`apply_promotion_rule` — the §9.1 pairwise child-vs-parent decision
  over paired seeds.  If the child fails any clause the parent is retained.
* :func:`rank_promotion_candidates` — the §9.1 multi-candidate ordering
  (group-median MSE ↓ → overall cos ↑ → fewer parameters) used when several
  children all pass.
* :func:`apply_40k_gate` — the §9.2 V4.7 40k-eligibility gate (10k→20k curve
  + timing fit within 95 min).
* :func:`select_raw_or_ema` — the once-per-night raw-vs-EMA read rule
  (group-median MSE ↓, then cos ↑, tie → raw).

These are engineering screening rules; the thresholds are NOT significance
bounds (plan §9.1 explicit disclaimer).
"""
from __future__ import annotations

import math
from typing import Final, Sequence

#: §9.1 frozen promotion thresholds (plan §9.1 / contract §7; literal).
PROMO_COS_GAIN: Final[float] = 0.01
PROMO_NEAR_TS_COS_DROP_LIMIT: Final[float] = 0.01
PROMO_GROUP_RATIO_INCREASE_LIMIT: Final[float] = 0.01
PROMO_POOLED_RATIO_INCREASE_LIMIT: Final[float] = 0.02
PROMO_NEAR_ZERO_RMSE_INCREASE_FRAC: Final[float] = 0.10

#: §9.2 40k-gate thresholds.
GATE_40K_GROUP_RATIO_DROP_MIN: Final[float] = 0.01
GATE_40K_COS_DROP_LIMIT: Final[float] = 0.01
GATE_40K_NEAR_ZERO_RMSE_INCREASE_FRAC: Final[float] = 0.10
GATE_40K_BUDGET_MINUTES: Final[float] = 95.0
GATE_40K_N_SEEDS: Final[int] = 3

#: Small tolerance for float boundary comparisons (exactly-at-threshold must
#: resolve deterministically; 0.01 is not exactly representable in binary).
_EPS: Final[float] = 1e-9


class PromotionError(RuntimeError):
    """Typed promotion-rule refusal (bad input schema)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# §9.1 pairwise promotion
# ---------------------------------------------------------------------------

def _seed_metrics(dev: dict, seed, *, label: str) -> dict:
    m = dev.get(seed)
    if not isinstance(m, dict):
        raise PromotionError(
            "MISSING_SEED_METRICS",
            f"{label} dev metrics missing seed {seed!r}")
    return m


def _check_clause(cond: bool, name: str, detail: str, out: dict) -> None:
    out["per_clause"][name] = {"pass": bool(cond), "detail": detail}
    if not cond:
        out["reasons"].append(f"{name}: {detail}")


def apply_promotion_rule(parent_dev: dict, child_dev: dict,
                         seeds: Sequence) -> dict:
    """§9.1 pairwise child-vs-parent promotion over paired seeds.

    ``parent_dev`` / ``child_dev`` map each seed to a dev-metrics dict with
    keys::

        overall_cos           # |a*|-weighted direction cosine (higher better)
        near_ts_cos           # near-TS direction cosine (higher better)
        group_median_ratio    # group-median MSE ratio vs zero (lower better)
        better_than_zero_groups  # count of groups below ratio 1.0
        pooled_ratio          # atom-pooled MSE ratio vs zero (lower better)
        near_zero_rmse_A      # near-zero atom absolute RMSE (Å)

    The child promotes only if, for EVERY seed: overall cos gains ≥ 0.01;
    near-TS cos does not drop > 0.01; group-median ratio does not rise >
    0.01; better-than-zero group count does not fall; pooled ratio does not
    rise > 0.02; and near-zero RMSE does not rise > 10% (parent-zero ⇒ child
    must not increase).  Otherwise the parent is retained (``promote=False``).
    """
    seeds = [s for s in seeds]
    if not seeds:
        raise PromotionError("NO_SEEDS", "at least one seed is required")

    out: dict = {
        "promote": True,
        "reasons": [],
        "metrics_used": ["overall_cos", "near_ts_cos", "group_median_ratio",
                         "better_than_zero_groups", "pooled_ratio",
                         "near_zero_rmse_A"],
        "per_clause": {},
        "per_seed": {},
    }

    for seed in seeds:
        pm = _seed_metrics(parent_dev, seed, label="parent")
        cm = _seed_metrics(child_dev, seed, label="child")
        s_out: dict = {"per_clause": {}, "reasons": []}

        # -- overall cos: child - parent >= 0.01 ----------------------------
        dcos = _num(cm.get("overall_cos")) - _num(pm.get("overall_cos"))
        ok_cos = dcos >= PROMO_COS_GAIN - _EPS
        s_out["per_clause"]["overall_cos_gain"] = {
            "parent": pm.get("overall_cos"), "child": cm.get("overall_cos"),
            "delta": dcos, "pass": bool(ok_cos),
        }

        # -- near-TS cos: not drop > 0.01 -----------------------------------
        dnt = _num(cm.get("near_ts_cos")) - _num(pm.get("near_ts_cos"))
        ok_nt = dnt >= -PROMO_NEAR_TS_COS_DROP_LIMIT - _EPS
        s_out["per_clause"]["near_ts_cos_drop"] = {
            "parent": pm.get("near_ts_cos"), "child": cm.get("near_ts_cos"),
            "delta": dnt, "pass": bool(ok_nt),
        }

        # -- group-median ratio: not rise > 0.01 ----------------------------
        dgr = _num(cm.get("group_median_ratio")) - _num(pm.get("group_median_ratio"))
        ok_gr = dgr <= PROMO_GROUP_RATIO_INCREASE_LIMIT + _EPS
        s_out["per_clause"]["group_median_ratio_increase"] = {
            "parent": pm.get("group_median_ratio"),
            "child": cm.get("group_median_ratio"),
            "delta": dgr, "pass": bool(ok_gr),
        }

        # -- better-than-zero group count: not decrease ---------------------
        p_btz = _int(pm.get("better_than_zero_groups"))
        c_btz = _int(cm.get("better_than_zero_groups"))
        ok_btz = c_btz >= p_btz
        s_out["per_clause"]["better_than_zero_groups"] = {
            "parent": p_btz, "child": c_btz, "pass": bool(ok_btz),
        }

        # -- pooled ratio: not rise > 0.02 ----------------------------------
        dpr = _num(cm.get("pooled_ratio")) - _num(pm.get("pooled_ratio"))
        ok_pr = dpr <= PROMO_POOLED_RATIO_INCREASE_LIMIT + _EPS
        s_out["per_clause"]["pooled_ratio_increase"] = {
            "parent": pm.get("pooled_ratio"), "child": cm.get("pooled_ratio"),
            "delta": dpr, "pass": bool(ok_pr),
        }

        # -- near-zero RMSE: not rise > 10%; parent-zero => not increase ----
        p_nz = _num(pm.get("near_zero_rmse_A"))
        c_nz = _num(cm.get("near_zero_rmse_A"))
        ok_nz = _near_zero_ok(p_nz, c_nz)
        s_out["per_clause"]["near_zero_rmse"] = {
            "parent": pm.get("near_zero_rmse_A"),
            "child": cm.get("near_zero_rmse_A"),
            "pass": bool(ok_nz),
        }

        seed_pass = all(ok for ok in (ok_cos, ok_nt, ok_gr, ok_btz, ok_pr, ok_nz))
        if not seed_pass:
            for name, clause in s_out["per_clause"].items():
                if not clause["pass"]:
                    s_out["reasons"].append(f"{name}: {clause}")
                    out["reasons"].append(f"seed {seed} :: {name}: {clause}")
        out["per_seed"][seed] = s_out
        out["promote"] = out["promote"] and seed_pass

    out["promote"] = bool(out["promote"])
    out["decision"] = "PROMOTE" if out["promote"] else "RETAIN_PARENT"
    out["rule_note"] = (
        "§9.1 engineering screening rule; thresholds are NOT significance "
        "bounds; multiple passing candidates are ordered by "
        "rank_promotion_candidates (group-median MSE ↓ → cos ↑ → params ↓)")
    return out


def _num(v) -> float:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        raise PromotionError("NONFINITE_METRIC",
                             f"metric is None or non-finite: {v!r}")
    return float(v)


def _int(v) -> int:
    if v is None:
        raise PromotionError("NONFINITE_METRIC", "metric is None")
    return int(v)


def _near_zero_ok(parent: float, child: float) -> bool:
    if parent == 0.0:
        return child <= _EPS
    return child <= parent * (1.0 + PROMO_NEAR_ZERO_RMSE_INCREASE_FRAC) \
        * (1.0 + _EPS) + _EPS


# ---------------------------------------------------------------------------
# §9.1 multi-candidate ordering
# ---------------------------------------------------------------------------

def rank_promotion_candidates(candidates: Sequence[dict]) -> list[dict]:
    """Order passing candidates: group-median MSE ↓ → overall cos ↑ → fewer params.

    ``candidates`` is a list of dicts each carrying at least::

        {"name": str, "group_median_ratio": float, "overall_cos": float,
         "n_parameters": int | None}

    Returns the list sorted best-first (stable for equal keys).
    """
    def key(c: dict):
        ratio = c.get("group_median_ratio")
        cos = c.get("overall_cos")
        nparams = c.get("n_parameters")
        ratio_k = ratio if ratio is not None else float("inf")
        cos_k = cos if cos is not None else float("-inf")
        # fewer parameters preferred; unknown count sorts last
        n_k = nparams if nparams is not None else float("inf")
        return (ratio_k, -cos_k, n_k, str(c.get("name", "")))
    return sorted(list(candidates), key=key)


# ---------------------------------------------------------------------------
# §9.2 40k gate
# ---------------------------------------------------------------------------

def apply_40k_gate(winner_dev_curves: dict, timing: dict) -> dict:
    """§9.2 V4.7 40k-eligibility gate.

    ``winner_dev_curves`` maps each seed to a dict with ``u10000`` / ``u20000``
    dev metrics::

        {"group_median_ratio", "overall_cos", "near_ts_cos", "near_zero_rmse_A"}

    ``timing`` carries::

        {"seed_40k_seconds": float,   # one seed @40k wall incl. eval
         "n_seeds": int = 3, "budget_minutes": float = 95.0}

    Allow 40k only if, for BOTH seeds: group-median ratio drops ≥ 0.01 from
    10k→20k; overall and near-TS cos each do not drop > 0.01; near-zero RMSE
    does not rise > 10%; AND three seeds @40k including eval fit within the
    95-minute budget.
    """
    seeds = [s for s in winner_dev_curves]
    if not seeds:
        raise PromotionError("NO_SEEDS", "winner_dev_curves is empty")
    n_seeds = int(timing.get("n_seeds", GATE_40K_N_SEEDS))
    budget_min = float(timing.get("budget_minutes", GATE_40K_BUDGET_MINUTES))
    seed_seconds = timing.get("seed_40k_seconds")
    if seed_seconds is None:
        raise PromotionError(
            "TIMING_MISSING", "timing['seed_40k_seconds'] is required")

    out: dict = {"allow_40k": True, "reasons": [], "per_seed": {}}

    for seed in seeds:
        curve = winner_dev_curves.get(seed)
        if not isinstance(curve, dict) or "u10000" not in curve \
                or "u20000" not in curve:
            raise PromotionError(
                "CURVE_MISSING", f"seed {seed!r} missing u10000/u20000 curves")
        k10 = curve["u10000"]
        k20 = curve["u20000"]
        s_out: dict = {}

        d_ratio = _num(k20.get("group_median_ratio")) \
            - _num(k10.get("group_median_ratio"))
        ok_ratio = d_ratio <= -GATE_40K_GROUP_RATIO_DROP_MIN + _EPS
        s_out["group_median_ratio_drop"] = {
            "u10000": k10.get("group_median_ratio"),
            "u20000": k20.get("group_median_ratio"),
            "delta": d_ratio, "pass": bool(ok_ratio),
        }

        d_cos = _num(k20.get("overall_cos")) - _num(k10.get("overall_cos"))
        ok_cos = d_cos >= -GATE_40K_COS_DROP_LIMIT - _EPS
        s_out["overall_cos_drop"] = {
            "u10000": k10.get("overall_cos"), "u20000": k20.get("overall_cos"),
            "delta": d_cos, "pass": bool(ok_cos),
        }

        d_nt = _num(k20.get("near_ts_cos")) - _num(k10.get("near_ts_cos"))
        ok_nt = d_nt >= -GATE_40K_COS_DROP_LIMIT - _EPS
        s_out["near_ts_cos_drop"] = {
            "u10000": k10.get("near_ts_cos"), "u20000": k20.get("near_ts_cos"),
            "delta": d_nt, "pass": bool(ok_nt),
        }

        p_nz = _num(k10.get("near_zero_rmse_A"))
        c_nz = _num(k20.get("near_zero_rmse_A"))
        ok_nz = c_nz <= p_nz * (1.0 + GATE_40K_NEAR_ZERO_RMSE_INCREASE_FRAC) \
            * (1.0 + _EPS) + _EPS
        s_out["near_zero_rmse"] = {
            "u10000": k10.get("near_zero_rmse_A"),
            "u20000": k20.get("near_zero_rmse_A"), "pass": bool(ok_nz),
        }

        seed_pass = all((ok_ratio, ok_cos, ok_nt, ok_nz))
        if not seed_pass:
            for name, clause in s_out.items():
                if not clause["pass"]:
                    out["reasons"].append(f"seed {seed} :: {name}: {clause}")
        out["per_seed"][seed] = s_out
        out["allow_40k"] = out["allow_40k"] and seed_pass

    # -- timing gate --------------------------------------------------------
    total_seconds = n_seeds * float(seed_seconds)
    budget_seconds = budget_min * 60.0
    ok_time = total_seconds <= budget_seconds + _EPS
    out["timing"] = {
        "n_seeds": n_seeds,
        "seed_40k_seconds": float(seed_seconds),
        "total_40k_seconds": float(total_seconds),
        "budget_minutes": budget_min,
        "budget_seconds": budget_seconds,
        "pass": bool(ok_time),
    }
    if not ok_time:
        out["reasons"].append(
            f"timing: {n_seeds} seeds @40k = {total_seconds:.1f}s > "
            f"{budget_seconds:.1f}s budget")
    out["allow_40k"] = bool(out["allow_40k"] and ok_time)
    out["decision"] = "ALLOW_40K" if out["allow_40k"] else "STAY_20K"
    out["rule_note"] = (
        "§9.2 40k gate is engineering evidence for extending training, NOT a "
        "guarantee that 40k beats 20k; on budget failure fall back to 20k "
        "uniformly (plan §9.2)")
    return out


# ---------------------------------------------------------------------------
# raw-vs-EMA selection
# ---------------------------------------------------------------------------

def select_raw_or_ema(dev_raw: dict, dev_ema: dict) -> str:
    """Raw-vs-EMA read rule: group-median MSE ↓ first, then overall cos ↑, tie → raw.

    ``dev_raw`` / ``dev_ema`` are dev-metrics dicts with ``group_median_ratio``
    (lower better) and ``overall_cos`` (higher better).  Returns ``"raw"`` or
    ``"ema"``.
    """
    rr = _num(dev_raw.get("group_median_ratio"))
    er = _num(dev_ema.get("group_median_ratio"))
    if rr < er - _EPS:
        return "raw"
    if er < rr - _EPS:
        return "ema"
    rc = _num(dev_raw.get("overall_cos"))
    ec = _num(dev_ema.get("overall_cos"))
    if rc > ec + _EPS:
        return "raw"
    if ec > rc + _EPS:
        return "ema"
    return "raw"  # tie → raw (deterministic default)


__all__ = [
    "GATE_40K_BUDGET_MINUTES",
    "GATE_40K_COS_DROP_LIMIT",
    "GATE_40K_GROUP_RATIO_DROP_MIN",
    "GATE_40K_NEAR_ZERO_RMSE_INCREASE_FRAC",
    "GATE_40K_N_SEEDS",
    "PROMO_COS_GAIN",
    "PROMO_GROUP_RATIO_INCREASE_LIMIT",
    "PROMO_NEAR_TS_COS_DROP_LIMIT",
    "PROMO_NEAR_ZERO_RMSE_INCREASE_FRAC",
    "PROMO_POOLED_RATIO_INCREASE_LIMIT",
    "PromotionError",
    "apply_40k_gate",
    "apply_promotion_rule",
    "rank_promotion_candidates",
    "select_raw_or_ema",
]
