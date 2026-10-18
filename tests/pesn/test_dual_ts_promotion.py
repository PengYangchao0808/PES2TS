"""Lane D §9.1 / §9.2 promotion + raw/EMA selection tests (pure dict).

Covers every §9.1 clause and its boundary (cos gain, near-TS drop, group
ratio +0.01, pooled +0.02, near-zero +10%, parent-zero ⇒ no increase,
better-than-zero count, candidate ordering), the §9.2 40k gate conditions +
timing, and raw/EMA selection with ties.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.evaluation.dual_ts_promotion import (  # noqa: E402
    PromotionError,
    apply_40k_gate,
    apply_promotion_rule,
    rank_promotion_candidates,
    select_raw_or_ema,
)

S1 = 20261017
S2 = 20261018


def _dev(overall_cos=0.20, near_ts_cos=0.30, group_median_ratio=0.90,
         better_than_zero_groups=5, pooled_ratio=0.85, near_zero_rmse_A=0.05):
    return {
        "overall_cos": overall_cos,
        "near_ts_cos": near_ts_cos,
        "group_median_ratio": group_median_ratio,
        "better_than_zero_groups": better_than_zero_groups,
        "pooled_ratio": pooled_ratio,
        "near_zero_rmse_A": near_zero_rmse_A,
    }


def _pair(parent, child):
    return ({S1: parent, S2: parent}, {S1: child, S2: child})


def _clear_child(parent):
    """A child that clearly beats the parent on every clause."""
    return _dev(overall_cos=parent["overall_cos"] + 0.05,
                near_ts_cos=parent["near_ts_cos"] + 0.02,
                group_median_ratio=parent["group_median_ratio"] - 0.05,
                better_than_zero_groups=parent["better_than_zero_groups"] + 1,
                pooled_ratio=parent["pooled_ratio"] - 0.05,
                near_zero_rmse_A=parent["near_zero_rmse_A"] * 0.9)


# ---------------------------------------------------------------------------
# §9.1 pairwise promotion
# ---------------------------------------------------------------------------

def test_promote_when_all_clauses_pass():
    parent = _dev()
    child = _clear_child(parent)
    pdev, cdev = _pair(parent, child)
    r = apply_promotion_rule(pdev, cdev, [S1, S2])
    assert r["promote"] is True
    assert r["decision"] == "PROMOTE"
    assert r["reasons"] == []


def test_one_seed_cos_bad_retains_parent():
    parent = _dev()
    good = _clear_child(parent)
    bad = _dev(overall_cos=parent["overall_cos"] - 0.05)  # cos drops
    pdev = {S1: parent, S2: parent}
    cdev = {S1: good, S2: bad}
    r = apply_promotion_rule(pdev, cdev, [S1, S2])
    assert r["promote"] is False
    assert r["decision"] == "RETAIN_PARENT"
    assert any("overall_cos_gain" in reason for reason in r["reasons"])


def test_cos_gain_boundary_plus_001():
    parent = _dev(overall_cos=0.20)
    child = _clear_child(parent)
    child["overall_cos"] = 0.20 + 0.01  # exactly +0.01 → passes (>= 0.01)
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is True


def test_near_ts_drop_boundary():
    parent = _dev(near_ts_cos=0.30)
    child = _clear_child(parent)
    child["near_ts_cos"] = 0.30 - 0.01  # exactly -0.01 → passes (not > 0.01)
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is True
    # -0.0100001 → fail
    child["near_ts_cos"] = 0.30 - 0.0101
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is False


def test_group_ratio_increase_boundary():
    parent = _dev(group_median_ratio=0.90)
    child = _clear_child(parent)
    child["group_median_ratio"] = 0.90 + 0.01  # +0.01 exactly → passes
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is True
    child["group_median_ratio"] = 0.90 + 0.0101  # +0.0101 → fail
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is False


def test_pooled_ratio_increase_boundary():
    parent = _dev(pooled_ratio=0.85)
    child = _clear_child(parent)
    child["pooled_ratio"] = 0.85 + 0.02  # +0.02 exactly → passes
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is True
    child["pooled_ratio"] = 0.85 + 0.0201  # +0.0201 → fail
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is False


def test_near_zero_rmse_plus_10_percent_boundary():
    parent = _dev(near_zero_rmse_A=1.0)
    child = _clear_child(parent)
    child["near_zero_rmse_A"] = 1.0 * 1.10  # +10% exactly → passes
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is True
    child["near_zero_rmse_A"] = 1.0 * 1.10 + 0.001  # > +10% → fail
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is False


def test_parent_zero_near_zero_must_not_increase():
    parent = _dev(near_zero_rmse_A=0.0)
    child = _clear_child(parent)
    child["near_zero_rmse_A"] = 0.0  # no increase → passes
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is True
    child["near_zero_rmse_A"] = 0.001  # parent zero → any increase fails
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is False


def test_better_than_zero_groups_not_decrease():
    parent = _dev(better_than_zero_groups=6)
    child = _clear_child(parent)
    child["better_than_zero_groups"] = 5  # decrease → fail
    _, cdev = _pair(parent, child)
    r = apply_promotion_rule({S1: parent, S2: parent}, cdev, [S1, S2])
    assert r["promote"] is False


def test_rank_promotion_candidates_ordering():
    cands = [
        {"name": "high_mse", "group_median_ratio": 0.9, "overall_cos": 0.5,
         "n_parameters": 100},
        {"name": "low_mse", "group_median_ratio": 0.5, "overall_cos": 0.1,
         "n_parameters": 500},
        {"name": "tie_ratio_higher_cos", "group_median_ratio": 0.5,
         "overall_cos": 0.9, "n_parameters": 300},
        {"name": "tie_ratio_fewer_params", "group_median_ratio": 0.5,
         "overall_cos": 0.9, "n_parameters": 200},
    ]
    ranked = rank_promotion_candidates(cands)
    # group-median MSE ↓ first
    assert ranked[0]["name"] in ("low_mse", "tie_ratio_higher_cos",
                                 "tie_ratio_fewer_params")
    # among the 0.5 ratio group, cos ↑ then fewer params
    names = [c["name"] for c in ranked]
    assert names.index("tie_ratio_higher_cos") < names.index("low_mse")
    assert names.index("tie_ratio_fewer_params") < names.index(
        "tie_ratio_higher_cos")
    # highest MSE last
    assert names[-1] == "high_mse"


# ---------------------------------------------------------------------------
# §9.2 40k gate
# ---------------------------------------------------------------------------

def _curve(r10, r20):
    return {"u10000": {"group_median_ratio": r10[0], "overall_cos": r10[1],
                       "near_ts_cos": r10[2], "near_zero_rmse_A": r10[3]},
            "u20000": {"group_median_ratio": r20[0], "overall_cos": r20[1],
                       "near_ts_cos": r20[2], "near_zero_rmse_A": r20[3]}}


def _good_timing(seed_seconds=60.0):
    return {"seed_40k_seconds": seed_seconds, "n_seeds": 3,
            "budget_minutes": 95.0}


def test_40k_gate_all_pass():
    curve = _curve((0.90, 0.20, 0.30, 0.05), (0.85, 0.22, 0.31, 0.05))
    r = apply_40k_gate({S1: curve, S2: curve}, _good_timing())
    assert r["allow_40k"] is True
    assert r["decision"] == "ALLOW_40K"


def test_40k_gate_group_ratio_not_dropping():
    curve = _curve((0.90, 0.20, 0.30, 0.05), (0.90, 0.22, 0.31, 0.05))
    r = apply_40k_gate({S1: curve, S2: curve}, _good_timing())
    assert r["allow_40k"] is False
    # exact -0.01 drop passes (>= 0.01)
    curve = _curve((0.90, 0.20, 0.30, 0.05), (0.89, 0.22, 0.31, 0.05))
    r = apply_40k_gate({S1: curve, S2: curve}, _good_timing())
    assert r["allow_40k"] is True


def test_40k_gate_cos_drop():
    curve = _curve((0.90, 0.20, 0.30, 0.05), (0.85, 0.18, 0.31, 0.05))
    r = apply_40k_gate({S1: curve, S2: curve}, _good_timing())
    assert r["allow_40k"] is False


def test_40k_gate_near_zero_rmse_increase():
    # +10% exactly (0.05 → 0.055) → allowed
    curve = _curve((0.90, 0.20, 0.30, 0.05), (0.85, 0.22, 0.31, 0.055))
    assert apply_40k_gate({S1: curve, S2: curve},
                          _good_timing())["allow_40k"] is True
    # > +10% (0.05 → 0.056) → fail
    curve = _curve((0.90, 0.20, 0.30, 0.05), (0.85, 0.22, 0.31, 0.056))
    assert apply_40k_gate({S1: curve, S2: curve},
                          _good_timing())["allow_40k"] is False


def test_40k_gate_timing_not_fitting():
    curve = _curve((0.90, 0.20, 0.30, 0.05), (0.85, 0.22, 0.31, 0.05))
    # 3 seeds * 2000s = 6000s > 95*60 = 5700s
    r = apply_40k_gate({S1: curve, S2: curve},
                       _good_timing(seed_seconds=2000.0))
    assert r["allow_40k"] is False
    assert r["decision"] == "STAY_20K"


def test_40k_gate_requires_timing():
    curve = _curve((0.90, 0.20, 0.30, 0.05), (0.85, 0.22, 0.31, 0.05))
    with pytest.raises(PromotionError):
        apply_40k_gate({S1: curve}, {})


# ---------------------------------------------------------------------------
# raw/EMA selection
# ---------------------------------------------------------------------------

def test_select_raw_or_ema_prefers_lower_mse():
    better = {"group_median_ratio": 0.8, "overall_cos": 0.1}
    worse = {"group_median_ratio": 0.9, "overall_cos": 0.9}
    assert select_raw_or_ema(better, worse) == "raw"
    assert select_raw_or_ema(worse, better) == "ema"


def test_select_raw_or_ema_tie_then_cos():
    raw = {"group_median_ratio": 0.8, "overall_cos": 0.5}
    ema = {"group_median_ratio": 0.8, "overall_cos": 0.4}
    assert select_raw_or_ema(raw, ema) == "raw"
    assert select_raw_or_ema(ema, raw) == "ema"


def test_select_raw_or_ema_full_tie_returns_raw():
    raw = {"group_median_ratio": 0.8, "overall_cos": 0.4}
    ema = {"group_median_ratio": 0.8, "overall_cos": 0.4}
    assert select_raw_or_ema(raw, ema) == "raw"
    assert select_raw_or_ema(ema, raw) == "raw"
