"""G5 zero-QC diagnosis slice — metric unit tests (synthetic, offline).

Covers the deterministic building blocks of
``scripts/diagnose_wp4_hard_groups.py`` WITHOUT touching disk artifacts:
  * group_feature_profile: length normalization (atom-mean per state) +
    state-mean profile on a synthetic shard;
  * nearest_fit_distance: exact nearest-group selection, exact L2, and the
    per-channel absolute-difference vector on a hand-checkable case;
  * geometry_arc_profile: near-TS stratum rule (remaining_arc <= 0.2 *
    branch max), step counts, and max |a*| on a synthetic teacher payload.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / (
    "scripts/diagnose_wp4_hard_groups.py")
_spec = importlib.util.spec_from_file_location("diagnose_wp4_hard_groups",
                                               _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _synthetic_shard(n_states: int, n_atoms: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ns = rng.normal(size=(n_states, n_atoms, 5))
    return {
        "node_scalars": ns,
        "x_angstrom": rng.normal(size=(n_states, n_atoms, 3)),
    }


def test_group_feature_profile_length_normalization():
    # atom-mean per state, then state-mean: verify against a hand roll.
    shard = _synthetic_shard(n_states=5, n_atoms=4, seed=0)
    prof, per_state = mod.group_feature_profile(shard)
    assert per_state.shape == (5, 5)
    assert prof.shape == (5,)
    manual = shard["node_scalars"].mean(axis=1)     # (5,5) atom-mean
    assert np.allclose(per_state, manual)
    assert np.allclose(prof, manual.mean(axis=0))


def test_group_feature_profile_atom_count_invariance():
    # a state duplicated across a different atom count (constant block)
    # yields the SAME profile: the metric is length-normalized.
    block = np.full((3, 5, 5), 0.5)
    block_small = np.full((3, 2, 5), 0.5)
    p_big, _ = mod.group_feature_profile({"node_scalars": block})
    p_small, _ = mod.group_feature_profile({"node_scalars": block_small})
    assert np.allclose(p_big, p_small)


def test_nearest_fit_distance_exact():
    fit = {
        "F1": np.zeros(5),
        "F2": np.ones(5) * 2.0,
    }
    prof = np.ones(5) * 1.9          # nearest to F2 (0.707.. < 5.05..)
    per_state = np.stack([prof, prof])
    out = mod.nearest_fit_distance(prof, per_state, fit)
    assert out["nearest_fit_group"] == "F2"
    expect = float(np.linalg.norm(prof - fit["F2"]))
    assert out["profile_l2_to_nearest_fit"] == pytest.approx(expect)
    assert out["mean_per_state_l2_to_nearest_fit_profile"] == \
        pytest.approx(expect)
    # per-channel |Δ| = 0.1 on all 5 channels (prof 1.9 vs F2 2.0)
    assert out["per_channel_abs_diff_to_nearest_fit"] == \
        pytest.approx([0.1] * 5)
    # deterministic tie handling: equal distances -> lexicographically
    # first name wins
    fit_eq = {"A": np.zeros(5), "B": np.zeros(5)}
    out_eq = mod.nearest_fit_distance(np.zeros(5), np.zeros((1, 5)),
                                      fit_eq)
    assert out_eq["nearest_fit_group"] == "A"


def _synthetic_teacher() -> dict:
    """Two branches; branch max arc 10 (brA) / 4 (brB).

    brA: 6 states, remaining arcs 10,8,6,4,2,0 (steps of 2.0); the last
    state is the TS endpoint (arc 0, teacher_endpoint=True). The arc-2.0
    state is inside the frozen 0.2*10 near-TS stratum, and the step FROM
    it (2.0 -> 0.0, successor = endpoint) is counted: the endpoint check
    applies to the STEP SOURCE only (the approach step is real motion).
    brB: 5 states, remaining arcs 4,3,2,1,0 (steps of 1.0); 0.2*4 = 0.8
    is never reached on a non-endpoint source, so brB contributes ZERO
    near-TS steps.
    """
    states, labels = [], []
    for br, (n, arc0, step) in {"A": (6, 10.0, 2.0),
                                "B": (5, 4.0, 1.0)}.items():
        for i in range(n):
            sid = f"R|{br}|{i:04d}"
            states.append({"state_id": sid, "branch_id": f"R#br{br}",
                           "step_index": i,
                           "atomic_numbers": [1, 6] if br == "A"
                           else [1],
                           "coordinates": [[0.0, 0.0, i * 0.1]
                                           for _ in range(2 if br == "A"
                                                          else 1)]})
            labels.append({
                "state_id": sid,
                "remaining_arc_length": arc0 - i * step,
                "teacher_endpoint": i == n - 1,
                "action_delta": [[0.03, 0.0, 0.0]] * (2 if br == "A"
                                                      else 1),
            })
    return {"states": states, "labels": labels}


def test_geometry_arc_profile_synthetic():
    geo = mod.geometry_arc_profile(_synthetic_teacher())
    assert geo["n_atoms"] == 2           # first state is brA
    assert geo["n_states"] == 11 and geo["n_action_states"] == 9
    assert geo["n_branches"] == 2
    assert geo["branch_lengths_states"] == {"R#brA": 6, "R#brB": 5}
    # endpoints excluded from the remaining-arc population: brA arcs
    # 10..2, brB arcs 4..1 -> min 1.0, max 10.0
    assert geo["remaining_arc_A"]["min"] == pytest.approx(1.0)
    assert geo["remaining_arc_A"]["max"] == pytest.approx(10.0)
    # near-TS brA: arc[i] <= 0.2*10 = 2.0 fires on the arc-2.0 state;
    # the endpoint flag check applies to the STEP SOURCE only, so the
    # approach step (2.0 -> 0.0) IS counted -> exactly 1 step of 2.0.
    assert geo["near_ts_stratum"]["n_steps"] == 1
    assert geo["near_ts_stratum"]["arc_progress_per_step_A"]["median"] == \
        pytest.approx(2.0)
    # brB: max arc 4 -> 0.2*4 = 0.8; smallest non-endpoint arc is 1.0
    # -> no brB near-TS steps.  Deterministic.
    # max |a*|: per-atom L2 of [0.03,0,0] = 0.03 on both branches
    assert geo["max_action_norm_A"] == pytest.approx(0.03)
    # cartesian RMS step near TS on brA: 2 atoms moving 0.1 in z ->
    # RMS = 0.1
    assert geo["near_ts_stratum"]["cartesian_rms_step_A"]["median"] == \
        pytest.approx(0.1)


def test_geometry_arc_profile_endpoint_excluded_from_max_action():
    t = _synthetic_teacher()
    # give an endpoint a huge action; it must NOT drive max |a*|
    for lab in t["labels"]:
        if lab["teacher_endpoint"]:
            lab["action_delta"] = [[9.9, 9.9, 9.9]] * len(lab["action_delta"])
    geo = mod.geometry_arc_profile(t)
    assert geo["max_action_norm_A"] == pytest.approx(0.03)


def test_geometry_arc_profile_total_arc_per_branch():
    geo = mod.geometry_arc_profile(_synthetic_teacher())
    assert geo["branch_total_arc_A"]["max"] == pytest.approx(10.0)
    assert geo["branch_total_arc_A"]["min"] == pytest.approx(4.0)
    assert geo["branch_total_arc_A"]["median"] == pytest.approx(7.0)
