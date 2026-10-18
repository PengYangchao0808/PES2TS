"""WP4-A dual-TS BC trainer tests (synthetic; G4 C1–C4 + plan §I.10).

Covers: teacher-latent NLL bounded below + gradient μ→u* (analytic +
finite-difference + Adam descent), fixed §I.10 reduction order, near-zero
mask exclusion from L_dir but inclusion in L_a, epoch accounting, prereg
gate refusals (missing / digest-changed / data-missing), σ-floor pathology
detector, same-density baseline MLE, leakage assertions, and bundle +
checkpoint identity on a tiny synthetic population.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from pes2ts_core.pesn.contracts.dual_ts import (
    validate_record,
)
from pes2ts_core.pesn.models.dual_ts_policy import (
    DEFAULT_A_MAX_ANGSTROM,
    DEFAULT_SIGMA_RANGE,
    DualTSPolicy,
)
from pes2ts_core.pesn.offline.training.dual_ts_bc import (
    DENSITY_TARGET_VERSION,
    DualTSBCSettings,
    DualTSBCTrainer,
    DualTSBCTrainerError,
    build_targets,
    epoch_state_order,
    latent_targets,
    load_dual_ts_states,
    per_atom_action_mse,
    per_atom_dir_loss,
    per_atom_nll,
    reduce_step_sums,
)
from pes2ts_core.pesn.offline.training.dual_ts_monitor import (
    MONITOR_PREREG_SCHEMA,
    MonitorGateError,
    apply_selection_rule,
    assert_fit_monitor_disjoint,
    c3_trigger_evaluation,
    fit_same_density_baseline_mle,
    gate_prereg,
    latent_targets_np,
    load_prereg,
    metrics_block,
    numpy_radial_log_det,
    same_density_deterministic,
    same_density_per_atom,
    sigma_pathology_summary,
    summarize_over_groups,
)

SIGMA_MIN = DEFAULT_SIGMA_RANGE[0]


# ---------------------------------------------------------------------------
# Synthetic dual-TS states
# ---------------------------------------------------------------------------

def make_state(rng, group_id="RXN_G", branch_id="brA", n_atoms=6,
               state_id="s0", is_teacher=False, action=None,
               remaining_arc=10.0):
    z = rng.integers(1, 9, size=n_atoms)
    node_scalars = np.concatenate([
        z[:, None].astype(float),
        rng.normal(size=(n_atoms, 1)),
        np.full((n_atoms, 1), 0.4),
        np.zeros((n_atoms, 2)),
    ], axis=1)  # 5 channels: z_id, |g|/s_g, dE/s_dE, charge, mult
    near_zero = (np.linalg.norm(action, axis=1) < 0.01) \
        if action is not None else np.zeros(n_atoms, dtype=bool)
    # union edge set (i<j): (0,1), (1,2) — directed both ways, feats (2E, 4)
    edge_index = np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    edge_feats = rng.normal(size=(4, 4))
    return {
        "state_id": state_id,
        "reaction_group_id": group_id,
        "branch_id": branch_id,
        "episode_id": f"EP_{branch_id}",
        "step_index": 0,
        "is_teacher": is_teacher,
        "remaining_arc_length": remaining_arc,
        "action": action,
        "near_zero_mask": near_zero,
        "atomic_numbers": z,
        "node_scalars": node_scalars,
        "x_angstrom": rng.normal(scale=2.0, size=(n_atoms, 3)),
        "g_scaled": rng.normal(size=(n_atoms, 3)),
        "atom_feats_a": rng.normal(size=(n_atoms, 4)),
        "atom_feats_b": rng.normal(size=(n_atoms, 4)),
        "edge_feats_a": edge_feats,
        "edge_feats_b": edge_feats,
        "edge_index": edge_index,
    }


def make_population(seed=7):
    rng = np.random.default_rng(seed)
    states = []
    for g in ("RXN_A", "RXN_B", "RXN_C"):
        for br in ("brA", "brB"):
            for k in range(3):
                direction = rng.normal(size=(6, 3))
                direction /= np.maximum(
                    np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
                magnitude = rng.uniform(0.008, 0.045, size=(6, 1))
                action = direction * magnitude
                if k == 0:  # two near-rest atoms (below the frozen 0.01 A)
                    action[0] = 0.004
                    action[1] = 0.003
                states.append(make_state(
                    rng, g, br, state_id=f"{g}_{br}_{k}",
                    action=action, remaining_arc=10.0 - 3.0 * k))
    return states


# ---------------------------------------------------------------------------
# C1: teacher-latent NLL bounded below + gradient pushes μ → u*
# ---------------------------------------------------------------------------

def test_teacher_latent_nll_bounded_below():
    """NLL >= 1.5 log2π + 3 log σ_min + log_det(u*) for ANY μ (G4 §2)."""
    rng = np.random.default_rng(0)
    u_star = torch.as_tensor(rng.normal(scale=0.2, size=(64, 3)))
    log_sigma = torch.full((64,), math.log(SIGMA_MIN))
    floor = torch.as_tensor(
        1.5 * math.log(2 * math.pi) + 3.0 * math.log(SIGMA_MIN)
        + numpy_radial_log_det(u_star.numpy()), dtype=torch.float64)
    for shift in (0.0, 0.5, 2.0, 10.0):
        mu = u_star + shift
        nll = per_atom_nll(_StubPolicy(), mu, log_sigma, u_star)
        assert bool((nll.detach().to(torch.float64) >= floor - 1e-4).all())
    # and the bound is attained at mu = u*, sigma = sigma_min
    nll = per_atom_nll(_StubPolicy(), u_star.clone(), log_sigma, u_star)
    assert torch.allclose(nll.detach(), floor, atol=1e-5)


class _StubPolicy:
    """log_prob_z without an encoder (pure head math on tensors)."""

    def log_prob_z(self, mu, log_sigma, u):
        from pes2ts_core.pesn.models.direct_policy import (
            radial_log_det_per_atom)
        sigma = log_sigma.exp().unsqueeze(-1)
        resid = (u - mu) / sigma
        gaussian = (-1.5 * math.log(2 * math.pi)
                    - 3.0 * torch.log(sigma).squeeze(-1)
                    - 0.5 * (resid * resid).sum(-1))
        log_det = radial_log_det_per_atom(torch, u)
        return {"per_atom": gaussian - log_det, "log_det": log_det,
                "gaussian": gaussian}


def test_nll_gradient_pushes_mu_toward_ustar():
    """Analytic + finite-difference + descent: ∂NLL/∂μ = (μ−u*)/σ²."""
    rng = np.random.default_rng(1)
    u_star = torch.as_tensor(rng.normal(scale=0.2, size=(8, 3)))
    mu0 = u_star + 0.5
    mu = mu0.clone().requires_grad_(True)
    log_sigma = torch.full((8,), math.log(0.05))
    nll = per_atom_nll(_StubPolicy(), mu, log_sigma, u_star).sum()
    nll.backward()
    sigma = math.exp(float(log_sigma[0]))
    expected = (mu0 - u_star) / sigma ** 2
    assert torch.allclose(mu.grad, expected, atol=1e-5)
    # finite difference on one coordinate of atom 0
    mu_p = mu0.clone(); mu_p[0, 0] += 1e-5
    mu_m = mu0.clone(); mu_m[0, 0] -= 1e-5
    g_fd = (
        per_atom_nll(_StubPolicy(), mu_p, log_sigma,
                     u_star).detach().sum()
        - per_atom_nll(_StubPolicy(), mu_m, log_sigma,
                       u_star).detach().sum()) / 2e-5
    assert abs(float(g_fd) - float(expected[0, 0])) < 1e-3
    # Adam descent on NLL-only moves μ → u* (G4: "gradients drive μ→u*")
    mu = (u_star + 0.5).clone().requires_grad_(True)
    opt = torch.optim.Adam([mu], lr=0.05)
    d0 = float((mu.detach() - u_star).norm())
    for _ in range(300):
        opt.zero_grad()
        loss = per_atom_nll(_StubPolicy(), mu, log_sigma.detach(),
                            u_star).sum()
        loss.backward()
        opt.step()
    d1 = float((mu.detach() - u_star).norm())
    assert d1 < 0.1 * d0


# ---------------------------------------------------------------------------
# Fixed §I.10 reduction
# ---------------------------------------------------------------------------

def test_reduction_order_branch_mean_reaction_mean_group_mean():
    """Flat atom-mean ≠ per-state sum → branch mean → reaction mean →
    group mean; the trainer's reduction matches the hand-computed value."""
    t = torch
    # 2 groups; g1 has branches b1 (1 state, loss 6) and b2 (1 state, loss 2)
    # g2 has one branch (1 state, loss 4)
    state_sums = t.tensor([6.0, 2.0, 4.0])
    cell_index = t.tensor([0, 1, 2])
    cell_group = t.tensor([0, 0, 1])
    loss, group_means = reduce_step_sums(t, state_sums, cell_index, 3,
                                         cell_group, 2)
    # branch means: g1 -> (6 + 2)/2 = 4; g2 -> 4; group mean = 4
    assert float(loss) == pytest.approx(4.0)
    assert float(group_means[0]) == pytest.approx(4.0)
    assert float(group_means[1]) == pytest.approx(4.0)
    # flat per-state mean would be 4.0 here, so make it distinguishable:
    state_sums = t.tensor([9.0, 3.0, 4.0])  # g1 branches 9 and 3 -> 6; g2 4
    loss, _ = reduce_step_sums(t, state_sums, cell_index, 3, cell_group, 2)
    assert float(loss) == pytest.approx(5.0)  # (6 + 4)/2, not 16/3
    # per-reaction mean over AVAILABLE branches only (partial steps)
    state_sums = t.tensor([9.0])
    cell_index = t.tensor([0])
    loss, group_means = reduce_step_sums(t, state_sums, cell_index, 3,
                                         cell_group, 2)
    assert float(loss) == pytest.approx(9.0)
    # absent groups hold a 0 placeholder and are excluded from the loss
    assert float(group_means[1]) == 0.0


def test_per_state_sums_and_recipe_composition():
    """L_a includes near-zero atoms; L_dir excludes them (C2 mask rule)."""
    states = make_population()
    targets = build_targets(states, DEFAULT_A_MAX_ANGSTROM)
    n_atoms = targets["a_star"].shape[0]
    rng = np.random.default_rng(3)
    a_det = torch.as_tensor(rng.normal(size=(n_atoms, 3)) * 0.05)
    a_star = targets["a_star"]
    mse_atom = per_atom_action_mse(a_det, a_star)
    dir_atom = per_atom_dir_loss(a_det, a_star)
    near_zero = ~targets["dir_mask"]
    assert bool(near_zero.any())
    # near-zero atoms are INSIDE L_a
    assert float(mse_atom[near_zero].abs().sum()) > 0.0
    # and OUTSIDE L_dir (their weight ||a*|| is ~0 by the mask rule, but the
    # exclusion is enforced structurally: masked terms are exactly 0)
    assert float(dir_atom[~targets["atom_valid"]].abs().sum()) == 0.0
    zero_terms = dir_atom[near_zero & targets["atom_valid"]]
    # magnitude weight is ||a*|| < 0.01 → each term < 0.01 * 2
    assert float(zero_terms.abs().max()) < 0.02
    # dir loss on well-masked atoms equals w * (1 - cos)
    m = targets["dir_mask"]
    w = a_star[m].norm(dim=-1)
    cos = torch.nn.functional.cosine_similarity(
        a_det[m], a_star[m], dim=-1)
    assert torch.allclose(dir_atom[m], w * (1 - cos), atol=1e-6)


# ---------------------------------------------------------------------------
# §I.10 sampler + epoch accounting
# ---------------------------------------------------------------------------

def test_epoch_order_group_balanced_arc_stratified():
    states = make_population()
    order, counts = epoch_state_order(states, seed=11, epoch=0)
    assert len(order) == len(states)
    assert set(counts.values()) == {6}  # 6 drawable states per group
    assert sum(counts.values()) == len(states)
    # every group drawn exactly once per epoch — group ids appear in
    # contiguous runs of the stream (group-major draw order)
    group_seq = [s["reaction_group_id"] for s in order]
    runs = [group_seq[0]] + [
        group_seq[i] for i in range(1, len(group_seq))
        if group_seq[i] != group_seq[i - 1]]
    assert len(runs) == 3  # 3 groups, each contiguous
    # arc-stratified within branch: strata are equal-count by remaining-arc
    # rank ascending and the near-TS stratum (smallest arc) is emitted FIRST
    for g in ("RXN_A", "RXN_B", "RXN_C"):
        for br in ("brA", "brB"):
            seq = [s["remaining_arc_length"] for s in order
                   if s["reaction_group_id"] == g and s["branch_id"] == br]
            assert seq == sorted(seq)  # 3 states -> 3 strata of 1, ascending
    # teacher states never drawn
    teacher = make_state(np.random.default_rng(0), is_teacher=True,
                         action=None)
    order2, counts2 = epoch_state_order(
        states + [teacher], seed=11, epoch=0)
    assert all(s is not teacher for s in order2)
    assert sum(counts2.values()) == len(states)


def test_trainer_epoch_accounting_and_records(tmp_path):
    states = make_population()
    settings = DualTSBCSettings(
        recipe="bc_d", budget_updates=5, batch_states=4, seed=5,
        device="cpu", require_cuda=False, curve_every=1, eval_every=0)
    trainer = DualTSBCTrainer(
        settings=settings, states=states,
        data_meta={"spec_digest": "0" * 64, "shard_digests": [],
                   "norm_params": {"s_g_hartree_per_angstrom": 0.0154,
                                   "s_delta_e": 0.0789}},
        run_dir=tmp_path)
    record = trainer.train()
    assert record["accounting"]["updates_run"] == 5
    # unified group-round-robin: 5 updates x batch 4 = 20 draws (NOT the old
    # full-epoch traversal of 18 states)
    assert record["accounting"]["samples_exposed"] == 20
    assert record["accounting"]["batch_states"] == 4
    assert record["accounting"]["drawable_states"] == 18
    assert record["accounting"]["n_groups"] == 3
    assert record["accounting"]["nll_min_max_per_state"][1] < 1e4
    # full exposure accounting present (no first-3+last truncation)
    exp = record["exposure"]
    assert exp["drawable_states"] == 18 and exp["n_groups"] == 3
    assert exp["total_draws"] == 20
    assert 0 < exp["distinct_states"] <= 18
    # accounting identities hold exactly (repeat = (draws - distinct)/draws)
    assert exp["coverage"] == pytest.approx(exp["distinct_states"] / 18.0)
    assert exp["repeat_rate"] == pytest.approx(
        (20.0 - exp["distinct_states"]) / 20.0)
    assert 0.0 <= exp["near_ts_fraction"] <= 1.0
    assert "per_group" in exp and set(exp["per_group"]) == \
        {"RXN_A", "RXN_B", "RXN_C"}
    for g, pg in exp["per_group"].items():
        assert pg["drawable_states"] == 6
        assert pg["draws"] > 0
        assert pg["distinct_states"] <= 6
    assert len(exp["history"]) >= 2  # >=1 epoch snapshot + 1 final
    assert any(h["kind"] == "final" for h in exp["history"])
    assert len(record["history"]) == 5
    # v2 checkpoint identity (IQL format); v1 remains at checkpoint_last.pt
    import torch as _t
    ckpt = _t.load(tmp_path / "checkpoint_v2.pt", map_location="cpu",
                   weights_only=False)
    assert ckpt["schema_version"] == "pesn_dual_ts_bc_checkpoint_v2"
    assert (tmp_path / "checkpoint_last.pt").is_file()  # v1 compat kept
    for key in ("torch_rng_state", "numpy_rng_state", "python_rng_state",
                "optimizer_state", "model_state", "identity", "content_digest"):
        assert key in ckpt
    ident = ckpt["identity"]
    assert ident["seed"] == 5 and ident["recipe"] == "bc_d"
    assert ident["lambda_a"] == 1.0 and ident["lambda_dir"] == 1.0
    assert ident["features_spec_digest"] == "0" * 64
    assert ident["sampler_config"]["sampling_rule"].startswith(
        "reaction_group_balanced")
    assert ckpt["extra"]["density_target_version"] == DENSITY_TARGET_VERSION
    assert ckpt["extra"]["settings"]["recipe"] == "bc_d"
    assert ckpt["extra"]["signed_mu_gate"] is False


def test_trainer_bundle_validates(tmp_path):
    states = make_population()
    settings = DualTSBCSettings(
        recipe="bc_nll", budget_updates=2, batch_states=8, seed=5,
        device="cpu", require_cuda=False, curve_every=1, eval_every=0)
    trainer = DualTSBCTrainer(
        settings=settings, states=states,
        data_meta={"spec_digest": "0" * 64, "shard_digests": ["a" * 64],
                   "norm_params": {"s_g_hartree_per_angstrom": 0.0154,
                                   "s_delta_e": 0.0789}},
        run_dir=tmp_path)
    record = trainer.train()
    bundle = trainer.build_bundle(record)
    payload = bundle.to_payload()
    validate_record("pesn_dual_ts_bundle_v2", payload)
    assert DENSITY_TARGET_VERSION in payload["action_version"]
    assert payload["hyperparams"]["lambda_a"] == 1.0
    assert payload["hyperparams"]["lambda_dir"] == 1.0
    assert payload["input_eligibility"]["fresh_init"].startswith(
        "no WP3 checkpoint warm start")
    assert payload["seeds"] == {"torch": 5, "numpy": 5, "python": 5}


def test_settings_freeze_guards():
    with pytest.raises(DualTSBCTrainerError):
        DualTSBCSettings(recipe="bc_x")
    with pytest.raises(DualTSBCTrainerError):
        DualTSBCSettings(sigma_min=0.001)
    # λ is no longer hard-frozen (v4.1 arm B may unfreeze λ_dir): finite
    # non-negative values are accepted at the settings level; registration
    # is enforced separately (resolve_lambda_registration / CLI).
    assert DualTSBCSettings(lambda_a=1000.0).lambda_a == 1000.0
    assert DualTSBCSettings(lambda_dir=1000.0).lambda_dir == 1000.0
    with pytest.raises(DualTSBCTrainerError):
        DualTSBCSettings(lambda_a=float("nan"))
    with pytest.raises(DualTSBCTrainerError):
        DualTSBCSettings(lambda_dir=-1.0)


# ---------------------------------------------------------------------------
# Same-density baseline + σ pathology
# ---------------------------------------------------------------------------

def test_same_density_baseline_mle_recovers_gaussian():
    rng = np.random.default_rng(4)
    mu0 = np.asarray([0.01, -0.02, 0.005])
    sigma0 = 0.08
    u = rng.normal(loc=mu0, scale=sigma0, size=(2000, 3))
    # map u -> z -> a (forward radial transform, numpy)
    r = np.linalg.norm(u, axis=1, keepdims=True)
    z = np.tanh(r) * u / np.maximum(r, 1e-12)
    a = DEFAULT_A_MAX_ANGSTROM * z
    states = [make_state(rng, action=np.tile(aa, (6, 1)),
                         state_id=f"s{i}")
              for i, aa in enumerate(a)]
    base = fit_same_density_baseline_mle(states,
                                         action_max=DEFAULT_A_MAX_ANGSTROM)
    assert np.allclose(base["mu_bar"], mu0, atol=0.01)
    assert abs(base["sigma_bar"] - sigma0) < 0.01
    # same-density evaluation identity: NLL at u* matches the analytic form
    u_star = latent_targets_np(a[0][None, ...], DEFAULT_A_MAX_ANGSTROM)[0]
    nll = same_density_per_atom(base["mu_bar"], base["sigma_bar"],
                                u_star[None])[0]
    resid = u_star - base["mu_bar"]
    expect = (1.5 * math.log(2 * math.pi)
              + 3 * math.log(base["sigma_bar"])
              + 0.5 * float((resid ** 2).sum()) / base["sigma_bar"] ** 2
              + float(numpy_radial_log_det(u_star[None])[0]))
    assert nll == pytest.approx(expect, rel=1e-9)
    det = same_density_deterministic(base["mu_bar"], DEFAULT_A_MAX_ANGSTROM)
    assert det.shape == (3,)  # constant action vector (state-independent)


def test_sigma_pathology_flags():
    healthy = sigma_pathology_summary(
        sigma=np.full(100, 0.09), u_norm=np.full(100, 0.23),
        near_rest=np.zeros(100, dtype=bool))
    assert healthy["pathology"] == {"sigma_pinned_at_floor": False,
                                    "sigma_dominates_near_rest": False}
    pinned = sigma_pathology_summary(
        sigma=np.full(100, 0.02), u_norm=np.full(100, 0.23),
        near_rest=np.zeros(100, dtype=bool))
    assert pinned["pathology"]["sigma_pinned_at_floor"] is True
    dominates = sigma_pathology_summary(
        sigma=np.full(100, 0.5), u_norm=np.full(100, 0.02),
        near_rest=np.ones(100, dtype=bool))
    assert dominates["pathology"]["sigma_dominates_near_rest"] is True


# ---------------------------------------------------------------------------
# C4 gates: prereg digest + monitor data + leakage + selection
# ---------------------------------------------------------------------------

def test_prereg_gate_refusals(tmp_path):
    missing = tmp_path / "absent.json"
    with pytest.raises(MonitorGateError) as ei:
        load_prereg(missing)
    assert ei.value.code == "PREREG_MISSING"
    with pytest.raises(MonitorGateError) as ei:
        gate_prereg(missing, "0" * 64)
    assert ei.value.code == "PREREG_MISSING"
    payload = {"schema_version": MONITOR_PREREG_SCHEMA,
               "population": {}, "baselines": {}, "metrics": {},
               "sampling": {}, "reduction": {}, "recipes": {},
               "budget": {}, "model_selection": {}}
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(payload))
    _, digest = load_prereg(path)
    with pytest.raises(MonitorGateError) as ei:
        gate_prereg(path, "f" * 64)
    assert ei.value.code == "PREREG_DIGEST_CHANGED"
    got, _ = gate_prereg(path, digest)
    assert got["schema_version"] == MONITOR_PREREG_SCHEMA
    # tampering changes the digest -> refusal
    payload["budget"] = {"updates": 1}
    path.write_text(json.dumps(payload))
    with pytest.raises(MonitorGateError) as ei:
        gate_prereg(path, digest)
    assert ei.value.code == "PREREG_DIGEST_CHANGED"


def test_monitor_data_missing_refusal(tmp_path):
    from pes2ts_core.pesn.offline.training.dual_ts_monitor import (
        load_monitor_states)
    with pytest.raises(MonitorGateError) as ei:
        load_monitor_states(tmp_path / "features",
                            tmp_path / "observed",
                            expected_group_ids=["RXN_X"])
    assert ei.value.code == "MONITOR_DATA_MISSING"


def test_load_states_population_mismatch(tmp_path):
    """The loader gate refuses a shard set that != expected population;
    a missing manifest surfaces as a plain file error."""
    with pytest.raises((FileNotFoundError, DualTSBCTrainerError)):
        load_dual_ts_states(tmp_path / "no_features",
                            tmp_path / "no_observed",
                            expected_group_ids=["RXN_X"])


def test_leakage_assertions():
    lineage = {
        "partitions": {"monitor": {"group_ids": ["RXN_M1", "RXN_M2",
                                                 "RXN_F1"]}},
        "family_audit": {"sealed_reaction_ids": ["RXN_F1"]},
    }
    ev = assert_fit_monitor_disjoint(
        fit_group_ids=["RXN_A", "RXN_B"],
        monitor_group_ids=["RXN_M1", "RXN_M2"],
        lineage=lineage, disputed_ids=["RXN_D1"])
    assert ev["fit_intersection"] == []
    assert ev["n_monitor_groups"] == 2
    with pytest.raises(MonitorGateError) as ei:
        assert_fit_monitor_disjoint(
            fit_group_ids=["RXN_M1"], monitor_group_ids=["RXN_M1"],
            lineage=lineage)
    assert ei.value.code == "FIT_MONITOR_LEAK"
    with pytest.raises(MonitorGateError) as ei:
        assert_fit_monitor_disjoint(
            fit_group_ids=["RXN_A"],
            monitor_group_ids=["RXN_M1", "RXN_OUT"],
            lineage=lineage)
    assert ei.value.code == "MONITOR_PARTITION_MISMATCH"
    with pytest.raises(MonitorGateError) as ei:
        assert_fit_monitor_disjoint(
            fit_group_ids=["RXN_A"], monitor_group_ids=["RXN_M1"],
            lineage=lineage, disputed_ids=["RXN_M1"])
    assert ei.value.code == "DISPUTED_IN_MONITOR"
    with pytest.raises(MonitorGateError) as ei:
        assert_fit_monitor_disjoint(
            fit_group_ids=["RXN_A"],
            monitor_group_ids=["RXN_M1", "RXN_F1"],
            lineage=lineage)
    assert ei.value.code == "FAMILY_SEAL_OVERLAP"


def _fake_summary(ratio, cos, near_ts_cos=None, pathology=False):
    n = 8
    ratios = [ratio] * n
    cosines = [cos] * n
    return {
        "n_groups": n,
        "mse_ratio_per_group": ratios,
        "mse_ratio_median": ratio,
        "mse_ratio_min": ratio, "mse_ratio_max": ratio,
        "n_groups_ratio_below_one": sum(1 for r in ratios if r < 1.0),
        "direction_cosine_per_group": cosines,
        "direction_cosine_median": cos,
        "n_groups_cosine_positive": sum(1 for c in cosines if c > 0),
        "nll_per_atom_per_group": [1.0] * n,
        "nll_same_density_per_group": [1.0] * n,
        "strata": {
            "near_ts": {"mse_ratio_median": ratio,
                        "direction_cosine_median": near_ts_cos},
            "moving": {"mse_ratio_median": ratio,
                       "direction_cosine_median": cos},
            "near_rest": {"mse_ratio_median": ratio,
                          "direction_cosine_median": None},
        },
        "pathology": {
            "sigma_pinned_at_floor_groups": ["RXN_X"] if pathology else [],
            "sigma_dominates_near_rest_groups": [],
        },
    }


THRESHOLDS = {
    "gate": {"median_mse_ratio_max": 1.0, "n_groups_ratio_below_one_min": 6,
             "median_cosine_min": 0.30, "near_ts_cosine_min": 0.40,
             "n_groups": 8},
    "selection_rule": {
        "eligibility_median_cosine_min": 0.0,
        "eligibility_n_cosine_positive_min": 6,
        "metric": "median ratio", "tie_band": 0.01,
        "simplicity_order": ["bc_nll", "bc_a", "bc_d"],
        "exclusions": "sigma pathology"},
}


def test_selection_rule_and_c3_trigger():
    good = {"bc_a": _fake_summary(0.8, 0.45, near_ts_cos=0.5)}
    out = apply_selection_rule(good, THRESHOLDS)
    assert out["per_recipe"]["bc_a"]["gate_decision"] == "PASS"
    assert out["selected_recipe"] == "bc_a"
    assert c3_trigger_evaluation(good["bc_a"])["status"] == "not_triggered"
    # C3 trigger fires on weak direction evidence
    weak = _fake_summary(0.8, 0.25, near_ts_cos=0.25)
    trig = c3_trigger_evaluation(weak)
    assert trig["status"] == "triggered"
    # gate fails on cosine despite good ratio, yet the recipe stays the
    # best eligible selection — and the C3 trigger fires for it
    out = apply_selection_rule({"bc_a": weak}, THRESHOLDS)
    assert out["per_recipe"]["bc_a"]["gate_decision"] == "BELOW_GATE"
    assert out["selected_recipe"] == "bc_a"
    assert c3_trigger_evaluation(weak)["status"] == "triggered"
    # tie band prefers the simpler recipe
    out = apply_selection_rule({
        "bc_d": _fake_summary(0.80, 0.45, near_ts_cos=0.5),
        "bc_nll": _fake_summary(0.805, 0.45, near_ts_cos=0.5),
    }, THRESHOLDS)
    assert out["selected_recipe"] == "bc_nll"
    # sigma pathology excludes a recipe from selection
    out = apply_selection_rule({
        "bc_d": _fake_summary(0.5, 0.45, near_ts_cos=0.5, pathology=True),
        "bc_nll": _fake_summary(0.9, 0.45, near_ts_cos=0.5),
    }, THRESHOLDS)
    assert out["per_recipe"]["bc_d"]["selection_eligible"] is False
    assert out["selected_recipe"] == "bc_nll"


def test_summarize_and_metrics_block_consistency():
    rng = np.random.default_rng(9)
    n = 50
    a_star = rng.normal(scale=0.03, size=(n, 3))
    a_det = a_star * 0.5
    u_star = latent_targets_np(a_star, DEFAULT_A_MAX_ANGSTROM)
    nll_model = rng.normal(size=n)
    nll_sd = rng.normal(size=n)
    mask = np.ones(n, dtype=bool)
    dir_mask = np.linalg.norm(a_star, axis=1) >= 0.01
    block = metrics_block(a_det=a_det, a_star=a_star, u_star=u_star,
                          nll_model=nll_model, nll_same_density=nll_sd,
                          atom_mask=mask, dir_mask=dir_mask)
    assert block["n_atoms"] == n
    assert block["mse_ratio_vs_zero"] == pytest.approx(0.25)
    assert 0 < block["direction_cosine_absa_weighted"] <= 1.0
    group = {
        "group_id": "RXN_T", "n_states": 1, "n_action_states": 1,
        "all": block,
        "strata": {s: dict(block) for s in
                   ("near_ts", "moving", "near_rest")},
        "zero_baseline_reference": metrics_block(
            a_det=np.zeros_like(a_det), a_star=a_star, u_star=u_star,
            nll_model=nll_model, nll_same_density=nll_sd, atom_mask=mask,
            dir_mask=dir_mask, zero_baseline=True),
        "sigma_summary": sigma_pathology_summary(
            np.full(n, 0.05), np.linalg.norm(u_star, axis=1),
            ~dir_mask),
        "disputed5_slice": {"n_disputed_atoms": 0,
                            "disputed_group_ids": []},
    }
    summary = summarize_over_groups([group])
    assert summary["n_groups"] == 1
    assert summary["mse_ratio_median"] == pytest.approx(0.25)
    assert summary["n_groups_ratio_below_one"] == 1
    # zero baseline block: ratio 1 by construction, cosine undefined
    zb = group["zero_baseline_reference"]
    assert zb["mse_ratio_vs_zero"] == pytest.approx(1.0)
    assert zb["direction_cosine_absa_weighted"] is None
    assert zb["nll_model_per_atom"] is None


def test_latent_targets_roundtrip_and_a_max_guard():
    rng = np.random.default_rng(12)
    a = rng.normal(scale=0.02, size=(40, 3))
    u = latent_targets_np(a, DEFAULT_A_MAX_ANGSTROM)
    r = np.linalg.norm(u, axis=1, keepdims=True)
    z = np.tanh(r) * u / np.maximum(r, 1e-12)
    assert np.allclose(z * DEFAULT_A_MAX_ANGSTROM, a, atol=1e-12)
    u_t, z_t = latent_targets(np, a, DEFAULT_A_MAX_ANGSTROM)
    assert np.allclose(u_t, u, atol=1e-12)
    assert np.abs(z_t).max() < 1.0
    with pytest.raises(Exception):
        latent_targets_np(a * 10.0, DEFAULT_A_MAX_ANGSTROM)
    with pytest.raises(DualTSBCTrainerError):
        latent_targets(np, a * 10.0, DEFAULT_A_MAX_ANGSTROM)
