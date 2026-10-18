"""Dual-TS full-model symmetry gates + policy contracts (WP3-B, plan v4 §D).

Gates (plan §D WP3 row: 交换/旋转/平移/原子排列通过), on synthetic states
AND a real WP3-A feature shard, float64, tol 1e-6 (measured ~1e-14):

(a) A/B endpoint swap → identical μ, σ, log-prob, a;
(b) translation invariance;
(c) rotation equivariance: (X, g̃) → (XRᵀ, g̃Rᵀ) ⇒ μ, a → (μRᵀ, aRᵀ),
    σ, log-prob invariant;
(d) permutation equivariance with relabeled atoms/edges.

Policy contracts reuse the v1 radial-tanh patterns: σ = σ_min + softplus
(never-saturating gradient), a = a_max·z, log-prob = gaussian − Jacobian.
The signed-vector-weight variant is a FLAGGED alternative and must pass
the same gates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.models.dual_egnn import (
    DualTSModelConfig,
    collate_dual_states,
)
from pes2ts_core.pesn.models.dual_ts_policy import (
    DEFAULT_A_MAX_ANGSTROM,
    DEFAULT_SIGMA_RANGE,
    DualTSPolicy,
)

torch = require_torch()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_SHARD = (
    PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp3_features_v41"
    / "features_RXN_0000000072.npz"
)

TOL = 1e-6


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

def synthetic_states(n_states: int = 2, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    n, pairs = 6, np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 3], [1, 4]])
    directed = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T
    states = []
    for _ in range(n_states):
        x = rng.normal(size=(n, 3))
        x -= x.mean(axis=0)
        states.append({
            "atomic_numbers": np.array([6, 1, 7, 6, 8, 1]),
            "node_scalars": rng.normal(size=(n, 5)),
            "x_angstrom": x,
            "g_scaled": rng.normal(size=(n, 3)),
            "atom_feats_a": rng.random((n, 4)),
            "atom_feats_b": rng.random((n, 4)),
            "edge_feats_a": rng.random((2 * len(pairs), 4)),
            "edge_feats_b": rng.random((2 * len(pairs), 4)),
            "edge_index": directed,
        })
    return states


def real_shard_states(n_states: int = 3) -> list[dict]:
    if not REAL_SHARD.is_file():
        pytest.skip("real WP3-A feature shard not present (local outputs)")
    shard = np.load(REAL_SHARD)
    return [{
        "atomic_numbers": shard["atomic_numbers"],
        "node_scalars": shard["node_scalars"][row],
        "x_angstrom": shard["x_angstrom"][row],
        "g_scaled": shard["g_scaled"][row],
        "atom_feats_a": shard["atom_feats_a"],
        "atom_feats_b": shard["atom_feats_b"],
        "edge_feats_a": shard["edge_feats_a"],
        "edge_feats_b": shard["edge_feats_b"],
        "edge_index": shard["edge_index"],
    } for row in range(n_states)]


def make_policy(signed: bool = False, seed: int = 11) -> DualTSPolicy:
    torch.manual_seed(seed)
    pol = DualTSPolicy(
        config=DualTSModelConfig(hidden=32, n_layers=2, light=False),
        action_max=DEFAULT_A_MAX_ANGSTROM,
        sigma_range=DEFAULT_SIGMA_RANGE,
        signed_mu_gate=signed,
    ).to(torch.float64)
    pol.train_mode(False)
    return pol


def rotation_matrix(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


def swapped_endpoint_blocks(batch: dict) -> dict:
    out = dict(batch)
    out["atom_feats_a"] = batch["atom_feats_b"].clone()
    out["atom_feats_b"] = batch["atom_feats_a"].clone()
    out["edge_feats_a"] = batch["edge_feats_b"].clone()
    out["edge_feats_b"] = batch["edge_feats_a"].clone()
    return out


def full_outputs(pol: DualTSPolicy, batch: dict, eps=None) -> dict:
    with torch.no_grad():
        out = pol(batch)
        a_det = pol.deterministic(out["mu"])
        gen = torch.Generator().manual_seed(5)
        if eps is None:
            sampled = pol.sample(out["mu"], out["log_sigma"], generator=gen)
        else:
            sigma = out["log_sigma"].exp().unsqueeze(-1)
            u = out["mu"] + sigma * eps
            from pes2ts_core.pesn.models.direct_policy import (
                radial_transform_stable,
            )
            z = radial_transform_stable(torch, u)
            sampled = {"u": u, "z": z, "a": pol.action_max * z}
        lp = pol.log_prob_z(out["mu"], out["log_sigma"], sampled["u"])
    return {"mu": out["mu"], "sigma": out["sigma"],
            "log_sigma": out["log_sigma"], "a": a_det,
            "per_atom": lp["per_atom"], "sampled_a": sampled["a"],
            "sampled_u": sampled["u"],
            "h": out["h"], "v": out["v"]}


# ---------------------------------------------------------------------------
# (a) A/B endpoint swap consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("state_builder", [synthetic_states, real_shard_states])
def test_ab_swap_consistency(signed, state_builder):
    pol = make_policy(signed=signed)
    states = state_builder()
    batch = collate_dual_states(torch, states)
    base = full_outputs(pol, batch)
    swap = full_outputs(pol, swapped_endpoint_blocks(batch))
    for key in ("mu", "sigma", "log_sigma", "a", "per_atom", "sampled_a"):
        err = float((base[key] - swap[key]).abs().max())
        assert err <= TOL, f"A/B swap changed {key} by {err:.3e}"


# ---------------------------------------------------------------------------
# (b) translation invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("state_builder", [synthetic_states, real_shard_states])
def test_translation_invariance(signed, state_builder):
    pol = make_policy(signed=signed)
    batch = collate_dual_states(torch, state_builder())
    base = full_outputs(pol, batch)
    shifted = dict(batch)
    shifted["x"] = batch["x"] + torch.as_tensor([0.37, -1.2, 2.5]).double()
    out = full_outputs(pol, shifted)
    for key in ("mu", "sigma", "log_sigma", "a", "per_atom", "sampled_a"):
        err = float((base[key] - out[key]).abs().max())
        assert err <= TOL, f"translation changed {key} by {err:.3e}"


# ---------------------------------------------------------------------------
# (c) rotation equivariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("state_builder", [synthetic_states, real_shard_states])
def test_rotation_equivariance(signed, state_builder):
    pol = make_policy(signed=signed)
    batch = collate_dual_states(torch, state_builder())
    gen = torch.Generator().manual_seed(5)
    eps = torch.randn(batch["z"].shape[0], 3, generator=gen, dtype=torch.float64)
    base = full_outputs(pol, batch, eps=eps)
    R = torch.as_tensor(rotation_matrix(0.9))
    rotated = dict(batch)
    rotated["x"] = batch["x"] @ R.T
    rotated["g"] = batch["g"] @ R.T
    out = full_outputs(pol, rotated, eps=eps @ R.T)
    # equivariant channels rotate with R
    for key in ("mu", "a", "sampled_a", "v"):
        err = float((base[key] @ R.T - out[key]).abs().max())
        assert err <= TOL, f"rotation equivariance failed for {key}: {err:.3e}"
    # invariant channels unchanged
    for key in ("h", "sigma", "log_sigma"):
        err = float((base[key] - out[key]).abs().max())
        assert err <= TOL, f"rotation invariance failed for {key}: {err:.3e}"
    # log-prob invariant under jointly rotated latent u
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        sampled = pol.sample(base["mu"], base["log_sigma"], generator=gen)
        lp_base = pol.log_prob_z(base["mu"], base["log_sigma"], sampled["u"])
        lp_rot = pol.log_prob_z(out["mu"], out["log_sigma"],
                                sampled["u"] @ R.T)
    err = float((lp_base["per_atom"] - lp_rot["per_atom"]).abs().max())
    assert err <= TOL, f"log-prob not rotation-invariant: {err:.3e}"


# ---------------------------------------------------------------------------
# (d) permutation equivariance (relabeled atoms + edges)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("state_builder", [synthetic_states, real_shard_states])
def test_permutation_equivariance(signed, state_builder):
    pol = make_policy(signed=signed)
    states = state_builder()
    batch = collate_dual_states(torch, states)
    gen = torch.Generator().manual_seed(5)
    eps = torch.randn(batch["z"].shape[0], 3, generator=gen, dtype=torch.float64)
    base = full_outputs(pol, batch, eps=eps)
    rng = np.random.default_rng(9)
    perm_states, perms = [], []
    offset = 0
    for s in states:
        n = int(np.asarray(s["atomic_numbers"]).size)
        perm = rng.permutation(n)
        perms.append(perm + offset)  # global permutation across the batch
        offset += n
        inv = np.argsort(perm)
        perm_states.append({
            "atomic_numbers": np.asarray(s["atomic_numbers"])[perm],
            "node_scalars": np.asarray(s["node_scalars"])[perm],
            "x_angstrom": np.asarray(s["x_angstrom"])[perm],
            "g_scaled": np.asarray(s["g_scaled"])[perm],
            "atom_feats_a": np.asarray(s["atom_feats_a"])[perm],
            "atom_feats_b": np.asarray(s["atom_feats_b"])[perm],
            "edge_feats_a": np.asarray(s["edge_feats_a"]),
            "edge_feats_b": np.asarray(s["edge_feats_b"]),
            "edge_index": inv[np.asarray(s["edge_index"])],  # relabeled edges
        })
    perm_global = np.concatenate(perms)
    out = full_outputs(pol, collate_dual_states(torch, perm_states),
                       eps=eps[perm_global])
    for key in ("mu", "a", "sigma", "log_sigma", "per_atom"):
        err = float((base[key][perm_global] - out[key]).abs().max())
        assert err <= TOL, f"permutation equivariance failed for {key}: {err:.3e}"


# ---------------------------------------------------------------------------
# policy contracts (v1 patterns reused)
# ---------------------------------------------------------------------------

def test_action_carriers_match_contract():
    """u → z = tanh(||u||)u/||u||, a = a_max z; deterministic mode too."""
    pol = make_policy()
    batch = collate_dual_states(torch, real_shard_states())
    with torch.no_grad():
        out = pol(batch)
        gen = torch.Generator().manual_seed(3)
        sampled = pol.sample(out["mu"], out["log_sigma"], generator=gen)
    u, z, a = sampled["u"], sampled["z"], sampled["a"]
    norms = u.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    z_expected = torch.tanh(norms) * u / norms
    assert float((z - z_expected).abs().max()) <= TOL
    assert float((a - DEFAULT_A_MAX_ANGSTROM * z).abs().max()) <= TOL
    assert float(a.norm(dim=-1).max()) <= DEFAULT_A_MAX_ANGSTROM * (1 + 1e-5)
    a_det = pol.deterministic(out["mu"])
    mu_norms = out["mu"].norm(dim=-1, keepdim=True).clamp(min=1e-12)
    det_expected = DEFAULT_A_MAX_ANGSTROM * torch.tanh(mu_norms) * out["mu"] / mu_norms
    assert float((a_det - det_expected).abs().max()) <= TOL
    # zero mu -> zero action (continuous limit, no NaN)
    zero = pol.deterministic(torch.zeros_like(out["mu"]))
    assert float(zero.abs().max()) == 0.0


def test_sigma_floor_and_nonsaturating_gradient():
    """σ > σ_min strictly, and the σ head keeps a nonzero NLL gradient."""
    pol = make_policy()
    batch = collate_dual_states(torch, real_shard_states())
    out = pol(batch)
    assert float(out["sigma"].min()) > DEFAULT_SIGMA_RANGE[0]
    u = out["mu"] + 0.05
    lp = pol.log_prob_z(out["mu"], out["log_sigma"], u)
    (-lp["per_atom"].sum()).backward()
    grads = [p.grad for n, p in pol.module.named_parameters()
             if "raw_sigma" in n]
    assert all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0


def test_log_prob_is_gaussian_minus_jacobian():
    """per-atom log pi_z == log N(u; μ, σ) − radial log|det J| (exact)."""
    import math

    pol = make_policy()
    batch = collate_dual_states(torch, real_shard_states())
    with torch.no_grad():
        out = pol(batch)
        gen = torch.Generator().manual_seed(7)
        sampled = pol.sample(out["mu"], out["log_sigma"], generator=gen)
        lp = pol.log_prob_z(out["mu"], out["log_sigma"], sampled["u"])
        sigma = out["log_sigma"].exp().unsqueeze(-1)
        resid = (sampled["u"] - out["mu"]) / sigma
        gaussian = (-1.5 * math.log(2.0 * math.pi)
                    - 3.0 * torch.log(sigma).squeeze(-1)
                    - 0.5 * (resid * resid).sum(-1))
    err = float((lp["per_atom"] - (gaussian - lp["log_det"])).abs().max())
    assert err <= TOL
    assert torch.isfinite(lp["per_atom"]).all()


def test_default_mu_is_positive_gated_on_v():
    """Document the probed structure: default μ = softplus(w(h)) ⊙ v with
    softplus > 0, so μ is a POSITIVE scalar multiple of v per atom; the
    flagged signed variant can flip the direction."""
    from pes2ts_core.pesn.models.dual_ts_policy import DUAL_TS_POLICY_VERSION

    assert DUAL_TS_POLICY_VERSION == "pesn_dual_ts_policy_v1.0"
    pol = make_policy(signed=False)
    batch = collate_dual_states(torch, real_shard_states())
    with torch.no_grad():
        out = pol(batch)
    mu, v = out["mu"], out["v"]
    vn = v.norm(dim=-1)
    live = vn > 1e-8
    cos = (mu[live] * v[live]).sum(-1) / (mu[live].norm(dim=-1) * vn[live])
    assert float((cos - 1.0).abs().max()) <= TOL
    # signed gate: force a negative pre-activation -> μ anti-parallel to v
    pol_s = make_policy(signed=True)
    with torch.no_grad():
        pol_s.module.mu_gate.weight.fill_(0.0)
        pol_s.module.mu_gate.bias.fill_(-3.0)
        out_s = pol_s(batch)
    mu_s, v_s = out_s["mu"], out_s["v"]
    vs_n = v_s.norm(dim=-1)
    live_s = vs_n > 1e-8
    cos_s = (mu_s[live_s] * v_s[live_s]).sum(-1) / (
        mu_s[live_s].norm(dim=-1) * vs_n[live_s])
    assert float((cos_s + 1.0).abs().max()) <= TOL


def test_real_shard_batch_forward_finite_and_bound():
    """Real shard states: full forward + sample + log-prob all finite."""
    pol = make_policy()
    batch = collate_dual_states(torch, real_shard_states(4))
    with torch.no_grad():
        out = pol(batch)
        gen = torch.Generator().manual_seed(1)
        sampled = pol.sample(out["mu"], out["log_sigma"], generator=gen)
        lp = pol.log_prob_z(out["mu"], out["log_sigma"], sampled["u"])
    for t in (out["mu"], out["sigma"], out["h"], out["v"],
              sampled["u"], sampled["z"], sampled["a"], lp["per_atom"]):
        assert torch.isfinite(t).all()
    assert int(batch["n_graphs"]) == 4
    assert int(batch["node_counts"].sum()) == int(out["mu"].shape[0])
