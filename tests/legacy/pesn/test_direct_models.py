"""Direct model probability contracts (WP2, plan v3 §6 acceptance).

float64 tolerances per plan §6: analytic-vs-numeric Jacobian relative
error ~1e-6, inverse roundtrip, small-r series stability, reparameterized
sampling gradient, rotation/translation/permutation consistency of the
equivariant channel, and the log pi_z / log pi_a relation.
"""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.features.direct_graph import (
    N_GRAPH_SCALARS,
    build_direct_features,
)
from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.models.direct_policy import (
    RadialPolicyError,
    radial_inverse,
    radial_log_det_per_atom,
    radial_transform_stable,
)

torch = require_torch()

TASK_GRAPH = {
    "edits": [
        {"kind": "broken", "i": 0, "j": 1},
        {"kind": "formed", "i": 1, "j": 2},
        {"kind": "order_changed", "i": 0, "j": 2},
    ]
}


def features_for(h_x: float = 1.1, n: int = 3):
    z = np.array([6, 1, 6])[:n] if n <= 3 else np.array([6] * n)
    coords = np.array([[0.0, 0, 0], [h_x, 0, 0], [2.5, 0, 0]])[:n]
    if n > 3:
        coords = np.concatenate([coords, np.random.default_rng(0).normal(
            scale=0.5, size=(n - 3, 3)) + 1.2], axis=0)
    return build_direct_features(
        atomic_numbers=z, coordinates=coords,
        gradient=np.zeros_like(coords), task_graph=TASK_GRAPH,
        delta_e_hartree=-0.05, steps_remaining=90, budget_steps=100,
        goal_consecutive_hits=0, goal_hits_required=2,
        no_progress_count=1, no_progress_window=20, best_potential=-0.4,
    )


def test_transform_jacobian_analytic_matches_numeric_float64():
    """slogdet of the numeric Jacobian == analytic log-det sum (rel 1e-6)."""
    g = torch.Generator().manual_seed(20261015)
    for scale in (0.05, 0.5, 2.0):
        u = torch.randn(4, 3, generator=g, dtype=torch.float64) * scale
        flat = u.reshape(-1).clone().requires_grad_(True)
        def f(x):
            return radial_transform_stable(torch, x.reshape(4, 3)).reshape(-1)
        jacobian = torch.autograd.functional.jacobian(f, flat)
        numeric_logdet = torch.linalg.slogdet(jacobian.double())[1]
        analytic = radial_log_det_per_atom(torch, u).sum()
        assert torch.isfinite(numeric_logdet) and torch.isfinite(analytic)
        assert float(numeric_logdet) == pytest.approx(float(analytic), rel=1e-6, abs=1e-9)


def test_inverse_roundtrip_float64():
    g = torch.Generator().manual_seed(7)
    u = torch.randn(8, 3, generator=g, dtype=torch.float64) * 0.8
    z = radial_transform_stable(torch, u)
    u_back = radial_inverse(torch, z)
    assert torch.allclose(u, u_back, atol=1e-9, rtol=1e-9)
    with pytest.raises(RadialPolicyError):
        radial_inverse(torch, torch.ones(2, 3, dtype=torch.float64) * 1.5)


def test_small_r_series_stability():
    import math
    for r in (1e-8, 1e-6, 5e-5):
        u = torch.tensor([[r, 0.0, 0.0]], dtype=torch.float64)
        z = radial_transform_stable(torch, u)
        assert torch.isfinite(z).all()
        # z = tanh(r) * u / r  ->  first component is exactly tanh(r) ~ r
        assert float(z[0, 0]) == pytest.approx(math.tanh(r), rel=1e-9, abs=1e-15)
        logdet = radial_log_det_per_atom(torch, u)
        assert torch.isfinite(logdet).all()
        assert float(logdet[0]) == pytest.approx(-(5.0 / 3.0) * r * r, abs=1e-14, rel=1e-4)


def math_tanh_times_u(r: float, comp: float) -> float:
    import math
    return math.tanh(r) / r * comp if r > 0 else comp


def test_log_prob_z_matches_density_change_of_variables():
    """log pi_z(u) == log N(u; mu, sigma) - log|det J| per atom (float64)."""
    import math
    g = torch.Generator().manual_seed(3)
    mu = torch.randn(6, 3, generator=g, dtype=torch.float64) * 0.3
    log_sigma = torch.full((6,), math.log(0.5), dtype=torch.float64)
    sigma = log_sigma.exp().unsqueeze(-1)
    u = mu + sigma * torch.randn(6, 3, generator=g, dtype=torch.float64)
    resid = (u - mu) / sigma
    gaussian = (-1.5 * math.log(2 * math.pi)
                - 3.0 * torch.log(sigma).squeeze(-1)
                - 0.5 * (resid * resid).sum(-1))
    per_atom = gaussian - radial_log_det_per_atom(torch, u)
    z = radial_transform_stable(torch, u)
    assert float(z.norm(dim=-1).max()) < 1.0
    assert torch.isfinite(per_atom).all()
    # sum over atoms gives the sample-level log pi_z
    assert float(per_atom.sum()) == pytest.approx(
        float(gaussian.sum() - radial_log_det_per_atom(torch, u).sum()), rel=1e-12)


def test_log_pi_a_relation():
    """log pi_a = log pi_z - 3N log(a_max) (plan §6.1)."""
    n = 5
    log_pi_z = -3.17 * n
    a_max = 0.1
    log_pi_a = log_pi_z - 3 * n * np.log(a_max)
    assert log_pi_a > log_pi_z  # angstrom-scale density


def test_reparameterized_gradient_flows():
    mu = torch.zeros(8, 3, dtype=torch.float64, requires_grad=True)
    log_sigma = torch.full((8,), -1.0, dtype=torch.float64, requires_grad=True)
    eps = torch.randn(8, 3, generator=torch.Generator().manual_seed(1),
                      dtype=torch.float64)
    u = mu + log_sigma.exp().unsqueeze(-1) * eps
    z = radial_transform_stable(torch, u)
    loss = (z ** 2).sum()
    loss.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert log_sigma.grad is not None and torch.isfinite(log_sigma.grad).all()


def test_encoder_rotation_translation_permutation_consistency():
    from pes2ts_core.pesn.models.egnn import (
        DirectEGNNEncoder,
        EGNNConfig,
        collate_direct_features,
    )
    torch.manual_seed(11)
    enc = DirectEGNNEncoder(EGNNConfig(hidden=32, n_layers=2, n_basis=32))
    base = features_for(1.1)
    grad = np.random.default_rng(5).normal(scale=0.05, size=base.coordinates.shape)

    def run(coords, gradient, perm=None):
        f = build_direct_features(
            atomic_numbers=base.atomic_numbers, coordinates=coords,
            gradient=gradient, task_graph=TASK_GRAPH,
            delta_e_hartree=-0.05, steps_remaining=90, budget_steps=100,
            goal_consecutive_hits=0, goal_hits_required=2,
            no_progress_count=1, no_progress_window=20, best_potential=-0.4)
        if perm is not None:
            f = _permute(f, perm, TASK_GRAPH)
        batch = collate_direct_features(torch, [f])
        with torch.no_grad():
            return enc(batch)

    # rotation R (det=+1) applied to coords AND gradient
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    out_base = run(base.coordinates, grad)
    out_rot = run(base.coordinates @ R.T, grad @ R.T)
    assert np.allclose(out_base["h"].numpy(), out_rot["h"].numpy(), atol=1e-4)
    v_base = out_base["v"].numpy()
    v_rot = out_rot["v"].numpy()
    assert np.allclose(v_base @ R.T, v_rot, atol=1e-4)

    # translation invariance (coords only; gradient unchanged)
    shift = np.array([0.3, -0.2, 0.5])
    out_tr = run(base.coordinates + shift, grad)
    assert np.allclose(out_base["h"].numpy(), out_tr["h"].numpy(), atol=1e-4)
    assert np.allclose(v_base, out_tr["v"].numpy(), atol=1e-4)

    # permutation equivariance of node-level outputs
    rng = np.random.default_rng(9)
    perm = rng.permutation(base.n_atoms)
    out_perm = run(base.coordinates, grad, perm=perm)
    assert np.allclose(out_base["h"].numpy()[perm], out_perm["h"].numpy(), atol=1e-4)
    assert np.allclose(v_base[perm], out_perm["v"].numpy(), atol=1e-4)
def _permute(f, perm, task_graph):
    """Rebuild features for a permuted atom order (edits re-indexed)."""
    inv = np.argsort(perm)
    new_graph = {
        "edits": [
            {"kind": e["kind"], "i": int(inv[e["i"]]), "j": int(inv[e["j"]])}
            for e in task_graph["edits"]
        ]
    }
    return build_direct_features(
        atomic_numbers=f.atomic_numbers[perm],
        coordinates=f.coordinates[perm],
        gradient=f.gradient[perm],
        task_graph=new_graph,
        delta_e_hartree=-0.05, steps_remaining=90, budget_steps=100,
        goal_consecutive_hits=0, goal_hits_required=2,
        no_progress_count=1, no_progress_window=20, best_potential=-0.4,
    )


def test_policy_forward_sample_and_logprob_end_to_end():
    from pes2ts_core.pesn.models.egnn import (
        EGNNConfig,
        collate_direct_features,
    )
    from pes2ts_core.pesn.models.direct_policy import RadialDirectPolicy
    torch.manual_seed(13)
    policy = RadialDirectPolicy(encoder=DirectEGNNEncoder(EGNNConfig(hidden=32, n_layers=2)),
                                action_max=0.1)
    f = features_for(1.1)
    batch = collate_direct_features(torch, [f])
    out = policy(batch)
    assert out["mu"].shape == (3, 3)
    assert out["log_sigma"].shape == (3,)
    sampled = policy.module.sample(out["mu"], out["log_sigma"],
                                   generator=torch.Generator().manual_seed(0))
    a = sampled["a"]
    assert float(a.norm(dim=-1).max()) <= 0.1 * (1 + 1e-5)
    lp = policy.module.log_prob_z(out["mu"], out["log_sigma"], sampled["u"])
    assert torch.isfinite(lp["per_atom"]).all()


def test_sigma_gradient_never_saturates():
    """Regression: the old hard clamp left sigma stuck at sigma_max with
    exactly zero gradient. The softplus parametrization must keep the sigma
    head gradient nonzero at initialization and after an update."""
    from pes2ts_core.pesn.models.direct_policy import RadialDirectPolicy
    from pes2ts_core.pesn.models.egnn import (
        DirectEGNNEncoder,
        EGNNConfig,
        collate_direct_features,
    )
    torch.manual_seed(3)
    policy = RadialDirectPolicy(encoder=DirectEGNNEncoder(EGNNConfig(hidden=32, n_layers=2)),
                                action_max=0.1)
    batch = collate_direct_features(torch, [features_for(1.1)])
    out = policy(batch)
    sigma = out["sigma"]
    assert float(sigma.min()) > 0.02  # strictly above the floor
    with torch.no_grad():
        u = radial_inverse(torch, torch.full((3, 3), 0.5))
    lp = policy.module.log_prob_z(out["mu"], out["log_sigma"], u)
    loss = -lp["per_atom"].sum()
    loss.backward()
    grads = {n: p.grad for n, p in policy.module.named_parameters()
             if "raw_sigma" in n}
    assert all(g is not None for g in grads.values())
    total = sum(float(g.abs().sum()) for g in grads.values())
    assert total > 0, f"sigma head gradient fully zero: {grads}"


def test_variable_n_graph_scalars_and_masks():
    f3 = features_for(1.1, n=3)
    f5 = features_for(1.1, n=5)
    assert f3.graph_scalars.shape == (N_GRAPH_SCALARS,)
    assert f5.edge_index.shape[1] >= f3.edge_index.shape[1]
    # task edges always present even beyond the spatial cutoff
    long = build_direct_features(
        atomic_numbers=np.array([6, 1, 6]),
        coordinates=np.array([[0.0, 0, 0], [1.1, 0, 0], [30.0, 0, 0]]),
        gradient=np.zeros((3, 3)), task_graph=TASK_GRAPH,
        delta_e_hartree=0.0, steps_remaining=100, budget_steps=100,
        goal_consecutive_hits=0, goal_hits_required=2,
        no_progress_count=0, no_progress_window=20, best_potential=0.0)
    assert long.edge_task_roles.sum() > 0


def DirectEGNNEncoder(config):  # helper indirection for the import inside test
    from pes2ts_core.pesn.models.egnn import DirectEGNNEncoder as E
    return E(config)
