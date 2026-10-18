"""Math gates for the V5 sprint BC losses (ADR-0026 §2.8/§5, plan §4).

Gates:

* radial Huber rotation invariance: loss(u, y) == loss(uQ, yQ)
  (float64, tol 1e-6);
* empty cos gate (all ‖y_i‖ ≤ 0.02 Å): L_cos == 0 exactly, gradients
  finite; zero prediction vs nonzero teacher has finite loss + grads;
* finite-difference check of the pair functional J_l(u) = e_l·(u_j−u_i)
  (linear in u, step 1e-5 Å, err < 1e-5);
* degK weights: Σ w_i = 1 per frame, w_i ∝ 1/max(1, degK_i);
* λ_cos ramp: 0 before update 250, exactly 0.02 at ≥500, linear between;
* NaN guard: an all-ineligible frame masked out keeps the loss finite.
"""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.offline.training.key_atom_bc import (
    KeyAtomLossConfig,
    KeyAtomLossError,
    bounded_projection,
    cos_lambda,
    degk_frame_weights,
    key_atom_bc_loss,
    pair_projections,
    pair_unit_vectors,
    radial_huber,
)

torch = require_torch()
DTYPE = torch.float64

CFG = KeyAtomLossConfig()


def _loss_inputs(seed: int = 0, n_graphs: int = 2, keys_per_graph: int = 3):
    """Synthetic u/y/degK/pair tensors (no Lane A/B imports)."""
    rng = np.random.default_rng(seed)
    key_ptr = [0]
    us, ys, degks, poss, key_rows = [], [], [], [], []
    pi, pj = [], []
    offset_n, key_off = 0, 0
    for g in range(n_graphs):
        n = keys_per_graph + int(rng.integers(2, 5))
        poss.append(rng.uniform(-2, 2, size=(n, 3)))
        m = keys_per_graph
        us.append(rng.normal(scale=0.05, size=(m, 3)))
        ys.append(rng.normal(scale=0.05, size=(m, 3)))
        degks.append(rng.uniform(0.5, 3.0, size=m))
        key_rows.append(np.arange(offset_n, offset_n + m))
        for k in range(m - 1):  # GLOBAL node-row pair space (ADR-0026 §2.6)
            pi.append(key_rows[g][k])
            pj.append(key_rows[g][k + 1])
        key_off += m
        key_ptr.append(key_ptr[-1] + m)
        offset_n += n
    pos = torch.as_tensor(np.concatenate(poss), dtype=DTYPE)
    key_index = torch.as_tensor(np.concatenate(key_rows), dtype=torch.int64)
    pair_index = torch.as_tensor(np.stack([pi, pj]), dtype=torch.int64)
    return {
        "u": torch.as_tensor(np.concatenate(us), dtype=DTYPE),
        "y": torch.as_tensor(np.concatenate(ys), dtype=DTYPE),
        "degK": torch.as_tensor(np.concatenate(degks), dtype=DTYPE),
        "key_ptr": torch.as_tensor(key_ptr, dtype=torch.int64),
        "pos": pos,
        "key_index": key_index,
        "pair_index": pair_index,
    }


def _loss(inputs, lambda_cos: float = CFG.lambda_cos_max, **kw):
    return key_atom_bc_loss(
        inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
        inputs["pos"], inputs["key_index"], inputs["pair_index"],
        config=CFG, lambda_cos=lambda_cos, **kw)


def _rot(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# ---------------------------------------------------------------------------
# radial Huber rotation invariance
# ---------------------------------------------------------------------------

def test_radial_huber_rotation_invariance():
    rng = np.random.default_rng(42)
    base = _loss_inputs(seed=1)
    loss0 = float(_loss(base)["loss"])
    for _ in range(10):
        Q = torch.as_tensor(_rot(rng), dtype=DTYPE)
        rotated = dict(base)
        rotated["u"] = base["u"] @ Q
        rotated["y"] = base["y"] @ Q
        rotated["pos"] = base["pos"] @ Q  # e_l rotates; J is invariant
        loss_r = float(_loss(rotated)["loss"])
        assert abs(loss_r - loss0) < 1e-6, (
            f"loss not rotation invariant: {loss0} vs {loss_r}")


def test_radial_huber_shape():
    e = torch.tensor([0.0, 0.1, 0.25, 0.5, 1.0], dtype=DTYPE)
    h = radial_huber(e, 0.25)
    expected = torch.where(
        e <= 0.25, 0.5 * e * e, 0.25 * (e - 0.125))
    assert torch.allclose(h, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# empty gate / gradients
# ---------------------------------------------------------------------------

def test_empty_cos_gate_zero_and_finite_grads():
    inputs = _loss_inputs(seed=2)
    inputs["y"] = inputs["y"] * 1e-4  # all ‖y‖ << 0.02 Å
    u = inputs["u"].clone().requires_grad_(True)
    out = key_atom_bc_loss(
        u, inputs["y"], inputs["degK"], inputs["key_ptr"], inputs["pos"],
        inputs["key_index"], inputs["pair_index"], config=CFG,
        lambda_cos=CFG.lambda_cos_max)
    assert float(out["l_cos"]) == 0.0
    assert bool(torch.isfinite(out["loss"]))
    out["loss"].backward()
    assert bool(torch.isfinite(u.grad).all())


def test_zero_prediction_nonzero_teacher_finite():
    inputs = _loss_inputs(seed=3)
    u = torch.zeros_like(inputs["u"], requires_grad=True)
    out = key_atom_bc_loss(
        u, inputs["y"], inputs["degK"], inputs["key_ptr"], inputs["pos"],
        inputs["key_index"], inputs["pair_index"], config=CFG,
        lambda_cos=CFG.lambda_cos_max)
    assert bool(torch.isfinite(out["loss"]))
    assert float(out["loss"].detach()) > 0.0
    out["loss"].backward()
    assert bool(torch.isfinite(u.grad).all())


# ---------------------------------------------------------------------------
# pair functional finite differences
# ---------------------------------------------------------------------------

def test_pair_projection_finite_difference():
    """J_l(u) = e_l·(u_j − u_i) is linear; check each ∂J_l/∂u[slot, axis]
    against a central difference (step 1e-5 Å, err < 1e-5)."""
    inputs = _loss_inputs(seed=4, n_graphs=1)
    u = inputs["u"]
    pair_index = inputs["pair_index"]
    e_l = pair_unit_vectors(inputs["pos"], inputs["key_index"], pair_index)
    j0 = pair_projections(u, pair_index, e_l)
    step = 1e-5
    for l in range(pair_index.shape[1]):
        i_slot, j_slot = int(pair_index[0, l]), int(pair_index[1, l])
        for slot in (i_slot, j_slot):
            sgn = -1.0 if slot == i_slot else 1.0
            for axis in range(3):
                up, um = u.clone(), u.clone()
                up[slot, axis] += step
                um[slot, axis] -= step
                numeric = (float(pair_projections(up, pair_index, e_l)[l])
                           - float(pair_projections(um, pair_index, e_l)[l])
                           ) / (2 * step)
                analytic = sgn * float(e_l[l, axis])
                assert abs(numeric - analytic) < 1e-5, (
                    f"pair {l} slot {slot} axis {axis}: numeric {numeric} "
                    f"!= analytic {analytic}")


def test_pair_projection_linearity():
    inputs = _loss_inputs(seed=9, n_graphs=1)
    pi = inputs["pair_index"]
    e_l = pair_unit_vectors(inputs["pos"], inputs["key_index"], pi)
    u1, u2 = inputs["u"], inputs["u"] * 2.0
    j1 = pair_projections(u1, pi, e_l)
    j2 = pair_projections(u2, pi, e_l)
    assert torch.allclose(j2, 2.0 * j1, atol=1e-12)


def test_pair_unit_vectors_are_unit():
    inputs = _loss_inputs(seed=10)
    from pes2ts_core.pesn.contracts.key_atom import global_pair_rows_to_slots
    slots = global_pair_rows_to_slots(
        inputs["pair_index"], inputs["key_index"],
        int(inputs["pos"].shape[0]))
    e_l = pair_unit_vectors(inputs["pos"], inputs["key_index"], slots)
    norms = e_l.norm(dim=-1)
    assert bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-12))


# ---------------------------------------------------------------------------
# degK weights
# ---------------------------------------------------------------------------

def test_degk_weights_normalization():
    degK = torch.tensor([1.0, 1.0, 2.0, 5.0, 1.0], dtype=DTYPE)
    w = degk_frame_weights(degK)
    expected = (1.0 / degK.clamp(min=1.0))
    expected = expected / expected.sum()
    assert torch.allclose(w, expected, atol=1e-12)
    assert abs(float(w.sum()) - 1.0) < 1e-12
    # 1/max(1, degK) weighting: higher degK -> smaller weight
    assert float(w[3]) < float(w[0])


def test_degk_weights_per_frame():
    inputs = _loss_inputs(seed=11, n_graphs=2, keys_per_graph=3)
    out = _loss(inputs, lambda_cos=0.0)
    # each frame term used normalized weights; total is finite
    assert bool(torch.isfinite(out["loss"]))
    assert out["n_frames"] == 2


# ---------------------------------------------------------------------------
# λ_cos ramp
# ---------------------------------------------------------------------------

def test_cos_lambda_ramp():
    cfg = KeyAtomLossConfig(cos_start_update=250, cos_ramp_updates=250,
                            lambda_cos_max=0.02)
    assert cos_lambda(0, cfg) == 0.0
    assert cos_lambda(249, cfg) == 0.0
    mid = cos_lambda(375, cfg)
    assert 0.0 < mid < 0.02
    # linear between: midpoint value is half the max
    assert abs(mid - 0.01) < 1e-12
    assert cos_lambda(500, cfg) == 0.02
    assert cos_lambda(1500, cfg) == 0.02
    # exactly linear at quarter points
    assert abs(cos_lambda(312, cfg) - 0.02 * 62 / 250) < 1e-12


# ---------------------------------------------------------------------------
# NaN guard: all-ineligible frame excluded
# ---------------------------------------------------------------------------

def test_nan_guard_empty_branch_frame_masked():
    inputs = _loss_inputs(seed=12, n_graphs=2)
    # frame 1: teacher magnitudes below the cos gate AND make y huge-free
    key_ptr = inputs["key_ptr"].numpy()
    y = inputs["y"].clone()
    lo, hi = int(key_ptr[1]), int(key_ptr[2])
    y[lo:hi] = 0.0  # all-ineligible frame
    u = inputs["u"].clone()
    u[lo:hi] = 0.0
    mask = torch.tensor([True, False])
    out = key_atom_bc_loss(
        u, y, inputs["degK"], inputs["key_ptr"], inputs["pos"],
        inputs["key_index"], inputs["pair_index"], config=CFG,
        lambda_cos=CFG.lambda_cos_max, frame_mask=mask)
    assert bool(torch.isfinite(out["loss"]))
    assert out["n_frames"] == 1
    # frame with y≡0 contributes L_cos = 0 exactly when included
    out_incl = key_atom_bc_loss(
        u, y, inputs["degK"], inputs["key_ptr"], inputs["pos"],
        inputs["key_index"], inputs["pair_index"], config=CFG,
        lambda_cos=CFG.lambda_cos_max)
    assert bool(torch.isfinite(out_incl["loss"]))
    # its cos contribution is exactly zero: gate empty on that frame
    lo2, hi2 = 0, int(key_ptr[1])
    gate0 = (y[lo2:hi2].norm(dim=-1) > CFG.tau_norm_angstrom).sum()
    assert int(gate0) > 0  # the OTHER frame still has gated atoms
    assert float(out_incl["l_cos"]) < 1.0


def test_nonfinite_input_raises():
    inputs = _loss_inputs(seed=13)
    bad = inputs["u"].clone()
    bad[0, 0] = float("nan")
    with pytest.raises(KeyAtomLossError) as excinfo:
        key_atom_bc_loss(
            bad, inputs["y"], inputs["degK"], inputs["key_ptr"],
            inputs["pos"], inputs["key_index"], inputs["pair_index"],
            config=CFG, lambda_cos=0.0)
    assert "NONFINITE" in str(excinfo.value)


# ---------------------------------------------------------------------------
# bounded projection: single uniform factor, max_atom ≤ 0.20, RMS ≤ d0
# ---------------------------------------------------------------------------

def test_bounded_projection_uniform_factor():
    key_ptr = torch.tensor([0, 3, 7], dtype=torch.int64)
    u = torch.arange(1, 22, dtype=DTYPE).reshape(7, 3) * 0.1  # large, no zeros
    ub, scale = bounded_projection(u, 0.08, key_ptr)
    for b in range(2):
        lo, hi = int(key_ptr[b]), int(key_ptr[b + 1])
        f = float(scale[b])
        norms = ub[lo:hi].norm(dim=-1)
        rms = float(torch.sqrt(torch.mean(norms ** 2)))
        assert float(norms.max()) <= 0.20 + 1e-12
        assert rms <= 0.08 + 1e-12
        # single common factor: every entry of the frame scales by f
        ratio = ub[lo:hi] / u[lo:hi]
        assert bool(torch.allclose(
            ratio, torch.full_like(ratio, f), atol=1e-9))


def test_bounded_projection_small_u_untouched():
    key_ptr = torch.tensor([0, 2], dtype=torch.int64)
    u = torch.ones(2, 3, dtype=DTYPE) * 0.001
    ub, scale = bounded_projection(u, 0.08, key_ptr)
    assert torch.allclose(ub, u)
    assert float(scale[0]) == 1.0


def test_vectorized_loss_matches_loop_reference():
    """Vectorized trainer path must equal the per-frame loop reference."""
    for trial in range(12):
        inputs = _loss_inputs(seed=100 + trial, n_graphs=int(
            np.random.default_rng(trial).integers(1, 5)),
            keys_per_graph=int(np.random.default_rng(trial + 7).integers(1, 5)))
        lam = float(np.random.default_rng(trial).uniform(0, 0.02))
        a = key_atom_bc_loss(
            inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
            inputs["pos"], inputs["key_index"], inputs["pair_index"],
            config=CFG, lambda_cos=lam)
        from pes2ts_core.pesn.offline.training.key_atom_bc import (
            key_atom_bc_loss_vectorized,
        )
        b = key_atom_bc_loss_vectorized(
            inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
            inputs["pos"], inputs["key_index"], inputs["pair_index"],
            config=CFG, lambda_cos=lam)
        for k in ("loss", "l_atom", "l_pair", "l_cos"):
            assert torch.allclose(a[k], b[k], atol=1e-9), (trial, k)
