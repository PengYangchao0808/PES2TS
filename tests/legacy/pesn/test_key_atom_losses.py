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

import math

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


# ---------------------------------------------------------------------------
# V5.1a exact-Δd trio + magnitude supervision (plan §3.3) — G5.1 gates
# ---------------------------------------------------------------------------

from pes2ts_core.pesn.contracts.key_atom import global_pair_rows_to_slots
from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (  # noqa: E402
    exact_delta_d as exact_delta_d_numpy,
)
from pes2ts_core.pesn.offline.training.key_atom_bc import (  # noqa: E402
    V51_INNER_SPLIT_SEED,
    aux_ramp_fraction,
    exact_delta_d_torch,
    key_atom_bc_loss_vectorized,
)

CFG_EXACT = KeyAtomLossConfig(pair_loss="exact_trio")


def _v51_inputs(seed: int = 0, n_graphs: int = 2, keys_per_graph: int = 3,
                scale_y: float = 0.05):
    """Synthetic fixture with GLOBAL-row pairs + pair_edit (±1/0)."""
    inputs = _loss_inputs(seed=seed, n_graphs=n_graphs,
                          keys_per_graph=keys_per_graph)
    rng = np.random.default_rng(seed + 991)
    n_pairs = inputs["pair_index"].shape[1]
    edits = rng.choice([1.0, -1.0], size=n_pairs)
    inputs["pair_edit"] = torch.as_tensor(edits, dtype=DTYPE)
    inputs["y"] = torch.as_tensor(
        rng.normal(scale=scale_y, size=tuple(inputs["y"].shape)), dtype=DTYPE)
    inputs["s"] = inputs["u"].norm(dim=-1).detach() + 0.02
    return inputs


def _v51_kwargs(inputs, **over):
    kw = dict(pair_edit=inputs["pair_edit"], s=inputs["s"],
              lambda_mag=0.5, lambda_sign=0.2, lambda_dead=0.1,
              lambda_cos=0.02)
    kw.update(over)
    return kw


def test_exact_delta_d_torch_matches_numpy():
    """torch exact Δd == the G5.3 numpy protocol on random fixtures."""
    for trial in range(6):
        inputs = _v51_inputs(seed=200 + trial, n_graphs=1 + trial % 3,
                             keys_per_graph=3)
        pair_index = inputs["pair_index"]
        key_index = inputs["key_index"]
        slots = global_pair_rows_to_slots(
            pair_index, key_index, int(inputs["pos"].shape[0]))
        dd_t = exact_delta_d_torch(
            inputs["pos"][key_index], slots, inputs["u"])
        key_rows = [int(r) for r in key_index]
        pairs = [(key_rows[int(slots[0, l])], key_rows[int(slots[1, l])])
                 for l in range(slots.shape[1])]
        dd_n = exact_delta_d_numpy(
            inputs["pos"].numpy(), key_rows, pairs, inputs["u"].numpy())
        assert np.allclose(dd_t.numpy(), dd_n, atol=1e-12), trial


def test_exact_trio_vectorized_equals_loop():
    """Vectorized ≡ loop for the FULL v51 loss (plan §3.3 terms)."""
    for trial in range(10):
        rng = np.random.default_rng(trial)
        inputs = _v51_inputs(seed=300 + trial,
                             n_graphs=int(rng.integers(1, 4)),
                             keys_per_graph=int(rng.integers(1, 5)),
                             scale_y=float(rng.uniform(0.02, 0.15)))
        kw = _v51_kwargs(
            inputs, lambda_mag=float(rng.uniform(0, 0.5)),
            lambda_sign=float(rng.uniform(0, 0.2)),
            lambda_dead=float(rng.uniform(0, 0.1)))
        a = key_atom_bc_loss(
            inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
            inputs["pos"], inputs["key_index"], inputs["pair_index"],
            config=CFG_EXACT, **kw)
        b = key_atom_bc_loss_vectorized(
            inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
            inputs["pos"], inputs["key_index"], inputs["pair_index"],
            config=CFG_EXACT, **kw)
        for k in ("loss", "l_atom", "l_pair", "l_pair_reg", "l_cos",
                  "l_mag", "l_sign", "l_dead"):
            assert torch.allclose(a[k], b[k], atol=1e-9), (trial, k)
        assert a["n_pairs_active"] == b["n_pairs_active"]
        assert a["pair_loss"] == "exact_trio"


def test_exact_trio_pair_edit_filtering():
    """Zero-edit pairs are EXCLUDED from the exact_trio pair terms but
    the legacy_j term still sees every pair (unchanged V5 semantics)."""
    inputs = _v51_inputs(seed=5, n_graphs=1, keys_per_graph=3)
    zero_edit = torch.zeros_like(inputs["pair_edit"])
    kw_all = _v51_kwargs(inputs)
    kw_none = _v51_kwargs(inputs, pair_edit=zero_edit)
    out_all = key_atom_bc_loss(
        inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
        inputs["pos"], inputs["key_index"], inputs["pair_index"],
        config=CFG_EXACT, **kw_all)
    out_none = key_atom_bc_loss(
        inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
        inputs["pos"], inputs["key_index"], inputs["pair_index"],
        config=CFG_EXACT, **kw_none)
    assert out_all["n_pairs_active"] > 0
    assert out_none["n_pairs_active"] == 0
    assert float(out_none["l_sign"]) == 0.0
    assert float(out_none["l_dead"]) == 0.0
    # legacy_j ignores pair_edit entirely (V5 behavior preserved)
    out_legacy = key_atom_bc_loss(
        inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
        inputs["pos"], inputs["key_index"], inputs["pair_index"],
        config=CFG, pair_edit=zero_edit)
    out_legacy_all = key_atom_bc_loss(
        inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
        inputs["pos"], inputs["key_index"], inputs["pair_index"],
        config=CFG, pair_edit=inputs["pair_edit"])
    assert torch.allclose(out_legacy["loss"], out_legacy_all["loss"],
                          atol=1e-12)


def test_exact_trio_zero_case_empty_active_pairs():
    """Teacher below the 0.005 Å active gate → L_sign = L_dead = 0 EXACTLY
    with finite gradients (never NaN)."""
    inputs = _v51_inputs(seed=7, n_graphs=2, keys_per_graph=3)
    tiny = inputs["y"] * 1e-3  # |Δd_true| << 0.005
    u = inputs["u"].clone().requires_grad_(True)
    out = key_atom_bc_loss(
        u, tiny, inputs["degK"], inputs["key_ptr"], inputs["pos"],
        inputs["key_index"], inputs["pair_index"], config=CFG_EXACT,
        **_v51_kwargs(inputs, s=None))
    assert out["n_pairs_active"] == 0
    assert float(out["l_sign"].detach()) == 0.0
    assert float(out["l_dead"].detach()) == 0.0
    assert float(out["l_mag"].detach()) == 0.0  # s=None → head-off
    assert bool(torch.isfinite(out["loss"]))
    out["loss"].backward()
    assert bool(torch.isfinite(u.grad).all())
    # vectorized twin behaves identically
    out_v = key_atom_bc_loss_vectorized(
        inputs["u"].detach(), tiny, inputs["degK"], inputs["key_ptr"],
        inputs["pos"], inputs["key_index"], inputs["pair_index"],
        config=CFG_EXACT, **_v51_kwargs(inputs, s=None))
    assert float(out_v["l_sign"]) == 0.0
    assert float(out_v["l_dead"]) == 0.0


def test_sign_term_monotonicity():
    """L_sign decreases when the predicted Δd moves in the teacher's
    sign and grows with |Δd_pred| along that sign (plan §3.3)."""
    tau = CFG_EXACT.pair_sign_tau_angstrom
    dd_true = torch.tensor([0.02, -0.02], dtype=DTYPE)
    dd_wrong = torch.tensor([-0.02, 0.02], dtype=DTYPE)
    dd_right_small = torch.tensor([0.006, -0.006], dtype=DTYPE)
    dd_right_big = torch.tensor([0.03, -0.03], dtype=DTYPE)

    def sign_term(dd_pred):
        return float(torch.nn.functional.softplus(
            -torch.sign(dd_true) * dd_pred / tau).mean())
    assert sign_term(dd_right_big) < sign_term(dd_right_small)
    assert sign_term(dd_right_small) < sign_term(dd_wrong)


def test_dead_term_monotonicity():
    """L_dead penalizes |Δd_pred| inside the dead zone most (plan §3.3)."""
    thr = CFG_EXACT.pair_active_threshold_angstrom
    tau = CFG_EXACT.pair_dead_tau_angstrom

    def dead_term(abs_dd):
        return float(torch.nn.functional.softplus(
            (thr - abs_dd) / tau).mean())
    assert dead_term(torch.tensor([0.0], dtype=DTYPE)) > \
        dead_term(torch.tensor([0.004], dtype=DTYPE))
    assert dead_term(torch.tensor([0.004], dtype=DTYPE)) > \
        dead_term(torch.tensor([0.02], dtype=DTYPE))
    assert dead_term(torch.tensor([0.05], dtype=DTYPE)) < 0.01


def test_l_mag_log_form_sanity():
    """L_mag = radial_huber(log((s_eff+1e-6)/(‖y‖+1e-6))) mean over gated
    atoms (s_eff = EFFECTIVE magnitude, post deg_scale — the quantity the
    model stashes in last_magnitudes); quadratic branch for small
    log-ratio, exact huber formula."""
    gate = CFG_EXACT.mag_supervision_tau_angstrom
    y = torch.stack([torch.tensor([0.05, 0.0, 0.0], dtype=DTYPE),
                     torch.tensor([0.0, 0.08, 0.0], dtype=DTYPE),
                     torch.tensor([0.0, 0.0, 0.03], dtype=DTYPE)])
    y_norm = y.norm(dim=-1)
    key_ptr = torch.tensor([0, 3], dtype=torch.int64)
    degK = torch.ones(3, dtype=DTYPE)
    pos = torch.zeros(4, 3, dtype=DTYPE)
    key_index = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    u = y.clone()

    def expected_mag(s_val):
        logs = torch.log((torch.full_like(y_norm, s_val) + 1e-6)
                         / (y_norm + 1e-6))
        # mirror radial_huber's SIGNED branch (e <= δ → quadratic)
        quad = 0.5 * logs * logs
        linear = 0.25 * (logs - 0.125)
        return float(torch.where(logs <= 0.25, quad, linear).mean())

    for s_val in (0.05, 0.055, 0.02, 0.2):
        s = torch.full((3,), s_val, dtype=DTYPE)
        out = key_atom_bc_loss(
            u, y, degK, key_ptr, pos, key_index, None, config=CFG_EXACT,
            s=s, lambda_mag=1.0)
        assert abs(float(out["l_mag"]) - expected_mag(s_val)) < 1e-6, s_val
    # s == ‖y‖ per atom → L_mag = 0 exactly
    out_match = key_atom_bc_loss(
        u, y, degK, key_ptr, pos, key_index, None, config=CFG_EXACT,
        s=y_norm.clone(), lambda_mag=1.0)
    assert float(out_match["l_mag"]) < 1e-10
    # monotone in the log-ratio: overshooting s costs more than matching
    out_over = key_atom_bc_loss(
        u, y, degK, key_ptr, pos, key_index, None, config=CFG_EXACT,
        s=torch.full((3,), 0.2, dtype=DTYPE), lambda_mag=1.0)
    assert float(out_over["l_mag"]) > float(out_match["l_mag"])
    # teacher below the 0.01 Å gate → L_mag = 0 exactly
    out_small = key_atom_bc_loss(
        u, y * 1e-3, degK, key_ptr, pos, key_index, None,
        config=CFG_EXACT, s=torch.full((3,), 0.05, dtype=DTYPE),
        lambda_mag=1.0)
    assert float(out_small["l_mag"]) == 0.0
    assert float(gate) == 0.01


def test_v51_loss_finite_gradients_every_term():
    """Every new v51 term backprops with finite grads on a small fixture
    (gradcheck on the exact-Δd path, float64)."""
    inputs = _v51_inputs(seed=11, n_graphs=1, keys_per_graph=3)
    u = inputs["u"].clone().requires_grad_(True)
    out = key_atom_bc_loss(
        u, inputs["y"], inputs["degK"], inputs["key_ptr"], inputs["pos"],
        inputs["key_index"], inputs["pair_index"], config=CFG_EXACT,
        **_v51_kwargs(inputs))
    assert out["n_pairs_active"] > 0
    assert float(out["l_sign"].detach()) > 0.0
    assert float(out["l_dead"].detach()) > 0.0
    assert float(out["l_mag"].detach()) > 0.0
    out["loss"].backward()
    assert bool(torch.isfinite(u.grad).all())

    # central finite-difference check of ∂loss/∂u (analytic vs numeric)
    u0 = inputs["u"].detach().clone()
    eps = 1e-6

    def loss_of(uu):
        return float(key_atom_bc_loss(
            uu, inputs["y"], inputs["degK"], inputs["key_ptr"],
            inputs["pos"], inputs["key_index"], inputs["pair_index"],
            config=CFG_EXACT, **_v51_kwargs(inputs))["loss"])
    u_t = u0.clone().requires_grad_(True)
    value = key_atom_bc_loss(
        u_t, inputs["y"], inputs["degK"], inputs["key_ptr"], inputs["pos"],
        inputs["key_index"], inputs["pair_index"], config=CFG_EXACT,
        **_v51_kwargs(inputs))["loss"]
    value.backward()
    for slot in range(u0.shape[0]):
        for axis in range(3):
            up, um = u0.clone(), u0.clone()
            up[slot, axis] += eps
            um[slot, axis] -= eps
            numeric = (loss_of(up) - loss_of(um)) / (2 * eps)
            assert abs(numeric - float(u_t.grad[slot, axis])) < 1e-5, (
                slot, axis)


def test_aux_ramp_schedule():
    """plan §3.3: first 250 updates only L_atom + L_pair_reg; then 250
    updates linear ramp of the auxiliary terms."""
    cfg = KeyAtomLossConfig()
    assert aux_ramp_fraction(0, cfg) == 0.0
    assert aux_ramp_fraction(249, cfg) == 0.0
    assert abs(aux_ramp_fraction(250, cfg) - 0.0) < 1e-12
    assert abs(aux_ramp_fraction(375, cfg) - 0.5) < 1e-12
    assert abs(aux_ramp_fraction(312, cfg) - 62 / 250) < 1e-12
    assert aux_ramp_fraction(500, cfg) == 1.0
    assert aux_ramp_fraction(3000, cfg) == 1.0
    nested = KeyAtomLossConfig.from_campaign_config(
        {"v51": {"ramp": {"start_update": 100, "ramp_updates": 50}}})
    assert nested.aux_start_update == 100
    assert nested.aux_ramp_updates == 50
    assert aux_ramp_fraction(125, nested) == 0.5
    # frozen seed constant (plan §3.4)
    assert V51_INNER_SPLIT_SEED == 20261019


def test_pair_loss_mode_validation():
    with pytest.raises(KeyAtomLossError):
        KeyAtomLossConfig(pair_loss="hinge")
    cfg = KeyAtomLossConfig.from_campaign_config({"pair_loss": "exact_trio"})
    assert cfg.pair_loss == "exact_trio"
    # exact_trio without pair_edit is a typed refusal
    inputs = _v51_inputs(seed=13, n_graphs=1)
    with pytest.raises(KeyAtomLossError) as excinfo:
        key_atom_bc_loss(
            inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
            inputs["pos"], inputs["key_index"], inputs["pair_index"],
            config=CFG_EXACT, s=inputs["s"])
    assert "EXACT_TRIO_NEEDS_PAIR_EDIT" in str(excinfo.value)


# ---------------------------------------------------------------------------
# REGRESSION (2026-10-19 real-run bug): L_mag must supervise the
# EFFECTIVE magnitude s_eff = s_raw·deg_scale, NOT the raw head output.
# Supervising raw s drives s_raw → ‖y‖ and collapses the delivered
# displacement ‖u‖ = s_eff·‖v_hat‖ to ‖y‖·deg_scale (α ≈ 0.04 at
# degK=2) — the B1/B3 ablation crash at the aux-ramp start (u250).
# ---------------------------------------------------------------------------

def _mag_regression_batch(*, n: int = 8, n_keys: int = 3, seed: int = 0):
    """Single-graph KeyAtomBatch with degK≡2 (deg_scale = a_ref/2 = 0.04)
    for the effective-magnitude regression test."""
    from pes2ts_core.pesn.contracts.key_atom import (
        EDGE_DIM,
        EDGE_SLOTS,
        NODE_INV_DIM,
        TASK_DIM,
        KeyAtomBatch,
    )
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-2.5, 2.5, size=(n, 3))
    key_rows = [0, n // 2, n - 1][:n_keys]
    pairs = [(key_rows[i], key_rows[i + 1]) for i in range(n_keys - 1)]
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1, True))
        edges.append((i + 1, i, True))
    for a, b in pairs:
        edges.append((a, b, True))
        edges.append((b, a, True))
    e_src = [a for a, _b, _c in edges]
    e_dst = [b for _a, b, _c in edges]
    e_attr = np.zeros((len(edges), EDGE_DIM), dtype=np.float64)
    for k, (_a, _b, cov) in enumerate(edges):
        if cov:
            e_attr[k, EDGE_SLOTS["covalent_start"]] = 1.0
            e_attr[k, EDGE_SLOTS["covalent_other"]] = 1.0
    pos0 = pos + rng.normal(scale=0.3, size=pos.shape)
    anchor = (pos - pos.mean(0)) - (pos0 - pos0.mean(0))
    task = np.zeros(TASK_DIM, dtype=np.float64)
    task[0] = 1.0
    return KeyAtomBatch(
        z_atomic=torch.as_tensor(
            rng.integers(1, 10, size=n).astype(np.int64)),
        h_node_inv=torch.as_tensor(
            rng.normal(size=(n, NODE_INV_DIM)), dtype=DTYPE),
        pos=torch.as_tensor(pos, dtype=DTYPE),
        pos0=torch.as_tensor(pos0, dtype=DTYPE),
        anchor_vec=torch.as_tensor(anchor, dtype=DTYPE),
        edge_index=torch.as_tensor(np.stack([e_src, e_dst]),
                                   dtype=torch.int64),
        edge_attr=torch.as_tensor(e_attr, dtype=DTYPE),
        r_ij=torch.as_tensor(
            pos[np.asarray(e_src)] - pos[np.asarray(e_dst)], dtype=DTYPE),
        r0_ij=torch.as_tensor(
            pos0[np.asarray(e_src)] - pos0[np.asarray(e_dst)], dtype=DTYPE),
        task19=torch.as_tensor(task, dtype=DTYPE).unsqueeze(0),
        batch=torch.zeros(n, dtype=torch.int64),
        ptr=torch.tensor([0, n], dtype=torch.int64),
        key_index=torch.as_tensor(key_rows, dtype=torch.int64),
        key_ptr=torch.tensor([0, n_keys], dtype=torch.int64),
        pair_index=torch.as_tensor(np.stack(pairs), dtype=torch.int64),
        pair_edit=torch.as_tensor(
            np.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(pairs))]),
            dtype=DTYPE),
        degK=torch.full((n_keys,), 2.0, dtype=DTYPE),  # deg_scale = 0.04
    )


def test_l_mag_optimizes_effective_magnitude_to_teacher():
    """REGRESSION gate: GD on λ_mag·L_mag ALONE through the REAL
    model↔loss plumbing must drive the EFFECTIVE magnitude
    α = mean(‖u‖/‖y‖) → ~1 and must NOT settle at the crashed
    equilibrium α ≈ deg_scale (=0.04 at degK=2).

    With the pre-fix stash (raw softplus output), the L_mag optimum
    would be raw-s = ‖y‖ ⇒ ‖u‖ = ‖y‖·deg_scale ⇒ α → 0.04 — exactly
    the B1/B3 collapse (alpha_median 0.49→0.04, dead_zone→0.99)."""
    from pes2ts_core.pesn.models.key_atom_egnn import (
        MAG_DEG_SCALE_TYP,
        StarEGNN,
    )
    assert abs(MAG_DEG_SCALE_TYP - 0.04) < 1e-12  # degK_typ=2 fixture
    torch.manual_seed(7)
    model = StarEGNN(magnitude_head=True).to(DTYPE)
    batch = _mag_regression_batch(seed=3)
    n_keys = int(batch.key_index.numel())
    # teacher: uniform 0.15 Å steps (‖y‖ ≫ L_mag gate 0.01, α_init ≈ 1/3)
    y_norm = 0.15
    y = torch.zeros(n_keys, 3, dtype=DTYPE)
    y[:, 0] = y_norm
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    alphas = []
    for _ in range(300):
        u, _z = model(batch)
        out = key_atom_bc_loss(
            u, y, batch.degK, batch.key_ptr, batch.pos, batch.key_index,
            None, config=CFG, s=model.last_magnitudes,
            lambda_mag=1.0)  # only l_mag is BACKPROPPED (terms off)
        opt.zero_grad(set_to_none=True)
        out["l_mag"].backward()
        opt.step()
        alphas.append(
            float((u.detach().norm(dim=-1) / y.norm(dim=-1)).mean()))
    u, _z = model(batch)
    alpha_final = float(
        (u.detach().norm(dim=-1) / y.norm(dim=-1)).mean())
    s_eff = model.last_magnitudes
    # the trained EFFECTIVE magnitude must match the teacher norm
    assert alpha_final > 0.8, f"α converged to {alpha_final}, not ~1"
    # explicitly NOT the crashed raw-s equilibrium (‖u‖ = ‖y‖·deg_scale)
    assert abs(alpha_final - MAG_DEG_SCALE_TYP) > 0.5, (
        f"α settled at the crashed deg_scale equilibrium: {alpha_final}")
    # supervision moved α toward ‖y‖ from the calibrated init (~1/3)
    assert alpha_final > alphas[0] + 0.3, (alphas[0], alpha_final)
    # s_eff (not raw s) is what tracked ‖y‖
    assert float((s_eff - y.norm(dim=-1)).abs().max()) < 0.25 * y_norm
    # raw-vs-effective contract held throughout: s_eff = s_raw·deg_scale
    deg_scale = (model.a_ref / batch.degK.clamp(min=1.0)).to(DTYPE)
    assert torch.allclose(s_eff, model.last_raw_magnitudes * deg_scale,
                          atol=1e-12)
