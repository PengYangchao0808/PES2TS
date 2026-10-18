"""V4.2 physical prior spec tests — synthetic fixtures.

Covers: b/Δs_fit units (Δs_fit in Å·√amu, a_phys in Å), NNLS non-negativity
(max(0,·) clamp), zero/NaN denominator typed error, eps_g/tau_g frozen factor
rules (1e-3 / 1e-6 × median), a_s (non-zero-atom ‖a*‖ median), tau_g zeroing
coverage in ``a_phys_for_batch``, and spec save/load digest round-trip.
"""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.offline.data.direct_preprocess import masses_for
from pes2ts_core.pesn.offline.data.dual_ts_physical_prior import (
    EPS_G_FACTOR,
    PhysicalPriorError,
    PhysicalPriorSpec,
    TAU_G_FACTOR,
    a_phys_for_batch,
    estimate_physical_prior,
    load_physical_prior,
    save_physical_prior,
)


def _record(group, branch, gradient, action, z=(8, 1, 6, 1)):
    return {
        "group_id": group, "branch_id": branch,
        "gradient": np.asarray(gradient, dtype=np.float64),
        "atomic_numbers": np.asarray(z, dtype=int),
        "action": np.asarray(action, dtype=np.float64),
    }


def test_delta_s_fit_units_and_positive_scale():
    # gradient + action aligned -> delta_s_fit > 0 and a_phys in Å.
    z = (8, 1, 6, 1)
    g = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.3], [0.2, 0.0, 0.0]])
    a = 0.5 * g  # action proportional to gradient
    spec = estimate_physical_prior([
        _record("G0", "G0#brA", g, a, z),
        _record("G0", "G0#brB", g, a, z),
    ])
    assert spec.delta_s_fit > 0
    assert np.isfinite(spec.delta_s_fit)
    # a_phys for a single state lands in Å (finite, shape (1,N,3)).
    out = a_phys_for_batch(g[None], np.asarray(z, dtype=int), spec)
    assert out.shape == (1, 4, 3)
    assert np.isfinite(out).all()


def test_nnls_non_negative_clamp():
    z = (8, 1, 6, 1)
    g = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.3], [0.2, 0.0, 0.0]])
    a = -0.3 * g  # anti-aligned -> NNLS would be negative -> clamped to 0
    spec = estimate_physical_prior([_record("G0", "G0#brA", g, a, z)])
    assert spec.delta_s_fit == 0.0


def test_zero_gradient_denominator_error():
    z = (8, 1, 6, 1)
    g = np.zeros((4, 3))
    a = np.zeros((4, 3))
    with pytest.raises(PhysicalPriorError) as excinfo:
        estimate_physical_prior([_record("G0", "G0#brA", g, a, z)])
    assert excinfo.value.code == "NNLS_DENOMINATOR_DEGENERATE"


def test_eps_tau_factor_rules():
    z = (8, 1, 6, 1)
    g = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.3], [0.2, 0.0, 0.0]])
    a = 0.5 * g
    spec = estimate_physical_prior([_record("G0", "G0#brA", g, a, z)])
    # median_gradient_mass_norm == median over states of sqrt(sum||g||^2/m)
    m = masses_for(np.asarray(z, dtype=int))
    mag = np.sqrt(((g * g).sum(axis=1) / m).sum())
    assert spec.median_gradient_mass_norm == pytest.approx(mag)
    assert spec.eps_g == pytest.approx(EPS_G_FACTOR * mag)
    assert spec.tau_g == pytest.approx(TAU_G_FACTOR * mag)


def test_a_s_median_of_nonzero_atom_norms():
    z = (8, 1, 6, 1)
    g = np.eye(4, 3)
    a = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0],
                  [0.0, 0.0, 0.0], [0.0, 0.0, 0.3]])
    spec = estimate_physical_prior([_record("G0", "G0#brA", g, a, z)])
    norms = np.linalg.norm(a, axis=1)
    nonzero = sorted(norms[norms > 0])
    assert spec.a_s == pytest.approx(float(np.median(nonzero)))
    assert spec.n_atoms_nonzero == len(nonzero)


def test_a_phys_below_tau_g_is_zero():
    z = (8, 1, 6, 1)
    g = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.3], [0.2, 0.0, 0.0]])
    a = 0.5 * g
    spec = estimate_physical_prior([_record("G0", "G0#brA", g, a, z)])
    # tiny gradient -> below tau_g -> a_phys zero
    tiny = g * 1e-12
    out = a_phys_for_batch(tiny[None], np.asarray(z, dtype=int), spec)
    assert np.allclose(out, 0.0)
    # above tau_g -> non-zero
    out2 = a_phys_for_batch(g[None], np.asarray(z, dtype=int), spec)
    assert np.abs(out2).sum() > 0.0


def test_mass_shape_mismatch_refuses():
    spec = PhysicalPriorSpec(
        delta_s_fit=1.0, eps_g=1e-4, tau_g=1e-7, a_s=0.1,
        median_mass_weighted_step_length_angstrom=0.1,
        median_gradient_mass_norm=1.0)
    with pytest.raises(PhysicalPriorError):
        a_phys_for_batch(np.zeros((1, 4, 3)), np.asarray((8, 1), dtype=int), spec)


def test_save_load_round_trip(tmp_path):
    z = (8, 1, 6, 1)
    g = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.3], [0.2, 0.0, 0.0]])
    a = 0.5 * g
    spec = estimate_physical_prior([_record("G0", "G0#brA", g, a, z)])
    path = tmp_path / "spec.json"
    digest = save_physical_prior(spec, path)
    loaded = load_physical_prior(path)
    assert loaded.delta_s_fit == spec.delta_s_fit
    assert loaded.eps_g == spec.eps_g
    assert loaded.tau_g == spec.tau_g
    assert loaded.a_s == spec.a_s
    assert digest == save_physical_prior(loaded, tmp_path / "spec2.json")
